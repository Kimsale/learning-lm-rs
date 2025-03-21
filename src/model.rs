use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;

use crate::operators::rms_norm;
use crate::operators::matmul_transb;
use crate::operators::swiglu;

use tokenizers::Tokenizer;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

// 在model模块顶部添加类型定义
#[derive(Debug, PartialEq)]
pub enum ModelType {
    Story,
    Chat
}

/*LLama-7B
n_q_h: 8      // Q头数
n_kv_h: 4      // KV头数
d: 128        // 隐藏层维度
dqkv: 128      // 每个注意力头维度 (8头*128=1024)
di: 384      // FFN中间维度 (通常为4*d)
vocab: 2048   // 词表大小 */
impl Llama<f32> {

    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        // 检测模型类型 ---------------------------------------------------
        let model_type = if model_dir.as_ref().ends_with("chat") {
            ModelType::Chat
        } else {
            ModelType::Story
        };
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        // 根据模型类型验证关键参数
        match model_type {
            ModelType::Story => {
                assert_eq!(config.hidden_size, 128, "Story模型隐藏层维度应为128");
                assert_eq!(config.num_attention_heads, 8, "Story模型注意力头数应为8");
            },
            ModelType::Chat => {
                assert_eq!(config.hidden_size, 312, "Chat模型隐藏层维度应为2048");
                assert_eq!(config.num_key_value_heads, 4, "Chat模型KV头数应为4");
            }
        }
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        // let params = LLamaParams::from_safetensors(&safetensor, &config);
        // 参数加载时传递模型类型
        // 将ModelType转换为字符串表示
        let model_type_str = match model_type {
            ModelType::Chat => "chat",
            ModelType::Story => "story",
        };
        // let params = LLamaParams::from_safetensors(&safetensor, &config, model_type);
        // 参数加载时传递字符串类型
        let params = LLamaParams::from_safetensors(&safetensor, &config, model_type_str);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();     //用户输入的token数
        let past_seq_len = cache.len(); //历史上下文长度
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;
        assert!(
            self.n_q_h % self.n_kv_h == 0,
            "Q头数{}必须能被KV头数{}整除",
            self.n_q_h,
            self.n_kv_h
        );

        // println!(
        //     "输入维度检查: input={:?}, embedding_table={:?}, lm_head={:?}",
        //     input.shape(),
        //     self.params.embedding_table.shape(),
        //     self.params.lm_head.shape()
        // );

        // Some pre-allocated buffers that will be reused
        // 预分配缓冲区（复用内存提升性能）
        // residual: 残差连接中间结果 [seq_len, d]
        // hidden_states: 当前隐藏状态 [seq_len, d]
        // q_buf: 查询向量缓存 [seq_len, n_q_h*dqkv]
        // att_scores: 注意力分数 [n_kv_h, n_groups, seq_len, total_seq_len]
        // gate_buf: SwiGLU门控中间结果 [seq_len, di]
        // up_buf: FFN上分支中间结果 [seq_len, di]
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        // 1. 词嵌入查找（将离散token转换为连续向量）
        // input形状 [seq_len]，embedding_table形状 [vocab, d]
        // 结果 residual 形状变为 [seq_len, d]
        // 打印关键权重形状
        //println!("embedding_table 形状: {:?}", self.params.embedding_table.shape());
        //println!("lm_head 形状: {:?}", self.params.lm_head.shape());
        // 预期: embedding_table [vocab_size, d], lm_head [vocab_size, d]
        // 输出：embedding_table 形状: [2048, 128]
        // lm_head 形状: [2048, 128]

        assert_eq!(
            self.params.embedding_table.shape()[0], 
            self.vocab,
            "词表维度不匹配: 配置{} vs 实际{}",
            self.vocab,
            self.params.embedding_table.shape()[0]
        );

        OP::gather(&mut residual, input, &self.params.embedding_table);
        //println!("Embedding 后 residual 形状: {:?}", residual.shape());

        for layer in 0..self.n_layers {
            // 2. RMS归一化（Pre-Norm结构）
            // 输入 residual [seq_len, d]，输出 hidden_states [seq_len, d]
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );
            //println!("RMSNorm 后 hidden_states 形状: {:?}", hidden_states.shape());

            //println!("RMS归一化完成");

            // 3. 生成QKV向量
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            // 计算 Q = hidden_states * WQ^T
            // hidden_states [seq_len, d], wq [n_q_h*dqkv, d]
            // 使用matmul_transb实现矩阵乘（自动转置WQ）
            //OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            unsafe { OP::matmul_transb_avx(q, 0., &hidden_states, &self.params.wq[layer], 1.0) };
            unsafe { OP::matmul_transb_avx(k, 0., &hidden_states, &self.params.wk[layer], 1.0) };
            unsafe { OP::matmul_transb_avx(v, 0., &hidden_states, &self.params.wv[layer], 1.0) };

            // OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            // OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            // OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            
            // 4. 应用RoPE位置编码
            // 将q_buf重组为三维张量 [seq_len, n_q_h, dqkv]
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            //println!("RoPE 后 Q 形状: {:?}", q.shape());
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            //println!("RoPE 后 KV 形状: {:?}", k.shape());

            // 5. 构建完整KV缓存
            // full_k指向该层缓存的起点 [total_seq_len, n_kv_h*dqkv]
            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            /* todo!("self_attention(...)");
            todo!("down_proj matmul and add residual");

            todo!("mlp(...)"); */

            // Self-Attention
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            //println!("Self-Attention 后 hidden_states 形状: {:?}", hidden_states.shape());
            //println!("Self-Attention完成");


            // Down_proj matmul and add residual
            /*hidden_states: [seq_len, n_q_h*dqkv]
            wo: [d, n_q_hdqkv] （实际存储为 [n_q_hdqkv, d]，转置后使用）
            residual: [seq_len, d] 
            residual = residual + hidden_states * WO^T*/
            OP::matmul_transb(
                &mut residual,
                1.0,            // β=1表示保留原始残差值
                &hidden_states,
                &self.params.wo[layer], // 下投影矩阵 [d, n_q_h*dqkv]
                1.0,
            );
            //println!("Down Projection 后 residual 形状: {:?}", residual.shape());

            // MLP
            /*典型SwiGLU计算流程：
            ​上分支：up = RMSNorm(x) * W_up
            ​门控分支：gate = RMSNorm(x) * W_gate
            ​激活：gate = silu(gate)
            ​混合：hidden = up ⊙ gate（逐元素乘）
            ​下投影：out = hidden * W_down
            ​残差连接：residual += out

            参数维度：
            w_up: [di, d]
            w_gate: [di, d]
            w_down: [d, di] */
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
            //println!("MLP 后 residual 形状: {:?}", residual.shape());

            //println!("mlp完成");

        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        // 取序列最后一个token的隐藏状态
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);
        //println!("RMSNorm 的 hidden_states 形状: {:?}", hidden_states.shape());
        //println!("RMSNorm 的 residual 形状: {:?}", residual.shape());


        // 最终RMSNorm
        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        //println!("最终RMSNorm完成");

        // 生成词表logits
        OP::matmul_transb(
            &mut logits, 
            0., 
            &hidden_states, 
            &self.params.lm_head, 
            1.0
        );

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = vec![self.bos_token_id];
        result.extend_from_slice(token_ids);           // 将输入的 token_ids 切片中的元素追加到 result 向量的末尾。
        let mut cache = self.new_cache();              // 初始化一个与模型计算相关的缓存结构，以便后续重用。

        let input_tensor = Tensor::new(result.clone(), &vec![result.len()]);
        let logits = self.forward(&input_tensor, &mut cache);

        let mut next_token = OP::random_sample(&logits, top_p, top_k, temperature);
        result.push(next_token);

        while result.len() < max_len {
            let input_tensor = Tensor::new(vec![next_token], &vec![1]);
            let logits = self.forward(&input_tensor, &mut cache);

            next_token = OP::random_sample(&logits, top_p, top_k, temperature);

            if next_token == self.eos_token_id {
                break;
            }

            result.push(next_token);
        }

        result
    }


    pub fn generate_cache(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        cache: &mut KVCache<f32>, // 改为传入可变缓存
    ) -> Vec<u32> {
        // 移除 let mut cache = self.new_cache();
        let mut result = vec![self.bos_token_id];
        result.extend_from_slice(token_ids);           // 将输入的 token_ids 切片中的元素追加到 result 向量的末尾。

        let input_tensor = Tensor::new(result.clone(), &vec![result.len()]);
        // let logits = self.forward(&input_tensor, &mut cache);
        let logits = self.forward(&input_tensor, cache);

        let mut next_token = OP::random_sample(&logits, top_p, top_k, temperature);
        result.push(next_token);

        while result.len() < max_len {
            let input_tensor = Tensor::new(vec![next_token], &vec![1]);
            // let logits = self.forward(&input_tensor, &mut cache);
            let logits = self.forward(&input_tensor, cache);

            next_token = OP::random_sample(&logits, top_p, top_k, temperature);

            if next_token == self.eos_token_id {
                break;
            }

            result.push(next_token);
        }

        result
    }


}


fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let scale = (dqkv as f32).sqrt();
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let att_data = unsafe { att_scores.data_mut() };

    // Q 和 K 的维度被分组和分头打散了，不能直接用矩阵乘
    
    // 1. score = Q @ K.T / sqrt(dim)
    // 用手动索引 + 向量乘
    for kv_head in 0..n_kv_h {                      // 遍历每个键/值头
        for group in 0..n_groups {                  // 遍历每个查询组
            for q_seq in 0..seq_len {               // 遍历输入序列
                for k_seq in 0..total_seq_len {     // 遍历所有的序列
                    let mut dot_product = 0.0;
                    for d in 0..dqkv {              // 遍历单个头维度
                        // q_seq 表示输入的序列位置，共有n_kv_h*n_groups个查询头，每个输入有n_groups个查询组，每个键头被 n_groups 个查询组共享，每个键头有 dqkv 个维度
                        let q_idx = q_seq * (n_kv_h * n_groups * dqkv)    // 当前序列位置的起始地址
                            + (kv_head * n_groups + group) * dqkv         // 计算当前查询头的全局索引， * dqkv 的操作是为了 ​跳过一个完整注意力头的内存块，确保索引指向当前头的起始位置。
                            + d;                                          // 当前维度
                        let k_idx = k_seq * (n_kv_h * dqkv) + kv_head * dqkv + d; // k_seq 表示键的序列位置（行），每个序列位置包含 n_kv_h 个键头，每个头有 dqkv 个特征。
                        dot_product = dot_product + q_data[q_idx] * k_data[k_idx];
                    }
                    let score_idx = kv_head * (n_groups * seq_len * total_seq_len)
                        + group * (seq_len * total_seq_len)
                        + q_seq * total_seq_len
                        + k_seq;
                    att_data[score_idx] = dot_product / scale;
                }
            }
        }
    }

    // 2. attn = softmax(score)
    OP::masked_softmax(att_scores);

    let att_data = att_scores.data();
    // 输出 hidden_data 的初始化
    let hidden_data = unsafe { hidden_states.data_mut() };
    for i in 0..hidden_data.len() {
        hidden_data[i] = 0.0;
    }

    // 3. attn_V = attn @ V
    // 用手动索引 + 向量乘
    for kv_head in 0..n_kv_h {
        for group in 0..n_groups {
            for q_seq in 0..seq_len {
                for d in 0..dqkv {
                    let mut sum = 0.0;
                    for k_seq in 0..total_seq_len {
                        let score_idx = kv_head * (n_groups * seq_len * total_seq_len)
                            + group * (seq_len * total_seq_len)
                            + q_seq * total_seq_len
                            + k_seq;
                        let v_idx = k_seq * (n_kv_h * dqkv) + kv_head * dqkv + d;
                        sum = sum + att_data[score_idx] * v_data[v_idx];
                    }
                    let out_idx = q_seq * (n_kv_h * n_groups * dqkv)
                        + (kv_head * n_groups + group) * dqkv
                        + d;
                    hidden_data[out_idx] = sum;
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    //todo!("Implement mlp");
    // Step 1: RMS Normalization
    rms_norm(hidden_states, &residual, &rms_w, eps);

    // Step 2: Matrix Multiplication for gate
    matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);

    // Step 3: Matrix Multiplication for up
    matmul_transb(up, 0.0, hidden_states, w_up, 1.0);
    
    //swiglu(&mut up, &gate);
    // Step 4: SwiGLU Activation
    //let mut act = Tensor::new(vec![0.0; gate.size()], gate.shape());
    swiglu(up, gate);

    // Step 5: Matrix Multiplication for output
    //let mut output = Tensor::new(vec![0.0; up.size()], up.shape());
    matmul_transb(hidden_states, 0.0, up, w_down, 1.0);

    // Step 6: Residual Connection
    for i in 0..residual.size() {
        unsafe{
            residual.data_mut()[i] += hidden_states.data()[i];
        }
    }

}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    //println!("Model directory: {}", model_dir.display());
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}


/* #[test]
mod tests_generate_text {
    use super::*;

    #[test]
    fn test_text_generation() {
        let prompt = "Once upon a time";
        let result = text_generation(prompt);
        // 检查生成的文本是否不为空
        assert!(!result.is_empty());
    }
} */


/* #[test]
pub fn test_cache_management() {
    let model = Llama::from_safetensors("models/chat");
    let mut session = model.new_chat_session();
    
    // 第一轮对话
    let _ = model.chat(&mut session, "你好", 100, 0.9, 50, 0.7);
    assert_eq!(session.cache.len(), 100 + 2); // 输入+生成长度
    
    // 第二轮对话触发缓存修剪
    session.max_cache_tokens = 150; // 设置较小值测试
    let _ = model.chat(&mut session, "请解释量子力学", 200, 0.9, 50, 0.7);
    assert!(session.cache.len() <= 150);
    assert!(session.history.len() < 4); // 历史记录被修剪
} */