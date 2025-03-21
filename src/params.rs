use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use half::f16;
use byteorder::{LittleEndian, ByteOrder};

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}


impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson, model_type: &str) -> Self {
        // 根据模型类型确定参数前缀
        let (layer_prefix, norm_prefix, embedding_name) = match model_type {
            "chat" => (
                // "transformer.encoder.layers.{}.", 
                // "transformer.encoder.norm.",
                // "transformer.embeddings.word_embeddings.weight"  //不对
                "model.layers.{}.",           // 层前缀
                "model.norm.",                // 归一化层前缀
                "model.embed_tokens.weight"   // 嵌入层名称
            ),
            "story" | _ => (
                "model.layers.{}.", 
                "model.norm.", 
                "lm_head.weight"
            )
        };

        let emb_tensor = safetensor.tensor(embedding_name).unwrap_or_else(|_| panic!("嵌入层 {} 不存在", embedding_name));

        // println!(
        // "词表维度验证: config.vocab_size={}, 实际={}",
        // config.vocab_size,
        // emb_tensor.shape()[0]
        // );

        let get_tensor = |name: &str| {
            let tensor = safetensor.tensor(name).unwrap();
            let data = tensor.data();
            let mut float_data = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                float_data.push(LittleEndian::read_f32(chunk));
            }
            let shape = tensor.shape().to_vec();
            Tensor::new(float_data, &shape)
        };

        let n_layers = config.num_hidden_layers;

        // 初始化所有向量容器
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);

        // 动态生成各层参数名称
        for i in 0..n_layers {
            let current_layer = layer_prefix.replace("{}", &i.to_string());
            
            rms_att_w.push(get_tensor(&format!(
                "{}input_layernorm.weight", current_layer
            )));
            
            wq.push(get_tensor(&format!(
                "{}self_attn.q_proj.weight", current_layer
            )));
            
            // 其他层参数同理生成
            wk.push(get_tensor(&format!(
                "{}self_attn.k_proj.weight", current_layer
            )));
            wv.push(get_tensor(&format!(
                "{}self_attn.v_proj.weight", current_layer
            )));
            wo.push(get_tensor(&format!(
                "{}self_attn.o_proj.weight", current_layer
            )));
            
            rms_ffn_w.push(get_tensor(&format!(
                "{}post_attention_layernorm.weight", current_layer
            )));
            
            w_up.push(get_tensor(&format!(
                "{}mlp.up_proj.weight", current_layer
            )));
            w_gate.push(get_tensor(&format!(
                "{}mlp.gate_proj.weight", current_layer
            )));
            w_down.push(get_tensor(&format!(
                "{}mlp.down_proj.weight", current_layer
            )));
        }

        LLamaParams {
            embedding_table: get_tensor(embedding_name),
            // ... 其他参数不变 ...
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor(&format!("{}weight", norm_prefix)),
            lm_head: get_tensor("lm_head.weight"),
        }

    }
}