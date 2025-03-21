use crate::tensor::Tensor;
use half::f16;
use std::arch::x86_64::*;
// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}


pub fn rms_norm(
    y: &mut Tensor<f32>, 
    x: &Tensor<f32>, 
    w: &Tensor<f32>, 
    epsilon: f32
) {
    let len = y.size();
    assert!(len == x.size());
    let shape = y.shape().clone();
    let y_data = unsafe { y.data_mut() };
    let x = x.data();
    let w = w.data();
    for i in 0..shape[0] {
        let mut sum = 0.0;
        for j in 0..shape[1] {
            sum = sum + x[i * shape[1] + j] * x[i * shape[1] + j];
        }
        let rms = (sum / shape[1] as f32 + epsilon).sqrt();
        for j in 0..shape[1] {
            let idx = i * shape[1] + j;
            y_data[idx] = w[j] * (x[idx] / rms);
        }
    }
}


// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size(), "输入张量 x 和 y 的长度必须相同");

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for i in 0..len {
        let sigmod_x = 1.0 / (1.0 + (-_x[i]).exp());
        let silu_x = _x[i] * sigmod_x;
        _y[i] = silu_x * _y[i];
    }

    //todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 形状检查
    let (m, k_a) = (a.shape()[0], a.shape()[1]);
    let (n_b, k_b) = (b.shape()[0], b.shape()[1]);
    let (m_c, n_c) = (c.shape()[0], c.shape()[1]);
    
    assert_eq!(k_a, k_b, "A的列数({})必须等于B的列数({})", k_a, k_b);
    assert_eq!(m_c, m, "C的行数({})必须等于A的行数({})", m_c, m);
    assert_eq!(n_c, n_b, "C的列数({})必须等于B的行数({})", n_c, n_b);

    // 获取底层数据切片
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };
    let k = k_a; // 公共维度

    // 遍历C的每个元素
    for i in 0..m {
        for j in 0..n_b {
            // 计算A的第i行和B的第j行的点积
            let mut dot = 0.0;
            for l in 0..k {
                // A的行优先访问
                // B的行优先访问 (等价于B^T的列优先)
                dot += a_data[i * k + l] * b_data[j * k + l];
            }
            
            // 计算最终结果: alpha * dot + beta * c_old
            let idx = i * n_c + j;
            c_data[idx] = alpha * dot + beta * c_data[idx];
        }
    }
    
}


// // SIMD优化
#[target_feature(enable = "avx")]
pub unsafe fn matmul_transb_avx(
    c: &mut Tensor<f32>,
    beta: f32,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    alpha: f32,
) {
    use std::arch::x86_64::*;
    
    // 解包张量维度
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n_b = b.shape()[0];
    let n_c = c.shape()[1];

    // 获取底层数据指针
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() }; 

    // 主循环：遍历A的每一行
    for i in 0..m {
        // 遍历B的每一行（即B^T的每一列）
        for j in 0..n_b {
            let mut dot = 0.0f32;
            let mut l = 0;

            // AVX向量累加器（一次处理8个元素）
            let mut sum_vec = _mm256_setzero_ps();

            // 向量化部分：每次处理8个元素
            while l + 8 <= k {
                // 加载A[i]行的连续8个元素
                let a_ptr = a_data.as_ptr().add(i * k + l);
                let a_vec = _mm256_loadu_ps(a_ptr);

                // 加载B[j]行的连续8个元素 
                let b_ptr = b_data.as_ptr().add(j * k + l);
                let b_vec = _mm256_loadu_ps(b_ptr);

                // 乘积累加：sum_vec += a_vec * b_vec
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a_vec, b_vec));
                
                l += 8;
            }

            // 水平求和：将AVX向量中的8个值相加
            let mut temp = [0.0f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
            dot += temp.iter().sum::<f32>();

            // 标量处理剩余元素（不足8个的部分）
            while l < k {
                dot += a_data[i * k + l] * b_data[j * k + l];
                l += 1;
            }

            // 最终结果计算：C = alpha*A*B^T + beta*C
            let idx = i * n_c + j;
            c_data[idx] = alpha.mul_add(dot, beta * c_data[idx]);
        }
    }
}

// 普通矩阵乘法函数
pub fn matmul(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let m = a.shape()[0];
    let n = b.shape()[1];
    let k = a.shape()[1];

    // 检查输入张量的形状
    assert!(a.shape() == &[m, k], "A 的形状必须是 [m, k]");
    assert!(b.shape() == &[k, n], "B 的形状必须是 [k, n]");
    assert!(c.shape() == &[m, n], "C 的形状必须是 [m, n]");

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            c_data[i * n + j] = beta * c_data[i * n + j] + alpha * sum;
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_matmul_consistency() {
    // 测试用例1：小型矩阵
    let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &vec![2, 2]);
    let b = Tensor::<f32>::new(vec![5.0, 6.0, 7.0, 8.0], &vec![2, 2]);
    let mut c1 = Tensor::<f32>::new(vec![0.0, 0.0, 0.0, 0.0], &vec![2, 2]);
    let mut c2 = Tensor::<f32>::new(vec![0.0, 0.0, 0.0, 0.0], &vec![2, 2]);
    
    matmul_transb(&mut c1, 0.0, &a, &b, 1.0);
    //matmul_transb_avx(&mut c2, 0.0, &a, &b, 1.0);
    unsafe { matmul_transb_avx(&mut c2, 0.0, &a, &b, 1.0) };
    assert!(c1.approx_eq(&c2, 1e-6));

    // 测试用例2：非对齐数据
    let a = Tensor::<f32>::new(vec![1.0; 7], &vec![1, 7]);
    let b = Tensor::<f32>::new(vec![1.0; 7], &vec![1, 7]);
    let mut c1 = Tensor::<f32>::new(vec![0.0, 0.0, 0.0, 0.0], &vec![1, 1]);
    let mut c2 = Tensor::<f32>::new(vec![0.0, 0.0, 0.0, 0.0], &vec![1, 1]);

    matmul_transb(&mut c1, 0.0, &a, &b, 1.0);
    unsafe { matmul_transb_avx(&mut c2, 0.0, &a, &b, 1.0) };
    assert!(c1.approx_eq(&c2, 1e-6));
}