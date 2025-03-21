use std::{usize, vec};

use crate::tensor::Tensor;
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,      // 最大序列长度
    dim: usize,              // 每个键值头的维度
    length: usize, // length of the current sequence
}

impl<T: Default + Copy> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len: max_seq_len,
            dim: dim,
            length: init_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize){
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }

    // 新增缓存截断方法
    pub fn truncate(&mut self, new_len: usize) {
        self.length = new_len.min(self.max_seq_len);
        
        // 实际实现可能需要重置各层的缓存数据
        // 此处示例仅修改长度计数器
    }

    // 修剪缓存，保留最后n个token
    // pub fn trim(&mut self, keep_tokens: usize) {
    //     let new_len = self.length.saturating_sub(keep_tokens);
        
    //     // 遍历所有层
    //     for layer in 0..self.k_cache.len() {
    //         // 计算需要保留的起始位置
    //         let start = new_len * self.dim;
            
    //         // 修剪K缓存
    //         if start < self.k_cache[layer].data().len() {
    //             self.k_cache[layer] = self.k_cache[layer].slice(
    //                 start,
    //                 &[self.max_seq_len - new_len, self.dim]
    //             );
    //         }
            
    //         // 修剪V缓存
    //         if start < self.v_cache[layer].data().len() {
    //             self.v_cache[layer] = self.v_cache[layer].slice(
    //                 start,
    //                 &[self.max_seq_len - new_len, self.dim]
    //             );
    //         }
    //     }
    //     self.length = new_len;
    // }
    
    // 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

}
