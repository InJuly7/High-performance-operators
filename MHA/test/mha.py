import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model  # 模型维度，如512
        self.num_heads = num_heads  # 头数，如8
        self.d_k = d_model // num_heads  # 每个头的维度，如64
        
        # 一个大的权重矩阵，不是多个小矩阵
        self.W_q = nn.Linear(d_model, d_model)  # 512 -> 512
        self.W_k = nn.Linear(d_model, d_model)  # 512 -> 512
        self.W_v = nn.Linear(d_model, d_model)  # 512 -> 512
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # 1. 通过大权重矩阵计算 Q, K, V
        Q = self.W_q(query)  # [batch, seq_len, d_model]
        K = self.W_k(key)    # [batch, seq_len, d_model]
        V = self.W_v(value)  # [batch, seq_len, d_model]
        
        # 2. 将大矩阵分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 形状变为: [batch, num_heads, seq_len, d_k]
        
        # 3. 计算缩放点积注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # [batch, num_heads, seq_len, d_k]
        
        # 4. 连接所有头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        # [batch, seq_len, d_model]
        
        # 5. 输出投影
        output = self.W_o(attention_output)
        return output
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: [batch, num_heads, seq_len, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output

# 使用示例
d_model = 512
num_heads = 8
seq_len = 1024
batch_size = 1

mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)
output = mha(x, x, x)  # self-attention
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")