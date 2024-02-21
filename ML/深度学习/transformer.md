

## _preface_


Transformer 引入的自注意力机制能够有效 <u>捕捉序列信息中长距离依赖关系</u>，相比于以往的 RNNs, 它在处理长序列时的表现更好



- 序列处理问题，传统的循环神经网络(RNNs)、长短时记忆网络(LSTMs)等模型存在一些限制。 transformer 被设计用来解决这些问题。
- 自注意力机制（关键创新），使得模型在处理序列数据时能够同时关注序列中的不同位置，不必像传统模型一样逐步处理
- 编码器，解码器
- 多头注意力，对注意力机制的扩展，每个头都学习关注输入序列的不同部分，然后这些不同头的输出合并起来
- 其它基本模块：残差连接（允许信息直接通过跳跃连接传递）、层归一化（稳定训练过程）、位置编码


</br>

## _注意力机制_

直接字面意思理解

- Query , 是一个特征向量，描述注意的内容
- Keys, 输入元素键向量
- Values，输入元素值向量
- Score funcion，将查询和键作为输入，输出得分/注意力权重，通常用简单的相似度度量来实现（点积或MLP）



$$ \alpha_i = \frac{exp(f_{attn}(key_i, query))}{\sum_jexp(f_{attn}(key_j, query))}, \quad out = \sum_i\alpha_i \cdot value_i$$


自注意力机制，核心：缩放点积注意力，使序列中的任何元素都可以关注任何其它元素，同时提高效率

Scaled Dot Product Attention 计算方法：

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt d})V$$

Q K V 的矩阵尺寸都是 T*d, T 为序列长度，d 为查询键的维度



-------------

参考资料：
- 学校课程
- [论文原文：Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ⭐️ [中文版](https://blog.csdn.net/longxinchen_ml/article/details/86533005)
- [Transformer从零详细解读(可能是你见过最通俗易懂的讲解)](https://www.bilibili.com/video/BV1Di4y1c7Zm)
- [在线激情讲解transformer&Attention注意力机制](https://www.bilibili.com/video/BV1y44y1e7FW)
- [【Transformer模型】曼妙动画轻松学，形象比喻贼好记](https://www.bilibili.com/video/BV1MY41137AK)
- [手推transformer](https://www.bilibili.com/video/BV1UL411g7aX)
- https://blogs.nvidia.com/blog/2022/03/25/what-is-a-transformer-model/
- https://www.zhihu.com/question/445556653/answer/3254012065
- chatgpt


https://mp.weixin.qq.com/s/gvL6CjQWzhI5hBclBZk2qA

https://jalammar.github.io/illustrated-transformer/

https://zhuanlan.zhihu.com/p/54356280


很多经典的模型比如BERT、GPT-2都是基于Transformer的思想。


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220614001859.png">

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220614001902.png">

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220516160758.png">

$Qk^T$ 得到的会是两个矩阵每个行向量的点乘（用于反映相似度）




```python
class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    nn.init.xavier_uniform_(self.conv_v.weight)
    if proximal_init:
      with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)
      
  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x

  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query /math.sqrt(self.k_channels), key_relative_embeddings)
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores.masked_fill(block_mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    max_relative_position = 2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)
```



## _MHA_


## _LocalAttenation_

```python
import torch
from local_attention import LocalAttention

q = torch.randn(2, 8, 2048, 64)
k = torch.randn(2, 8, 2048, 64)
v = torch.randn(2, 8, 2048, 64)

attn = LocalAttention(
    dim = 64,                # dimension of each head (you need to pass this in for relative positional encoding)
    window_size = 512,       # window size. 512 is optimal, but 256 or 128 yields good enough results
    causal = True,           # auto-regressive or not
    look_backward = 1,       # each window looks at the window before
    look_forward = 0,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
    dropout = 0.1,           # post-attention dropout
    exact_windowsize = False # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
)

mask = torch.ones(2, 2048).bool()
out = attn(q, k, v, mask = mask) # (2, 8, 2048, 64)
```





## _other_

[Attention机制竟有bug?](https://mp.weixin.qq.com/s/cSwWapqFhxu9zafzPUeVEw)


$$(softmax_1(x))_i = \frac{exp(x_i)}{1 + \sum_jexp(x_j)}$$


分母上加 1 将改变注意力单元，不再使用真实的权重概率向量，而是使用加起来小于 1 的权重。其动机是该网络可以学习提供高权重，这样调整后的 softmax 非常接近概率向量。同时有一个新的选项来提供 all-low 权重（它们提供 all-low 输出权重），这意味着它可以选择不对任何事情具有高置信度。



[xFormers](https://github.com/facebookresearch/xformers) - Toolbox to Accelerate Research on Transformers


[Transformer的物理原理](https://mp.weixin.qq.com/s/wUtwfeWAbMCBzRu5pWk2Cg)


