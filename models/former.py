import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


BATCH_NORM_DECAY  =  1 - 0.9  # pytorch batch norm `momentum  =  1 - counterpart` of tensorflow
BATCH_NORM_EPSILON  =  1e-5


def get_act(activation): # 内嵌到BNReLU函数中！！
    """Only supports ReLU and SiLU/Swish."""
    assert activation in ['relu', 'silu']
    if activation  ==  'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()  # TODO: pytorch's nn.Hardswish() v.s. tf.nn.swish


class BNReLU(nn.Module):
    """"""

    def __init__(self, out_channels, activation = 'relu', nonlinearity = True, init_zero = False):
        super(BNReLU, self).__init__()

        self.norm  =  nn.BatchNorm2d(out_channels, momentum = BATCH_NORM_DECAY, eps = BATCH_NORM_EPSILON)
        if nonlinearity:
            self.act  =  get_act(activation)
        else:
            self.act  =  None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out  =  self.norm(input)
        if self.act is not None:
            out  =  self.act(out)
        return out  # 好像没用到？？

class RelPosSelfAttention(nn.Module): #实现了相对位置自注意力机制
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative = True, fold_heads = False):
        super(RelPosSelfAttention, self).__init__()
        self.relative  =  relative
        self.fold_heads  =  fold_heads # 是否折叠多头输出
        self.rel_emb_w  =  nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h  =  nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std = dim ** -0.5)  # 特定的标准差初始化self.rel_emb_w
        nn.init.normal_(self.rel_emb_h, std = dim ** -0.5) # 特定的标准差初始化self.rel_emb_w

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim  =  q.shape
        q  =  q * (dim ** -0.5)  # scaled dot-product
        logits  =  torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits +=  self.relative_logits(q)
        weights  =  torch.reshape(logits, [-1, heads, h, w, h * w])
        weights  =  F.softmax(weights, dim = -1)
        weights  =  torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out  =  torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out  =  torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w  =  self.relative_logits_1d(q, self.rel_emb_w, transpose_mask = [0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h  =  self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask = [0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim  =  q.shape
        rel_logits  =  torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits  =  torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits  =  self.rel_to_abs(rel_logits)
        rel_logits  =  torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits  =  torch.unsqueeze(rel_logits, dim = 3)
        rel_logits  =  rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits  =  rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _  =  x.shape
        col_pad  =  torch.zeros((bs, heads, length, 1), dtype = x.dtype).to(x.device)
        x  =  torch.cat([x, col_pad], dim = 3)
        flat_x  =  torch.reshape(x, [bs, heads, -1]).to(x.device)
        flat_pad  =  torch.zeros((bs, heads, length - 1), dtype = x.dtype).to(x.device)
        flat_x_padded  =  torch.cat([flat_x, flat_pad], dim = 2)
        final_x  =  torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x  =  final_x[:, :, :length, length - 1:]
        return final_x


class AbsPosSelfAttention(nn.Module):

    def __init__(self, W, H, dkh, absolute = True, fold_heads = False):
        super(AbsPosSelfAttention, self).__init__()
        self.absolute  =  absolute
        self.fold_heads  =  fold_heads

        self.emb_w  =  nn.Parameter(torch.Tensor(W, dkh))
        self.emb_h  =  nn.Parameter(torch.Tensor(H, dkh))
        nn.init.normal_(self.emb_w, dkh ** -0.5)
        nn.init.normal_(self.emb_h, dkh ** -0.5)

    def forward(self, q, k, v):
        bs, heads, h, w, dim  =  q.shape
        q  =  q * (dim ** -0.5)  # scaled dot-product
        logits  =  torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        abs_logits  =  self.absolute_logits(q)
        if self.absolute:
            logits +=  abs_logits
        weights  =  torch.reshape(logits, [-1, heads, h, w, h * w])
        weights  =  F.softmax(weights, dim = -1)
        weights  =  torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out  =  torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out  =  torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def absolute_logits(self, q):
        """Compute absolute position enc logits."""
        emb_h  =  self.emb_h[:, None, :]
        emb_w  =  self.emb_w[None, :, :]
        emb  =  emb_h + emb_w
        abs_logits  =  torch.einsum('bhxyd,pqd->bhxypq', q, emb)
        return abs_logits


class GroupPointWise(nn.Module): # 减少模型的参数数量和计算量
    """"""

    def __init__(self, in_channels, heads = 4, proj_factor = 1, target_dimension = None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels  =  target_dimension // proj_factor
        else:
            proj_channels  =  in_channels // proj_factor
        self.w  =  nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)  # proj_channels // heads表示每个head的输出通道
        )

        nn.init.normal_(self.w, std = 0.01) # 用标准差为0.01的正态分布初始化w

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input  =  input.permute(0, 2, 3, 1).to( self.w.dtype )
        # 更改输入的维度顺序，并确保输入的数据类型与权重w的数据类型相匹配.这是因为PyTorch的默认维度顺序是BCHW，而这里的操作可能需要BHWC的顺序。
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out  =  torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class MHSA(nn.Module):
    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type = 'relative', use_pos = True):
        super(MHSA, self).__init__()
        self.q_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)
        self.k_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)
        self.v_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type  ==  'relative':
            self.self_attention  =  RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads = True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q  =  self.q_proj(input)
        k  =  self.k_proj(input)
        v  =  self.v_proj(input)
        o  =  self.self_attention(q = q, k = k, v = v)
        return o