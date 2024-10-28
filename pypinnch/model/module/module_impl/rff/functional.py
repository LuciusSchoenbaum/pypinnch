


import torch
from torch import Tensor



@torch.jit.script
def basic_encoding(
        v: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{v}} , \sin{2 \pi \mathbf{v}})`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{input_size})`

    """
    pi = torch.pi
    vp = 2*pi*v
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@torch.jit.script
def positional_encoding(
        v: Tensor,
        sigma: float,
        m: int) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)`
        where :math:`j \in \{0, \dots, m-1\}`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        sigma (float): constant chosen based upon the domain of :attr:`v`
        m (int): [description]

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`

    """
    pi = torch.pi
    j = torch.arange(m, device=v.device)
    coeffs = 2*pi*sigma**(j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)




@torch.jit.script
def gaussian_encoding(
        v: Tensor,
        b: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`

    """
    pi = torch.pi
    vp = 2*pi* (v @ b.T)
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)



