import torch, pdb
import torch.nn as nn

class LPFOnline(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        tau_mem: float = 10,
        tau_syn: float = 5,
        initial_scale: float = 1.,
        train_scale: bool = False
    ):
        super().__init__()
        self.scale_factor = torch.nn.parameter.Parameter(
            torch.tensor(initial_scale), 
            requires_grad=train_scale
        )

        syn_kernel = (
            torch.exp(-torch.arange(kernel_size) / tau_syn).unsqueeze(0).unsqueeze(0)
        )
        mem_kernel = (
            torch.exp(-torch.arange(kernel_size) / tau_mem).unsqueeze(0).unsqueeze(0)
        )

        # "Padding" only at beginning of syn_kernel.
        padding = torch.zeros_like(syn_kernel)
        syn_kernel = torch.cat((padding, syn_kernel), -1)
        kernel = torch.nn.functional.conv1d(syn_kernel, mem_kernel.flip(-1))[..., :-1]
        self.pad_size = kernel.shape[-1] - 1
        self.conv = torch.nn.Conv1d(
            num_channels,
            num_channels,
            kernel.shape[-1],
            bias=False,
            groups=num_channels,
        )
        self.conv.weight.data = kernel.flip(-1).repeat(num_channels, 1, 1)
        self.conv.weight.requires_grad_(False)
        self.register_buffer("past_inputs", torch.zeros(1))

    def shift_past_inputs(self, shift_amount: int, assign: torch.Tensor):
        # Shift the past_inputs by N along the last axis (T)
        self.past_inputs = torch.roll(self.past_inputs, -shift_amount, -1)
        self.past_inputs[..., -shift_amount:] = assign

    def reset_past(self, shape=None):
        shape = shape or self.past_inputs.shape
        self.past_inputs = torch.zeros(shape, device=self.conv.weight.device)

    def forward(self, x):
        #pdb.set_trace()
        # Expect (N=275, ...., T=100)
        original_shape = x.shape
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if (shape := x.shape[:-1]) != self.past_inputs.shape[:-1]:
            self.reset_past(shape=(*shape, self.pad_size))

        # Padding
        padded = torch.cat((self.past_inputs, x), -1)
        convd = self.conv(padded) * self.scale_factor
        self.shift_past_inputs(x.shape[-1], x[..., -self.pad_size:].data)
        return convd.reshape(*original_shape)
