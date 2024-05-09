# xLSTM
PyTorch implementation of the xLSTM module found in [this](https://arxiv.org/abs/2405.04517) paper. I did not contribute to the cited paper.

As is noted in the paper, it is preferable to write a custom CUDA kernel for this module; the appendix of the paper outlines a way to compute the forward pass in parallel instead of sequentially. Its sequential implementation through the Python API, even when compiled with TorchScript, is slow.
