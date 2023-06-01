def get_summary(model):
    """
    Prints model memory
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_kb = ((param_size + buffer_size) / 1024**2)*1000
    print('Model size: {:.1f}KB'.format(size_all_kb))

    # Count the number of layers
    num_layers = sum(1 for _ in model.modules()) - 1
    print(f"Number of layers: {num_layers}")

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
