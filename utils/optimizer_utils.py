import torch.optim as optim

def optim(optimizer_name):
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(autoencoder.parameters(), lr=model_config['lr'])
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(autoencoder.parameters(), lr=model_config['lr'])
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(autoencoder.parameters(), lr=model_config['lr'])
    else:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_name))