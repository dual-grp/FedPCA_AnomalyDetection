import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias']
                            ]
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class DNN2(nn.Module):
    def __init__(self, input_dim = 784, mid_dim_in = 100, mid_dim_out= 100, output_dim = 10):
        super(DNN2, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim_in)
        self.fc2 = nn.Linear(mid_dim_in, mid_dim_out)
        self.fc3 = nn.Linear(mid_dim_out, output_dim)
    
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
#################################
##### Neural Network model #####
#################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, num_classes)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]
                            
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.weight_keys = [['fc1.weight', 'fc1.bias']]

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim = 16):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            #nn.Linear(64, 32),
            #nn.ReLU(True),
            nn.Linear(32, latent_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(True),
            #nn.Linear(32, 64),
            #nn.ReLU(True),
            nn.Linear(32, input_dim),
            nn.Tanh()  # Usually a Tanh activation function is used for the output layer of the decoder in an autoencoder.
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder_2(nn.Module):
    def __init__(self, input_dim, latent_dim = 16):
        super(AutoEncoder_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, latent_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim),
            nn.Tanh()  # Usually a Tanh activation function is used for the output layer of the decoder in an autoencoder.
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder_Ton2(nn.Module):
    def __init__(self, input_dim, latent_dim = 16):
        super(AutoEncoder_Ton2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, latent_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim),
            nn.Tanh()  # Usually a Tanh activation function is used for the output layer of the decoder in an autoencoder.
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder_Ton(nn.Module):
    def __init__(self, input_dim, latent_dim = 16):
        super(AutoEncoder_Ton, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            #nn.Linear(32, 16),
            #nn.ReLU(True),
            #nn.Linear(16, 8),
            #nn.ReLU(True),
            nn.Linear(32, latent_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(True),
            #nn.Linear(8, 16),
            #nn.ReLU(True),
            #nn.Linear(16, 32),
            #nn.ReLU(True),
            nn.Linear(32, input_dim),
            nn.Tanh()  # Usually a Tanh activation function is used for the output layer of the decoder in an autoencoder.
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    

# Define BiGAN model.

class Generator_Ton(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator_Ton, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, output_dim)           
        )

    def forward(self, z):
        return self.model(z)


class Discriminator_Ton(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator_Ton, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Encoder_Ton(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder_Ton, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

# Define the Decoder
class Decoder_Ton(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder_Ton, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim),
        )
        
    def forward(self, x):
        return self.model(x)
    

class BiGAN_Ton(nn.Module):
    def __init__(self, input_dim, latent_dim = 16):
        super(BiGAN_Ton, self).__init__()
        self.generator = Generator_Ton(latent_dim, input_dim)
        self.discriminator = Discriminator_Ton(input_dim + latent_dim)
        self.encoder = Encoder_Ton(input_dim, latent_dim)

    def forward(self, x, z):
        gen_data = self.generator(z)
        enc_data = self.encoder(x)
        disc_real = self.discriminator(torch.cat([x, enc_data], dim=1))
        disc_fake = self.discriminator(torch.cat([gen_data, z], dim=1))
        return disc_real, disc_fake

# BiGAN TON 2
class Generator_Ton2(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator_Ton2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, output_dim)           
        )

    def forward(self, z):
        return self.model(z)


class Discriminator_Ton2(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator_Ton2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Encoder_Ton2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder_Ton2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        
    def forward(self, x):
        return self.model(x)
    

class BiGAN_Ton2(nn.Module):
    def __init__(self, input_dim, latent_dim = 32):
        super(BiGAN_Ton2, self).__init__()
        self.generator = Generator_Ton2(latent_dim, input_dim)
        self.discriminator = Discriminator_Ton2(input_dim + latent_dim)
        self.encoder = Encoder_Ton2(input_dim, latent_dim)

    def forward(self, x, z):
        gen_data = self.generator(z)
        enc_data = self.encoder(x)
        disc_real = self.discriminator(torch.cat([x, enc_data], dim=1))
        disc_fake = self.discriminator(torch.cat([gen_data, z], dim=1))
        return disc_real, disc_fake
    
# BiGAN
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 20),
            nn.ReLU(True),
            nn.Linear(20, output_dim)           
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(True),
            nn.Linear(20, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(True),
            nn.Linear(20, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        
    def forward(self, x):
        return self.model(x)
    

class BiGAN(nn.Module):
    def __init__(self, input_dim, latent_dim = 32):
        super(BiGAN, self).__init__()
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim + latent_dim)
        self.encoder = Encoder(input_dim, latent_dim)

    def forward(self, x, z):
        gen_data = self.generator(z)
        enc_data = self.encoder(x)
        disc_real = self.discriminator(torch.cat([x, enc_data], dim=1))
        disc_fake = self.discriminator(torch.cat([gen_data, z], dim=1))
        return disc_real, disc_fake
    
## BiGAN-UNSW 2
class Generator2(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(True),     
            nn.Linear(16, output_dim)           
        )

    def forward(self, z):
        return self.model(z)


class Discriminator2(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Encoder2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

class BiGAN2(nn.Module):
    def __init__(self, input_dim, latent_dim = 32):
        super(BiGAN, self).__init__()
        self.generator = Generator2(latent_dim, input_dim)
        self.discriminator = Discriminator2(input_dim + latent_dim)
        self.encoder = Encoder2(input_dim, latent_dim)

    def forward(self, x, z):
        gen_data = self.generator(z)
        enc_data = self.encoder(x)
        disc_real = self.discriminator(torch.cat([x, enc_data], dim=1))
        disc_fake = self.discriminator(torch.cat([gen_data, z], dim=1))
        return disc_real, disc_fake
    
# Combine them in ALAD
class ALAD(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ALAD, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        prob = self.discriminator(x_hat)
        return x_hat, prob
