import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from dataset import Maps
import yaml
from model import Generator, Discriminator

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

train_dir = config['dataset']['train_dir']
val_dir = config['dataset']['val_dir']
batch_size = config['dataset']['batch_size']
num_workers = config['dataset']['num_workers']

lr = config['training']['learning_rate']
B1 = config['training']['B1']
B2 = config['training']['B2']
epochs = config['training']['num_epochs']
L1_lambda = config['training']['L1_lambda']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

map_train = Maps(train_dir, transform)
trainloader = torch.utils.data.DataLoader(map_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

map_val = Maps(val_dir, transform)
valloader = torch.utils.data.DataLoader(map_val, batch_size=batch_size, shuffle=False,num_workers=num_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(B1, B2))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr,betas=(B1, B2))
BCE_loss = nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()

for epoch in range(epochs):
    for i, (x,y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        
        discriminator_optimizer.zero_grad()

        # Train with real images
        real_outputs = discriminator(x,y)
        real_labels = torch.ones_like(real_outputs, device=device)
        d_real_loss = BCE_loss(real_outputs, real_labels)

        # Train with fake images        
        fake_images = generator(x)
        fake_outputs = discriminator(fake_images.detach())
        fake_labels = torch.zeros_like(fake_outputs, device=device)
        d_fake_loss = BCE_loss(fake_outputs, fake_labels)

        d_loss = (d_real_loss + d_fake_loss)/2
        d_loss.backward()
        discriminator_optimizer.step()
        
        generator_optimizer.zero_grad()
        gen_labels = torch.ones(batch_size, 1, device=device)
        fake_outputs = discriminator(fake_images)
        g_loss_bce = BCE_loss(fake_outputs, gen_labels)
        l1_loss = L1_loss(fake_images, y)*L1_lambda
        g_loss = g_loss_bce + l1_loss
  
        g_loss.backward()
        generator_optimizer.step()

        # Print losses and log progress
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{epoch}], Step [{i+1}/{len(trainloader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')