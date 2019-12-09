import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torch.functional as F
import torch.autograd as autograd
import data_loader

if __name__ == '__main__':

    max_epochs = 100  # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    batch_size = 64  # Batch size during training
    image_size = 128  # All images will be resized to this size using a transformer.

    is_load = False
    ckpt_path = './CGAN_Samples/checkpoint_iteration_5000.tar'

    # Root directory for dataset
    img_path = '/datasets/img_align_celeba'
    attr_path = './list_attr_celeba.txt'
    samples_path = './CGAN_Samples'
    os.makedirs(samples_path, exist_ok=True)

    latent_dim = 100  # Size of z latent vector (i.e. size of generator input)
    n_channels = 3  # Number of channels in the training images. For color images this is 3
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

    ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
    workers = 0  # Number of workers for dataloader
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create the dataset
    dataset = data_loader.CelebA_Slim(img_path=img_path,
                                      attr_path=attr_path,
                                      transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    classes = dataset.idx2attr
    n_classes = len(classes)

    print('classes:', classes)

    # --------------------------------------------------------------------------------------------------
    # sample_noise
    n_sample = 64  # 生成图片数量
    manualSeed = 999
    torch.manual_seed(manualSeed)
    # manual_seed的作用期很短

    n_condition = 2 ** n_classes

    sample_noise = torch.randn(n_sample, latent_dim, device=device)
    sample_labels = torch.zeros(n_sample, n_classes, dtype=torch.float32, device=device)

    # 生成噪声相同，但标签不同的示例噪声序列
    for i in range(n_sample):
        sample_noise[i] = sample_noise[i - i % n_condition]
        bi = bin(i % n_condition)[2:].zfill(n_classes)
        for j in range(len(bi)):
            if bi[j] == '1':
                sample_labels[i][j] = 1

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    # Pytorch 没有nn.Reshape, 且不推荐使用 Why？？
    class Reshape(nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.shape = args

        def forward(self, x):
            # 自动取得batch维
            return x.view((x.size(0),) + self.shape)
            # 若使用下式，batch的维数只能用-1代指
            # return x.view(self.shape)


    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.latent_class_dim = 10  # 包含分类信息的噪声维数
            # 如果输入c为1维int型,升到2维到且第2维为latent_class_dim
            # self.emb = nn.Embedding(n_classes, self.latent_class_dim)
            # 如果输入c为one-hot，第2维扩张到latent_class_dim
            self.exp = nn.Linear(n_classes, self.latent_class_dim)
            self.main = nn.Sequential(

                nn.Linear(latent_dim + self.latent_class_dim, ngf * 8 * (image_size // 16) ** 2),
                Reshape(ngf * 8, image_size // 16, image_size // 16),
                nn.BatchNorm2d(ngf * 8),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x (image_size//8) x (image_size//8)

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x (image_size//4) x (image_size//4)

                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x (image_size//2) x (image_size//2)

                nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x image_size x image_size
            )

        def forward(self, z, c):
            # c为一维int型
            # cat = torch.cat((z, self.emb(c)), 1)
            # c为one-hot
            cat = torch.cat((z, self.exp(c)), 1)
            return self.main(cat)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            # self.emb = nn.Embedding(n_classes, image_size * image_size)
            self.exp = nn.Linear(n_classes, image_size * image_size)
            self.main = nn.Sequential(
                # input is (nc) x image_size x image_size
                nn.Conv2d(n_channels + 1, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x (image_size//2) x (image_size//2)

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x (image_size//4) x (image_size//4)

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x (image_size//8) x (image_size//8)

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x (image_size//16) x (image_size//16)

                # Reshape(-1, ndf * 8 * (image_size // 16) ** 2),
                Reshape(ndf * 8 * (image_size // 16) ** 2),
                nn.Linear(ndf * 8 * (image_size // 16) ** 2, 1),
                nn.Sigmoid()
            )

        def forward(self, img, c):
            output = self.exp(c)
            output = output.view(c.size(0), 1, image_size, image_size)
            output = torch.cat((img, output), 1)
            output = self.main(output)
            return output


    net_G = Generator(ngpu).to(device)
    net_D = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net_G = nn.DataParallel(net_G, list(range(ngpu)))
        net_D = nn.DataParallel(net_D, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    net_G.apply(weights_init)
    net_D.apply(weights_init)

    print(net_G)
    print(net_D)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # BCE前加softmax，但这样D的输出就没有概率意义了
    # criterion = nn.BCEWithLogitsLoss()

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(beta1, 0.999))


    # define a method to save diagram
    def save_diagram(list_1, list_2,
                     label1="D", label2="G",
                     title="Generator and Discriminator loss During Training",
                     x_label="iterations", y_label="Loss",
                     path=samples_path,
                     name='loss.jpg'
                     ):
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.plot(list_1, label=label1)
        plt.plot(list_2, label=label2)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.savefig(os.path.join(path, name))
        plt.close()


    # Training Loop
    # Lists to keep track of progress
    if is_load:
        print("Loading checkpoint...")

        checkpoint = torch.load(ckpt_path)
        last_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        last_i = checkpoint['last_current_iteration']
        sample_noise = checkpoint['sample_noise']

        list_loss_D = checkpoint['list_loss_D']
        list_loss_G = checkpoint['list_loss_G']

        net_D.load_state_dict(checkpoint['netD_state_dict'])
        net_G.load_state_dict(checkpoint['netG_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        net_D.eval()
        net_G.eval()

    else:
        last_epoch = 0
        iteration = 0

        list_loss_G = []
        list_loss_D = []

    print("Starting Training Loop...")
    for epoch in range(last_epoch, max_epochs):
        # 若读取，重置当前周期，且只执行一次
        str_i = 0
        if is_load:
            str_i = last_i
            is_load = False
        for i, (real_img, real_c) in enumerate(dataloader, str_i):

            # -----------------------------------------------------------
            # Initial batch
            real_img, real_c = real_img.to(device), real_c.to(device)
            real_batch_size = real_img.size(0)
            ones = torch.full((real_batch_size, 1), 1, device=device)
            zeros = torch.full((real_batch_size, 1), 0, device=device)
            noise = torch.randn(real_batch_size, latent_dim, device=device)
            # random label for computer loss
            fake_c = torch.randint(2, (real_batch_size, n_classes), dtype=torch.float32, device=device)

            fake_img = net_G(noise, fake_c)
            # -----------------------------------------------------------
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            net_D.zero_grad()

            # Calculate D loss
            loss_real = criterion(net_D(real_img, real_c), ones)  # loss of real batch
            loss_fake = criterion(net_D(fake_img.detach(), fake_c.detach()), zeros)  # loss of fake batch

            loss_D = (loss_real + loss_fake) * 0.5  # total loss of D

            # Update D
            loss_D.backward()
            optimizer_D.step()

            # -----------------------------------------------------------
            # Update G network: maximize log(D(G(z)))
            net_G.zero_grad()
            # Calculate G loss

            loss_G = criterion(net_D(fake_img, fake_c), ones)

            # Update G
            loss_G.backward()
            optimizer_G.step()

            # -----------------------------------------------------------
            # Output training stats
            with torch.no_grad():
                list_loss_D.append(loss_D.item())
                list_loss_G.append(loss_G.item())

                if i % 100 == 0:
                    print(
                        '[%d/%d][%2d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, max_epochs, i, len(dataloader),
                           loss_D.item(), loss_G.item()))

                # Check how the generator is doing by saving G's output on sample_noise
                if (iteration % 100 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
                    # with torch.no_grad():
                    #     sample = netG(fixed_noise).detach().cpu()
                    samples = net_G(sample_noise, sample_labels).cpu()
                    vutils.save_image(samples, os.path.join(samples_path, '%d.jpg' % iteration), padding=2,
                                      normalize=True)
                    save_diagram(list_loss_D, list_loss_G, name='loss.jpg')

                # Save model
                if (iteration % 5000 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
                    save_path = os.path.join(samples_path, 'checkpoint_iteration_%d.tar' % iteration)
                    torch.save({
                        'epoch': epoch,
                        'iteration': iteration,
                        'last_current_iteration': i,
                        'netD_state_dict': net_D.state_dict(),
                        'netG_state_dict': net_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'list_loss_D': list_loss_D,
                        'list_loss_G': list_loss_G,
                        'sample_noise': sample_noise
                    }, save_path)

            # iteration: total iteration, i: iteration of current epoch
            iteration += 1
