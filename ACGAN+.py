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

    max_epochs = 200  # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    batch_size = 128  # Batch size during training
    image_size = 128  # All images will be resized to this size using a transformer.

    n_critic = 5
    lam_gp = 10

    is_load = False
    ckpt_path = './ACGAN+_Samples/checkpoint_iteration_20000.tar'

    # Root directory for dataset
    img_path = '/datasets/img_align_celeba'
    attr_path = './list_attr_celeba.txt'
    samples_path = './ACGAN+_Samples'
    os.makedirs(samples_path, exist_ok=True)

    latent_dim = 100  # Size of z latent vector (i.e. size of generator input)
    n_channels = 3  # Number of channels in the training images. For color images this is 3
    ngf = 128  # Size of feature maps in generator
    ndf = 128  # Size of feature maps in discriminator
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
                                      transform=transform,
                                      slice=[0, -1])

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
        # print(i % n_condition)
        bi = bin(i % n_condition)[2:].zfill(n_classes)
        # print(bi)
        for j in range(len(bi)):
            if bi[j] == '1':
                sample_labels[i][j] = 1
    # --------------------------------------------------------------------------------------------------
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
            return self.main(torch.cat((z, self.exp(c)), 1))


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x image_size x image_size
                nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x (image_size//2) x (image_size//2)

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.LayerNorm([ndf * 2, image_size // 4, image_size // 4]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x (image_size//4) x (image_size//4)

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.LayerNorm([ndf * 4, image_size // 8, image_size // 8]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x (image_size//8) x (image_size//8)

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.LayerNorm([ndf * 8, image_size // 16, image_size // 16]),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x (image_size//16) x (image_size//16)
                Reshape(ndf * 8 * (image_size // 16) ** 2),
            )

            self.adv = nn.Sequential(
                nn.Linear(ndf * 8 * (image_size // 16) ** 2, 1),
                # 注意没有WGAN-GP没有nn.Sigmoid()
            )

            self.aux = nn.Sequential(
                nn.Linear(ndf * 8 * (image_size // 16) ** 2, n_classes),
                # nn.Softmax(1)
                # 不能使用Softmax，因为各标签不相关，和也不一定为1
                nn.Sigmoid()
            )

        def forward(self, input):
            feature = self.main(input)
            v = self.adv(feature)
            c = self.aux(feature)
            return v, c


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

    # Initialize Loss function
    # BCE = nn.BCELoss()

    def MBCE(input, target, esp=1e-19):
        loss = - torch.mean(target * torch.log(input.clamp_min(esp))) - torch.mean(
            (1 - target) * torch.log((1 - input).clamp_min(esp)))
        return loss

    # MSE = nn.MSELoss()


    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(beta1, 0.999))


    def gradient_penalty(x, y, f):
        # interpolation
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = torch.rand(shape, device=device)
        z = x + alpha * (y - x)

        # gradient penalty
        # z = Variable(z, requires_grad=True).to(device)
        # z = z.to(device)
        z.requires_grad = True
        o = f(z)[0]
        ones = torch.ones(o.size(), device=device)
        g = autograd.grad(o, z, grad_outputs=ones, create_graph=True)[0].view(z.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

        return gp


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

        loss_D = list_loss_D[-1]
        loss_G = list_loss_G[-1]

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
        for i, (real_img, real_c) in enumerate(dataloader, str_i):

            # -----------------------------------------------------------
            # Initial batch
            real_img, real_c = real_img.to(device), real_c.to(device)
            real_batch_size = real_img.size(0)
            noise = torch.randn(real_batch_size, latent_dim, device=device)
            # random label for computer loss
            fake_c = torch.randint(2, (real_batch_size, n_classes), dtype=torch.float32, device=device)

            fake_img = net_G(noise, fake_c)

            # -----------------------------------------------------------
            # Update D network: minimize: -(D(x) - D(G(z)))+ lambda_gp * gp + class_loss
            net_D.zero_grad()

            v, c = net_D(real_img)
            loss_real = (- torch.mean(v) + MBCE(c, real_c)) * 0.5
            v, c = net_D(fake_img.detach())
            loss_fake = (torch.mean(v) + MBCE(c, fake_c)) * 0.5
            gp = gradient_penalty(real_img.detach(), fake_img.detach(), net_D)
            loss_D = (loss_real + loss_fake) * 0.5 + lam_gp * gp  # total loss of D

            # Update D
            loss_D.backward()
            optimizer_D.step()

            # -----------------------------------------------------------
            # Update G network: maximize D(G(z)) , equal to minimize - D(G(z))
            if i % n_critic == 0:
                net_G.zero_grad()

                # Calculate G loss
                v, c = net_D(fake_img)

                loss_G = (- torch.mean(v) + MBCE(c, fake_c)) * 0.5

                # Update G
                loss_G.backward()
                optimizer_G.step()

            # -----------------------------------------------------------
            # Output training stats
            with torch.no_grad():
                list_loss_D.append(loss_D.item())

                if type(loss_G) == float:
                    list_loss_G.append(loss_G)
                else:
                    list_loss_G.append(loss_G.item())

                if i % 100 == 0 or is_load:
                    print(
                        '[%d/%d][%2d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, max_epochs, i, len(dataloader),
                           list_loss_D[-1], list_loss_G[-1]))

                # Check how the generator is doing by saving G's output on sample_noise
                if (iteration % 500 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
                    # with torch.no_grad():
                    #     sample = netG(fixed_noise).detach().cpu()
                    samples = net_G(sample_noise, sample_labels).cpu()
                    vutils.save_image(samples, os.path.join(samples_path, '%d.jpg' % iteration), padding=2,
                                      normalize=True)
                    save_diagram(list_loss_D, list_loss_G, name='loss.jpg')

                # Save model
                if (iteration % 10000 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
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
            is_load = False


