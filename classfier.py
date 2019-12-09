import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time
import data_loader

init_time = time.time()

# paramaters
batch_size = 64

lr = 1e-4

max_epoch = 200
image_size = 64
# attr = [8, 9, 11, 15, 20, 25, 31, 39])

workers = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_path = '/dataset/img_align_celeba'
attr_path = './list_attr_celeba.txt'

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# loading data
train_set = data_loader.CelebA_Slim(img_path=img_path,
                                    attr_path=attr_path,
                                    transform=transform,
                                    slice=[0, 10000])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=workers)

classes = train_set.idx2attr
num_classes = len(classes)

print(num_classes)
print(classes)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
            nn.BatchNorm1d(120),
            nn.LeakyReLU(),

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.LeakyReLU(),

            nn.Linear(84, num_classes),
            nn.Sigmoid()

        )

    def forward(self, input):
        feature = self.conv(input)
        feature = feature.view(-1, 16 * 13 * 13)
        output = self.fc(feature)
        return output


net = Net().to(device)
# net = models.vgg16(num_classes=num_classes).to(device)

print(net)

# criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MultiLabelMarginLoss()
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

def criterion(input, target, esp=1e-19):
    loss = - torch.mean(target * torch.log(input.clamp_min(esp))) - torch.mean(
        (1 - target) * torch.log((1 - input).clamp_min(esp)))
    return loss


optimizer = optim.Adam(net.parameters(), lr=lr)

losses = []
# ---------------------------------------------------------------------------
# Save loss
plt.figure(figsize=(5, 5))
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
# plt.close()

# training
str_time = time.time()
iteration = 0
print("Starting Training Loop...")
print('Initial time:%.3f' % (str_time - init_time))

for epoch in range(max_epoch):
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(images)

        # loss = - torch.mean(labels * torch.log(outputs)) - torch.mean((1 - labels) * torch.log(1 - outputs))
        # loss = - torch.mean(labels * torch.log(outputs + esp)) - torch.mean((1 - labels) * torch.log(1 - outputs + esp))

        # 截断输出，避免出现log(0)
        # loss = - torch.mean(labels * torch.log(outputs.clamp_min(esp))) - torch.mean((1 - labels) * torch.log((1 - outputs).clamp_min(esp)))
        # 或者直接使用mse

        loss = criterion(outputs, labels)

        loss.backward()

        predicted = outputs.detach() > 0.5

        correct = (predicted == labels.type(torch.uint8))

        accuracy = correct.sum().item() / (len(correct) * num_classes)
        # nn.utils.clip_grad_norm(net.parameters(), 1e3, 2)

        optimizer.step()

        losses.append(loss.item())
        if i % 50 == 0 or ((epoch == (max_epoch - 1)) and i == (len(train_loader) - 1)):
            it_time = time.time()
            print('[%d, %5d] loss: %.3f accuracy:%.3f%% time:%.3f' %
                  (epoch, i, loss.item(), accuracy * 100, it_time - str_time))
            plt.plot(losses)
            plt.savefig('./loss.jpg')
            # plt.plotfile('./loss.jpg')

        if (iteration % 1000 == 0) or ((epoch == max_epoch - 1) and (i == len(train_loader) - 1)):
            # Save checkpoint
            save_path = 'checkpoin_CNN.pth.tar'
            torch.save({
                'epoch': epoch,
                'last_current_iteration': i,
                'net_state_dict': net.state_dict(),
                'optimizer_D_state_dict': optimizer.state_dict(),
                'losses': losses,
                'iteration': iteration
            }, save_path)

        iteration += 1

print('Finished Training')

# ---------------------------------------------------------------------------
# Test
test_set = data_loader.CelebA_Slim(img_path=img_path,
                                   attr_path=attr_path,
                                   transform=transform,
                                   slice=[10000, 12000])

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=workers)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)

        predicted = outputs > 0.5

        correct += (predicted == labels.type(torch.uint8)).sum().item()
        total += len(labels) * num_classes

    accuracy = correct / total
    print('Accuracy of all test images: %.3f' % (accuracy * 100))

    # ----------------------------------------------------------------------
    # # use
    # test_num = 8
    # data_iter = iter(test_loader)
    # images, labels = data_iter.next()
    # images, labels = images.to(device), labels.to(device)
    #
    # plt.figure(figsize=(8, 8))
    # plt.imshow(
    #     np.transpose(torchvision.utils.make_grid(images[:test_num], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()
    #
    # # print(labels)
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(test_num)))
    #
    # outputs = net(images)
    #
    # predicted = torch.max(outputs, 1)[1]
    # # print(predicted)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(test_num)))
    #
    # correct = predicted == labels
    #
    # accuracy = correct.sum().item() / len(correct)
    # print('Accuracy of these samples: %.2f' % (accuracy * 100))
