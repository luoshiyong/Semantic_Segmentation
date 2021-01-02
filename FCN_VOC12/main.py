import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from PIL import Image
from torchvision import transforms,utils
import torch.nn.functional as  F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable as Variable
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
voc_root = './segment_voc2012/'

def read_images(root=voc_root, train=True):
    txt_name = root + ('train.txt' if train else 'val.txt')
    with open(txt_name, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'image', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label
train_data,train_label = read_images()
random_data = Image.open(train_data[1])
random_data

random_label = Image.open(train_label[1])
random_label.size

def getsize(img,crop_size):
    start_x = np.random.randint(low=0,high=(img.size[0]-crop_size[1]+1))
    end_x = start_x + crop_size[1]
    start_y = np.random.randint(low=0,high=(img.size[0]-crop_size[1]+1))
    end_y = start_y + crop_size[0]
    crop = (start_y,start_x,end_y,end_x) # y1,x1,y2,x2
    return crop
c_size= getsize(random_label,(300,300))
print(c_size)
img = random_data.crop(c_size)
label = random_label.crop(c_size)


# RGB color for each class
classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

print("the len of class and colormap = ",len(classes), len(colormap))


color2int = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for idx,color in enumerate(colormap):
    color2int[(color[0]*256+color[1])*256+color[2]] = idx # 建立索引

def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(color2int[idx], dtype='int64') # 根据索引得到 label 矩阵
label_im = Image.open('./segment_voc2012/SegmentationClass/2007_000032.png').convert('RGB')

label = image2label(label_im)
label[150:160, 240:250]

def img_transforms(im, label, c_size):
    real_size= getsize(random_label,c_size)
    im = im.crop(real_size)
    label = label.crop(real_size)
    im_tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    im = im_tfs(im)
    label = image2label(label)
    label = torch.from_numpy(label)
    return im, label
class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''
    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')
        
    def _filter(self, images): # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and 
                                        Image.open(im).size[0] >= self.crop_size[1])]
        
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size)
        return img, label
    
    def __len__(self):
        return len(self.data_list)

# 实例化数据集
input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape, img_transforms)
voc_test = VOCSegDataset(False, input_shape, img_transforms)

train_loader = DataLoader(voc_train, 2, shuffle=True)
valid_loader = DataLoader(voc_test, 2)

# 定义 bilinear kernel
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

x = Image.open('./segment_voc2012/image/2007_000032.jpg')
x = np.array(x)
plt.imshow(x)
print(x.shape)

x = torch.from_numpy(x.astype('float32')).permute(2, 0, 1).unsqueeze(0)
# 定义转置卷积
conv_trans = nn.ConvTranspose2d(3, 3, 4, 2, 1)
# 将其定义为 bilinear kernel
conv_trans.weight.data = bilinear_kernel(3, 3, 4)

y = conv_trans(Variable(x)).data.squeeze().permute(1, 2, 0).numpy()
plt.imshow(y.astype('uint8'))
print(y.shape)

pretrained_net = models.vgg16_bn(pretrained=True)
num_classes = len(classes)
feature = pretrained_net.features

#定义模型
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        self.stage1 = pretrained_net.features[:7] # 第一段
        self.stage2 = pretrained_net.features[7:14] # 第二段
        self.stage3 = pretrained_net.features[14:24] # 第三段
        self.stage4 = pretrained_net.features[24:34] # 第四段
        self.stage5 = pretrained_net.features[34:] # 第五段
        
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        self.conv_trans1 = nn.Conv2d(512,256,1)
        self.conv_trans2 = nn.Conv2d(256,num_classes,1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) # 使用双线性 kernel
        
        self.upsample_2x_1 = nn.ConvTranspose2d(512, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x_1.weight.data = bilinear_kernel(512,512,4)
        self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False) 
        self.upsample_2x_2.weight.data = bilinear_kernel(256, 256, 4) # 使用双线性 kernel

        
    def forward(self, x):
       
        x = self.stage1(x)  #
        x = self.stage2(x)  #1/4
        s2 = x # 1/16
        
        x = self.stage3(x)   #1/8
        s3 = x # 1/32
        
        x = self.stage4(x)   #1/16
        s4 = x # 1/32
        
        x = self.stage5(x)   #1/32
        s5 = x # 1/32
        #此处经过32倍上采样即可得FCN32s
        # score1 = self.scores1(s5)  #用作预测
        
        s5_x2 = self.upsample_2x_1(s5)
        add1 = s5_x2 + s4 
        #此处经过16倍上采样即可得FCN16s
        score2 = self.scores2(add1)
        
        add1 = self.conv_trans1(add1)#转换通道
        add1 = self.upsample_2x_2(add1)
        add2 = add1 + s3
        add2 = self.conv_trans2(add2)
        score3 = self.upsample_8x(add2)
       
        return score3

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    
    """
    #print("label_true.shape",label_trues.shape)
    #print("label_preds.shape",label_preds.shape)
    #print("n_class = ",n_class)
    #print("label_true = ",label_trues)
    #print("label_preds",label_preds)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        #print("lt,lp+size = ",lt.shape,lp.shape)
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)
#记录过程中参数
train_loss1 = []
val_loss1 = []
train_miou1 = []
val_miou1 = []
train_acc1 = []
val_acc1 = []
# 定义 loss 和 optimizer

net = fcn(21)
net.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
lr_init = optimizer.param_groups[0]['lr']
#学习率调整
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    if epoch>25:
        lr_tz = optimizer.param_groups[0]['lr']*0.1
        print("lr_tz = ", lr_tz)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
for e in range(60):
    adjust_learning_rate(optimizer,e,lr_init)
    train_loss = 0
    train_acc = 0
    train_acc_cls = 0
    train_mean_iu = 0
    train_fwavacc = 0
    
    prev_time = datetime.now()
    net = net.train()
    for idx,data in enumerate(train_loader):
        im = Variable(data[0])
        label = Variable(data[1])
        #print("im.shape = ",im.shape)# torch.Size([6, 3, 480, 320])
        #print("label.shape = ",label.shape)#torch.Size([6, 480, 320])
        im = im.to(device)
        label = label.to(device)
        # forward
        out = net(im)# torch.Size([6, 21, 480, 320])
        out = F.log_softmax(out, dim=1) # (b, n, h, w)
        out_get = out.max(dim=1)[1]
        #print("out.shape = ",out.shape) #torch.Size(([batch_size, 480, 320]))
        loss = criterion(out, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        label_pred = out.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            #print("22222222222222222222222 =lbt,lbp",lbt.shape,lbp.shape)
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
            train_acc += acc
            train_acc_cls += acc_cls
            train_mean_iu += mean_iu
            train_fwavacc += fwavacc
        if idx%100==0:
            print("loss = ",loss.item())
    net = net.eval()
    eval_loss = 0
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0
    for data in valid_loader:
        im = Variable(data[0])
        label = Variable(data[1])
        im = im.to(device)
        label = label.to(device)
        # forward
        out = net(im)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, label)
        eval_loss += loss.data.item()
        
        label_pred = out.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
            eval_acc += acc
            eval_acc_cls += acc_cls
            eval_mean_iu += mean_iu
            eval_fwavacc += fwavacc
        
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
        e, train_loss / len(train_loader), train_acc / len(voc_train), train_mean_iu / len(voc_train),
        eval_loss / len(valid_loader), eval_acc / len(voc_test), eval_mean_iu / len(voc_test)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str )
    train_loss1.append(train_loss / len(train_loader))
    train_acc1.append(train_acc / len(voc_train))
    train_miou1.append(train_mean_iu / len(voc_train))
    val_loss1.append(eval_loss / len(valid_loader))
    val_acc1.append(eval_acc / len(voc_test))
    val_miou1.append(eval_mean_iu / len(voc_test))



#保存模型   
torch.save(net.state_dict(), 'D:\lsy\FCN_VOC2012\model_data\parameter.pkl')
#保存训练过程指标变化情况
      #- overall accuracy
      #- mean accuracy
      #- mean IU
      #- fwavacc
      #- loss
#数据写入csv
data = pd.DataFrame({"train_loss":train_loss1,
                    "train_acc":train_acc1,
                    "train_miou":train_miou1,
                    "val_loss":val_loss1,
                    "val_acc":val_acc1,
                    "val_miou":val_miou1} )
data.to_csv("data.csv")
#加载模型进行预测(可视化结果)