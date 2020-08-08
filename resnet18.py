import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

root = "./mnist/"

# 准备数据
def default_loader(path):
   return Image.open(path).convert('RGB')

class MyDataset(Dataset):
   def __init__(self, txt, transform=transforms.ToTensor(), target_transform=None, loader=default_loader):
       fh = open(txt, 'r')  # txt是路径和文件名
       rec = []
       for line in fh:
           line = line.strip('\n')
           line = line.rstrip() # 删除空格回车
           words = line.split() # 将记录分割为路径和标号，如./mnist/test/0.jpg tensor(7)
           rec.append((words[0], int(words[1][7]))) #存储路径和编号
       self.rec = rec
       self.transform = transform
       self.target_transform = target_transform
       self.loader = loader

   # train_loader里面的
   def __getitem__(self, index):
       fn, label = self.rec[index]  # fn是完整路径，label是标号
       rec = self.loader(fn)  # 调用default_loader(path)，按照路径读取图片
       if self.transform is not None:
           rec = self.transform(rec)  # 将图片转换成FloatTensor类型
       return rec, label

   def __len__(self):
       return len(self.rec)

class ResBlock(torch.nn.Module):
  def __init__(self, IN, OUT, S, change_channels=False):
    super(ResBlock, self).__init__()
    self.change_channels = change_channels
    self.resblock = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=IN, out_channels=OUT, kernel_size=3, stride=S, padding=1, bias=False),
      torch.nn.BatchNorm2d(OUT),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(in_channels=OUT, out_channels=OUT, kernel_size=3, stride=1, padding=1, bias=False),
      torch.nn.BatchNorm2d(OUT)
      )
    if self.change_channels:
      self.Res= torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=IN, out_channels=OUT, kernel_size=1, stride=S, bias=False),
        torch.nn.BatchNorm2d(OUT)
        )
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x):
    y = self.resblock(x)
    res = x
    if self.change_channels:
      res = self.Res(x)
    y += res
    y = self.relu(y)
    return y

def copy_res(IN, OUT, S, cnt):
  net = []
  net.append(ResBlock(IN, OUT, S, change_channels=True))
  for i in range(1, cnt):
    net.append(ResBlock(OUT, OUT, 1, change_channels=False))
  return torch.nn.Sequential(*net)


# 网络结构
class Net(torch.nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = torch.nn.Sequential(
           torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
           torch.nn.BatchNorm2d(64),
           torch.nn.ReLU(inplace=True),
           torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
       )
       self.conv2_x = copy_res(64, 64, S=1, cnt=2)
       self.conv3_x = copy_res(64, 128, S=2, cnt=2)
       self.conv4_x = copy_res(128, 256, S=2, cnt=2)
       self.conv5_x = copy_res(256, 512, S=2, cnt=2)

       self.average_pool = torch.nn.AvgPool2d(7, stride=1)
       self.FC = torch.nn.Linear(512, 10) #1*1*2048-1*10
   
   # 前向计算（输入为x）
   def forward(self, x):
       conv1_out = self.conv1(x)
       conv2_out = self.conv2_x(conv1_out)
       conv3_out = self.conv3_x(conv2_out)
       conv4_out = self.conv4_x(conv3_out)
       conv5_out = self.conv5_x(conv4_out)
       res = self.average_pool(conv5_out)
       res = res.view(res.size(0), -1) #将多维数据平铺为一维
       return self.FC(res)

def resnet():
   print("Reading train data...")
   train_data = MyDataset(txt=root + 'train.txt', transform=transforms.ToTensor()) #格式转换为tensor
   train_loader = DataLoader(dataset=train_data, batch_size=50, shuffle=True)
   #shuffle 随机数种子；batch_size为50，loader将数据集划分为n个50的集合
   print("Reading validation data...")
   val_data = MyDataset(txt=root + 'validation.txt', transform=transforms.ToTensor())
   val_loader = DataLoader(dataset=val_data, batch_size=50)
   print("Reading test data...")
   test_data = MyDataset(txt=root + 'test.txt', transform=transforms.ToTensor())
   test_loader = DataLoader(dataset=test_data, batch_size=50)
   # GPU or CPU
   if torch.cuda.is_available():
       is_cuda = True
       print("work on GPU")
   else:
       is_cuda = False
       print("work on CPU")

   print("Setup Net...")
   print("LR=0.01")
   if is_cuda:
       model = Net().cuda()
   else:
       model = Net()

   # 打印网络结构
   print(model)
   optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9, weight_decay=5e-4) #momentum：加速收敛
   #weight decay（权值衰减）目的是防止过拟合。在损失函数中，weight decay是放在正则项前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
   loss_func = torch.nn.CrossEntropyLoss()

   for epoch in range(3):  # 训练3批次
       print('epoch {}'.format(epoch + 1))
       # 训练
       train_loss = 0.
       train_acc = 0.
       cnt = 0
       total = 0
       for batch_x, batch_y in train_loader:  # 特征 标号
           if is_cuda:
               batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
           else:
               batch_x, batch_y = Variable(batch_x), Variable(batch_y)
           # Variable和Tensor都存储高维数据，Variable是可更改的，而Tensor是不可更改的
           out = model(batch_x)  # batch_x通过网络的结果是out
           loss = loss_func(out, batch_y)  # 网络结果out和实际batch_y对比的得到损失
           train_loss += loss.item()  # 累加训练损失
           if is_cuda:
              pred = torch.max(out, 1)[1].cuda()  # 返回最大值的索引
           else:
              pred = torch.max(out, 1)[1]  # 返回最大值的索引
           train_correct = (pred == batch_y).sum()  # 多少个预测为正确的
           total += len(batch_x)
           train_acc += train_correct.item()  # 累加训练正确的数量
           # print('length: %d'%total)
           print('[epoch:%d, iter:%d] Loss: %.6f | Acc: %.6f '
                          % (epoch + 1, (cnt + 1), train_loss / total, train_acc / total))
           optimizer.zero_grad()  # 清除所有优化的grad
           loss.backward()  # 误差反向传递
           optimizer.step()  # 单次优化
           cnt += 1
       print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))

       # 验证
       model.train(True)
       eval_loss = 0.
       eval_acc = 0.
       for batch_x, batch_y in val_loader:  # 特征 标号
           if is_cuda:
               batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
           else:
               batch_x, batch_y = Variable(batch_x), Variable(batch_y)
           out = model(batch_x)
           loss = loss_func(out, batch_y)
           eval_loss += loss.item()
           if is_cuda:
              pred = torch.max(out, 1)[1].cuda()  # 返回最大值的索引
           else:
              pred = torch.max(out, 1)[1]
           num_correct = (pred == batch_y).sum()
           eval_acc += num_correct.item()
       print('validation Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_data)), eval_acc / (len(val_data))))


  # 测试
   model.eval()
   eval_loss = 0.
   eval_acc = 0.
   for batch_x, batch_y in test_loader:  # 特征 标号
       if is_cuda:
          batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
       else:
          batch_x, batch_y = Variable(batch_x), Variable(batch_y)
       out = model(batch_x)
       loss = loss_func(out, batch_y)
       eval_loss += loss.item()
       if is_cuda:
          pred = torch.max(out, 1)[1].cuda()
       else:
          pred = torch.max(out, 1)[1]
       num_correct = (pred == batch_y).sum()
       eval_acc += num_correct.item()
   print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))
   #print('Test Acc: {:.6f}'.format(eval_acc / (len(test_data))))

if __name__ == '__main__':
   resnet()