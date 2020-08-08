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


# 网络结构
class Net(torch.nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = torch.nn.Sequential( #Sequential为有序的容器
           #卷积层
           # (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1(卷积核元素之间的间距), groups=1, bias=True)
           torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), #输入28*28*3，输出28*28*32
           
           #池化层
           # (self, kernel_size, stride=None(默认值是kernel_size), padding=0, dilation=1, return_indices=False, ceil_mode=False)
           # 2*2最大值池化，步长为默认即2*2，膨胀1*1（即不膨胀）
           torch.nn.MaxPool2d(kernel_size=2), ##宽高减半，通道数不变，输入28*28*32，输出14*14*32

           # 非线性层
           torch.nn.ReLU()
       )
       self.conv2 = torch.nn.Sequential(
           torch.nn.Conv2d(32, 64, 3, 1, 1), #输入14*14*32，输出14*14*64
           torch.nn.MaxPool2d(2),
           torch.nn.ReLU()
       )
       self.conv3 = torch.nn.Sequential(
           torch.nn.Conv2d(64, 64, 3, 1, 1), #输入14*14*64，输出7*7*64
           torch.nn.MaxPool2d(2), #输入7*7*64，输出3*3*64
           torch.nn.ReLU()
       )

       #全连接层
       self.dense = torch.nn.Sequential(
           # 维度变换，输入3*3*64，输出128
           torch.nn.Linear(3 * 3 * 64, 128),
           torch.nn.ReLU(),
           # 线性分类器 输入128，输出10
           torch.nn.Linear(128, 10) #实现十分类
       )

   # 前向计算（输入为x）
   def forward(self, x):
       # 第一层的输出是x经过conv1的结果
       conv1_out = self.conv1(x)
       # 第二层的输出是 第一层的输出经过conv2的结果
       conv2_out = self.conv2(conv1_out)
       # 第三层的输出是 第二层的输出经过conv3的结果
       conv3_out = self.conv3(conv2_out)
       res = conv3_out.view(conv3_out.size(0), -1) #将多维数据平铺为一维
       return self.dense(res)

def cnn():
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
   cnn()
