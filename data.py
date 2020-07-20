import os
from skimage import io
import torchvision.datasets.mnist as mnist

root = "./mnist/"
# 训练特征 训练标号
train_set = (
   mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
   mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
       )
# 测试特征 测试标号
test_set = (
   mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
   mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
       )

# 从原始数据转换到jpg图片并装到文件夹
def convert_to_img(train=True):
   if train:  # 训练数据（将其中一部分划分为验证集）
       f1 = open(root + 'train.txt', 'w')
       f2 = open(root + 'validation.txt', 'w')
       data_path1 = root + '/train/'
       data_path2 = root + '/validation/'
       if not os.path.exists(data_path1):
           os.makedirs(data_path1)
           os.makedirs(data_path2)
           # 用enumerate将可遍历对象组合成索引
       for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
       	   if i<55000:
           		img_path = data_path1 + str(i) + '.jpg'
           		# 保存图片
           		io.imsave(img_path, img.numpy())
           # 保存标号文件路径和标号
           		f1.write(img_path + ' ' + str(label) + '\n')
           else:
           		img_path = data_path2 + str(i) + '.jpg'
           		io.imsave(img_path, img.numpy())
           		f2.write(img_path + ' ' + str(label) + '\n')
       f1.close()
       f2.close()
   else:  # 测试数据
       f = open(root + 'test.txt', 'w')
       data_path = root + '/test/'
       if not os.path.exists(data_path):
           os.makedirs(data_path)
       for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
           img_path = data_path + str(i) + '.jpg'
           io.imsave(img_path, img.numpy())
           f.write(img_path+' '+str(label)+'\n')
       f.close()


print("Building training set...")
convert_to_img(True)
print("Building test set...")
convert_to_img(False)