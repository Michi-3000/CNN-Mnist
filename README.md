This is my code using CNN and Resnet to run Mnist Dataset(in the folder **mnist** or download [here](http://yann.lecun.com/exdb/mnist/)) with pytorch. <br> 
Before model training, you have to run **data.py** for data preprocessing first.

-------------------------
Code List
-------
* **mycnn.py**: CNN(CPU version)<br>
* **mycnn_cuda.py**: CNN(GPU version)<br>
* **myresnet.py**: Resnet built for Mnist Dataset (based on Resnet18)<br>
* **resnet18.py** and **resnet50.py**: Resnet18 and Resnet50 according to [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385). If you want to run these codes, you have to resize the pictures to 224*224 first.<br>
<br>
This code was tested on a MacOS 10.15.1 system using Pytorch version 1.1.0.post2. For more information about running the code on slurm, you may refer to SLU.sh.
