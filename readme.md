# 本文档指出一些帮助文档的例子和更新日志
## 10.16 train.py的使用说明
### 训练
**一定要添加的参数： 训练路径，配置文件**
1. 训练路径

路径下有 `c` 和 `nc`两个文件夹，分别存放演唱会和非演唱会的图片
2. 配置文件

包含`lr epochs optimizer criterion`的配置文件
举例：`PS E:\mypython\own\cv\zzy> python train.py --train_path E:\mypython\own\optical_flow\data1\train --config t.yaml`
3.其他参数

--save_path 保存模型的路径，以.pth为扩展名
--model_path 导入模型的路径，.pth为扩展名，一般是之前保存的模型，默认导入resnet18的参数
### 测试
**只需要输入--test_path参数就可以了**
### 一个综合的例子
```powershell
PS E:\mypython\own\cv\zzy> python train.py --model_path example.pth --train_path E:\mypython\own\optical_flow\data1\train --config example.yaml --save_path example.pth --test_path E:\mypython\own\optical_flow\data1\test
```
解释：从训练example.pth的参数，并将结果覆盖到example.pth上
## 9.2(2)
上传了训练日志，修改训练代码，增加训练脚本
## 9.2
增加了训练代码和工具库里移动文件操作的函数
## 8.31
修改了tools工具包里处理视频的函数和类
## 8.29(2)
修改了生成迭代器的方式,生成Loader时直接输入含有图片的两个目录即可,教程同样也更新了
## 8.29(1)
增加了直接从视频文档生成数据集的类，帮助文档在[这里](./help/data_pro_ex.ipynb)
