#!/bin/bash
echo 基于data1训练1个epoch，在四个测试集上测试
python train.py --save_path model_d1/model1.pth --train True --train_path /hy-tmp/data/data1/train
python train.py --model_path model_d1/model1.pth --test True --test_path /hy-tmp/data/data1/test
python train.py --model_path model_d1/model1.pth --test True --test_path /hy-tmp/data/data2/test
python train.py --model_path model_d1/model1.pth --test True --test_path /hy-tmp/data/data3/test
python train.py --model_path model_d1/model1.pth --test True --test_path /hy-tmp/data/data4/test

echo train 2 epoch based on data1,test it on four test datasets
python train.py --save_path model_d1/model2.pth --train True --train_path /hy-tmp/data/data1/train --epoch 2
python train.py --model_path model_d1/model2.pth --test True --test_path /hy-tmp/data/data1/test
python train.py --model_path model_d1/model2.pth --test True --test_path /hy-tmp/data/data2/test
python train.py --model_path model_d1/model2.pth --test True --test_path /hy-tmp/data/data3/test
python train.py --model_path model_d1/model2.pth --test True --test_path /hy-tmp/data/data4/test

echo train 1 epoch based on data4,test it on four test datasets
python train.py --save_path model_d4/model1.pth --train True --train_path /hy-tmp/data/data4/train
python train.py --model_path model_d4/model1.pth --test True --test_path /hy-tmp/data/data1/test
python train.py --model_path model_d4/model1.pth --test True --test_path /hy-tmp/data/data2/test
python train.py --model_path model_d4/model1.pth --test True --test_path /hy-tmp/data/data3/test
python train.py --model_path model_d4/model1.pth --test True --test_path /hy-tmp/data/data4/test

echo train 2 epoch based on data4,test it on four test datasets
python train.py --save_path model_d4/model2.pth --train True --train_path /hy-tmp/data/data4/train 
python train.py --model_path model_d4/model2.pth --test True --test_path /hy-tmp/data/data1/test
python train.py --model_path model_d4/model2.pth --test True --test_path /hy-tmp/data/data2/test
python train.py --model_path model_d4/model2.pth --test True --test_path /hy-tmp/data/data3/test
python train.py --model_path model_d4/model2.pth --test True --test_path /hy-tmp/data/data4/test

echo Now train it on data5
echo 1 epoch
python train.py --save_path model_d5/model1 --train True --train_path /hy-tmp/data/data5/train
echo test it on test data 5,6,7,8
python train.py --model_path model_d5/model1 --test True --test_path /hy-tmp/data/data5/test
python train.py --model_path model_d5/model1 --test True --test_path /hy-tmp/data/data6/test
python train.py --model_path model_d5/model1 --test True --test_path /hy-tmp/data/data7/test
python train.py --model_path model_d5/model1 --test True --test_path /hy-tmp/data/data8/test

echo 2 epoch
python train.py --save_path model_d5/model2 --train True --train_path /hy-tmp/data/data5/train --epoch 2
echo test it on test data 5,6,7,8
python train.py --model_path model_d5/model2 --test True --test_path /hy-tmp/data/data5/test
python train.py --model_path model_d5/model2 --test True --test_path /hy-tmp/data/data6/test
python train.py --model_path model_d5/model2 --test True --test_path /hy-tmp/data/data7/test
python train.py --model_path model_d5/model2 --test True --test_path /hy-tmp/data/data8/test

echo Now train it on data8
echo 1 epoch
python train.py --save_path model_d8/model1 --train True --train_path /hy-tmp/data/data8/train
echo test it on test data 5,6,7,8
python train.py --model_path model_d8/model1 --test True --test_path /hy-tmp/data/data5/test
python train.py --model_path model_d8/model1 --test True --test_path /hy-tmp/data/data6/test
python train.py --model_path model_d8/model1 --test True --test_path /hy-tmp/data/data7/test
python train.py --model_path model_d8/model1 --test True --test_path /hy-tmp/data/data8/test

echo 2 epoch
python train.py --save_path model_d8/model2 --train True --train_path /hy-tmp/data/data8/train --epoch 2
echo test it on test data 5,6,7,8
python train.py --model_path model_d8/model2 --test True --test_path /hy-tmp/data/data5/test
python train.py --model_path model_d8/model2 --test True --test_path /hy-tmp/data/data6/test
python train.py --model_path model_d8/model2 --test True --test_path /hy-tmp/data/data7/test
python train.py --model_path model_d8/model2 --test True --test_path /hy-tmp/data/data8/test
