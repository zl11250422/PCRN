UPLOADING.............................................


```

## Environment

[PyTorch >= 1.7](https://pytorch.org/)  


## How To Test
· Refer to ./options/test for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  

```
python test.py -opt options/test/benchmark_PCRN_x4.yml
```
The testing results will be saved in the ./results folder.

## How To Train
· Refer to ./options/train for the configuration file of the model to train.  
· Preparation of training data can refer to this page. All datasets can be downloaded at the official website.  
· Note that the default training dataset is based on lmdb, refer to [docs in BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) to learn how to generate the training datasets.  
· The training command is like  
```
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_PCRN_x4.yml
```
