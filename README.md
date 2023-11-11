# RWKV-ToolKit


本项目基于 [RWKV](https://github.com/BlinkDL/AI-Writer) 
站在大佬的肩膀上创作
## 目前是0.3版本
支持windows 训练，推理，断点训练，微调 web可视化懒人包 

因为是解压即用的懒人包，所以环境无法全部上传git，请到网盘下载 
```
链接: https://pan.baidu.com/s/1oh2p2C8qfRJ1XfrUzzj5nA 提取码: 9n3s
```

##使用方法 首先下载懒人包 然后把下载的替换掉懒人包里面的代码文件，如果已经下载懒人包，那直接替换

```
 如果缺少某些库，可以cd到懒人包的 torch->Scripts 然后在里间运行pip 安装缺少的包 懒人包已经把难装的包都预制了
```


# 参数说明
https://zhuanlan.zhihu.com/p/432715547

```
n_epoch：epoch次数。这里的epoch不代表训练了一轮语料，而是固定的长度。一个epoch训练ctx_len* epoch_length_fixed个字。默认n_epoch为500，可以根据自己的语料大小设置合适的n_epoch。
datafile：是语料文件的路径，需要修改成您自己准备的语料路径。
ctx_len：是机器学习时参考的上下文字数，建议512。显存不足的话可以设置成256。
n_layer、n_head：神经网络的层数，层数越高，能拟合的函数就越复杂，但是语料较小、层数过多的情况下，会过拟合。对于300m以下的文体比较统一的语料文件，n_layer可以设置成6。
300m以下，n_layer设置6，n_head设置8。
300~800m，n_layer设置8，n_head设置10。
800m以上，n_layer设置12，n_head设置12。小语料配合过多层数，会导致过拟合（由学习变成背诵），而大规模语料配合过少层数，会导致欠拟合（由学习变成茫然）。
lr_init：初始化学习率，决定学习发生的速度。数值越大，权值改变得越快，反之亦然。
lr_final：最终学习率。默认的 lr_init 8e-4，lr_final 1e-5。如果您想在其它成熟模型的基础上训练，可以将lr_init设置为1e-4学习率初始化，lr_final设置为1e-5为止。对于初学者来说，如果想兼顾多种语料特点并且风格偏向某种语料，可以先用语料A+语料B训练，之后再用语料B训练一遍。例如我想训练一个百科+文学的模型，还想让机器人写作风格倾向于文学，那么就可以先用百科+文学的语料训练，生成模型x，之后再用文学语料在模型x上继续训练生成模型y，之后就可以用模型y去写作了。
epoch_save_frequency：模型自动存档频率，默认10轮存一次。
epoch_save_path：模型自动存档文件名的前缀。默认为前缀为“trained-”，完整文件名为前缀+轮数，例如“trained-100.pth”表示第100轮训练的模型文件。
batch_size：训练时并行任务的数量，ctx_len、n_layer、n_head越大，单个任务所占显存就越多。可以先设置batch_size小一点，然后通过任务管理器观察显存占用情况，如果显存没占满，可以把batch_size调高以达到最快的训练速度。
```

# 为了项目更好发展，可以支持一下

> # ![](https://github.com/1694439208/BluePrint/raw/master/image/pay.png)