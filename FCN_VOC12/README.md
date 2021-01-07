## FCN_VOC12:Fully Convolutional Networks for Semantic Segmentation for voc2012全卷积网络语义分割pytorch复现
---

### 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [训练步骤 how2train](#训练步骤)
5. [预测步骤 how2predit](#预测步骤)
6. [参考资料 Referrence](#Reference)

### 性能情况
| 数据集 | 输入图片大小 | val_mIOU | val_ACC | val_LOSS |
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12 | 300x300 | 0.38 | 0.88 | 0.43 | 

### 所需环境
- 虚拟环境：Anaconda 4.5.4
- Python环境：Python 3.6.5
- 编译器：vs code
- 深度学习框架：pytorch1.5.1
- Python库：numpy,pandas,os,PIL,matplotlib等

### 文件下载
- VOC12语义分割数据集：![voc12](https://www.kaggle.com/luoshiyong/voc2012-semanticsegment)
- 权重文件：效果太差（如果需要请联系我）

### 训练步骤
### 预测步骤

### Reference
- 语义分割指标https://zhuanlan.zhihu.com/p/341375686
- FCN数据集获取https://zhuanlan.zhihu.com/p/337131142
- FCN模型实现https://zhuanlan.zhihu.com/p/337193820
- FCN训练及结果https://zhuanlan.zhihu.com/p/341369456


