# medical_test_ddpm

# 数据集来源
  我们在kaggle网站（https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/data）上找到了名为“Tuberculosis (TB) Chest X-ray Database”的数据集，其中包含了3500张正常X光图像和700张结核病患者的X光图像，我们自定义了Mydataset类，进行了数据的处理，将图片转化成三通道的张量，由于分辨率过高，在训练过程中耗费大量的显存，我们在实验中将其形状调整为3*128*128进行训练。
  
# 项目介绍
  本小组基于目前医学影像分析的复杂现状设计了一套基于概率扩散模型（DDPM）的胸腔X光图像异常检测方法，能够通过当前最先进的深度扩散模型对胸腔X光进行无监督地异常检测。该算法可以很大程度上帮助医生有效地进行病灶识别与标注，大大降低误诊概率，发现更有价值的罕见病状。
 
# 目录结构描述
    ├── README.md           // 帮助文档
    
    ├── Diffusion    // 合成DDS的 python脚本文件
    
    │   ├── Diffusion.py     // 基本加噪、去噪公式
    
    │   ├── Model.py         //模型搭建
    
    │   ├── Train.py         //训练与扩散过程
    
    │   ├── __init__.py      //初始化 导入相关包
    
    │   └── abnormal_map.py   //异常可视化
    
    ├── Datasets.py              // 数据集的获取和处理

    ├── Main.py              // 主函数

    └── Scheduler.py              // 学习率调试                 

 
 
