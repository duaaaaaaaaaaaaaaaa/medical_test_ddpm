
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib
from skimage import io, color
import os
 
def abnor_map(path1,path2,dir):
    # 1.读取gt图片，并转为灰度图像
    rgbImg=cv2.imread(path2)
    img_gt=cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
    
    # 2.读取生成的图片，并转为灰度图像
    rgb = io.imread(path1)  # 读取图片
    img_pred = color.rgb2gray(rgb)
    
    
    # 3.开始进行制作误差热力图
    A_img = img_gt
    B_img = img_pred
    
    #选取需要计算差值的两幅图片
    dimg1 = A_img[:, :]
    #归一化
    dimg1_2 = np.zeros(dimg1.shape, dtype=np.float32)
    cv2.normalize(dimg1, dimg1_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # 显示选取的图像
    plt.figure()
    plt.imshow(dimg1_2, cmap='gray')
    plt.show()
    
    dimg2 = B_img[:, :]
    dimg2_2 = np.zeros(dimg2.shape, dtype=np.float32)
    cv2.normalize(dimg2, dimg2_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.figure()
    plt.imshow(dimg2_2,cmap='gray')
    plt.show()
    
    d = abs(dimg1_2-dimg2_2)*1
    
    fig = plt.figure(dpi=200)
    plt.figure(num=1)
    cnorm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    m = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=matplotlib.cm.jet)
    m.set_array(d)
    plt.imshow(d, norm=cnorm, cmap="jet")
    plt.axis("off")
    plt.colorbar(m)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join(dir,'anomaly_map.png'), bbox_inches='tight', dpi=400,pad_inches=0)