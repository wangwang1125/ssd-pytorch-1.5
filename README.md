原github地址：https://github.com/amdegroot/ssd.pytorch

因源程序和最新的pytorch版本有冲突，本程序对此进行修改，目前可以正常训练即使用
pytorch版本1.5

其中video.py为摄像头测试，也可用源代码自带的测试程序在demo中


注意：要将data\voc0712.py 28行换成自己voc路径
          并将data\coco.py  10行换成本项目下/data路径