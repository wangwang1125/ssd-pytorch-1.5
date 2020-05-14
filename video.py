import time

import cv2
import torch
import numpy as np
from torch.autograd import Variable
from data import VOC_CLASSES as labels, BaseTransform, COLORS
from ssd import build_ssd
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def demo(net,transform):
    def predict(frame):
        x = cv2.resize(frame, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        height, width = x.shape[:2]
        x = Variable(x.unsqueeze(0))
        if torch.cuda.is_available():
            x = x.cuda()
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(frame.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                score = detections[0, i, j, 0]
                label_name = labels[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, label_name, (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    cap = cv2.VideoCapture(0)
    while True:
        t1=time.time()
        # grab next frame
        _,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = predict(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # keybindings for display

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:  # exit
            break
        print(1/(time.time()-t1))

if __name__ == '__main__':
    net = build_ssd('test', 300, 21)  # initialize SSD
    net.load_state_dict(torch.load('weights/ssd300_mAP_77.43_v2.pth'))
    transform = BaseTransform(net.size,(104/256.0, 117/256.0, 123/256.0))
    demo(net,transform)






