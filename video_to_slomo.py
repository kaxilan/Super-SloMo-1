# -*- coding:utf-8 -*-
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import model
import dataloader
import platform
from tqdm import tqdm
import time

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./video/2.gif', help='path of video to be converted')
parser.add_argument("--checkpoint", type=str, default='model/SuperSloMo.ckpt', help='path of checkpoint for pretrained model')
parser.add_argument("--fps", type=float, default=270, help='specify fps of output video. Default: 30.')
parser.add_argument("--sf", type=int, default=4, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--output", type=str, default="output.gif", help='Specify output file name. Default: output.mp4')
args = parser.parse_args()

def main():
    start_time = time.time()
    extractionDir = "tmpSuperSloMo"
    if os.path.isdir(extractionDir):
        rmtree(extractionDir)
    os.mkdir(extractionDir)

    extractionPath = os.path.join(extractionDir, "input")
    outputPath     = os.path.join(extractionDir, "output")
    os.mkdir(extractionPath)
    os.mkdir(outputPath)
    # 抽取一个视频的所有帧为图片
    os.system('ffmpeg -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(args.video, extractionPath))
    
    # Initialize transforms
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,std=std)
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Load data
    videoFrames = dataloader.Video(root=extractionPath, transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

    # 开启gpu
    device = torch.device("cuda:0")
    
    # Initialize model:流计算CNN
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    
    # Initialize model:流插值CNN
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False
    
    # used for backwarping to an image
    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)
    
    # 加载模型
    dict1 = torch.load(args.checkpoint)
    # 从字典中读取保存的参数
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 1
    # with torch.no_grad():对于不需要反向传播的情景(inference，测试推断)可以实现一定速度的提升
    with torch.no_grad():
        # frame0, frame1:是连续的两个视频帧
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader)):
            I0 = frame0.to(device)
            I1 = frame1.to(device)
            # flowComp对象会自动调用UNet类的forward方法提取特征，这个特征是四个（四通道）特征图，并不是一个向量
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            # 合成光流：I0 --> I1
            F_0_1 = flowOut[:,:2,:,:]
            # 合成光流：I1 --> I0
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            for batchIndex in range(args.batch_size):
                # detach()：截断反向传播的梯度流
                (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".jpg"))

            frameCounter += 1
            # Generate intermediate frames
            for intermediateIndex in range(1, args.sf):
                t = intermediateIndex / args.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]
                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)
                
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0     
                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)
                wCoeff = [1 - t, t]
                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                for batchIndex in range(args.batch_size):
                    (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".jpg"))
                frameCounter += 1
            # Set counter accounting for batching of frames
            frameCounter += args.sf * (args.batch_size - 1)

    # Generate video from interpolated frames
    os.system('ffmpeg -r {} -i {}/%d.jpg -crf 17 -vcodec libx264 {}'.format(args.fps, outputPath, args.output))
    # Remove temporary files
    rmtree(extractionDir)
    end_time = time.time()
    print("程序总共运行时间为： "+str(end_time - start_time))
main()
