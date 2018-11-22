
import time
import tensorflow as tf
import os
import random
# from real_time_input_data
import numpy as np
import cv2
import heapq
import PIL.Image as Image
import C3D_model_pytorch
import torch


def clip_images_to_tensor(imgs, num_frames_per_clip=16, crop_size=112):
    data = []
    np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    tmp_data = imgs
    img_datas = []
    if(len(tmp_data)!=0):
        for j in range(len(tmp_data)):
            img = Image.fromarray(tmp_data[j].astype(np.uint8))
            if img.width > img.height:
                scale = float(crop_size)/float(img.height)
                img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
            else:
                scale = float(crop_size)/float(img.width)
                img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
            crop_x = int((img.shape[0] - crop_size)/2)
            crop_y = int((img.shape[1] - crop_size)/2)
            img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
            img_datas.append(img)
    data.append(img_datas)
    np_arr_data = np.array(data).astype(np.float32)
    return np_arr_data



def predicted_clip(video_imgs_tensor, net, cuda):
    output = torch.nn.functional.softmax(net(video_imgs_tensor), dim=1)
    output = np.reshape(output.detach().cpu().numpy(), 101)
    predicted_five_accuracy = heapq.nlargest(5, output)
    predicted_five_label = np.argsort(output)[::-1][0:5]
    return predicted_five_accuracy, predicted_five_label


def real_time_recognition_video(video_path, model_path):
# 视频剪辑clips------------------------------------------------------------------------
    #  参数初始化-------------------------------
    count = 0
    classes = {}
    video_imgs = []
    flag = False
    frame_num = 0
    cuda = True if torch.cuda.is_available() else False
    net = C3D_model_pytorch.C3D(dropout_rate=1)
    model_data = torch.load(model_path)
    net.load_state_dict(model_data['state_dict'])
    if cuda:
        net = net.cuda()
    net.cuda()
    net.eval()
    frame_interval = 5  # 这东西有点关键
    torch.backends.cudnn.benchmark=True
    #  ----------------------------------------

    with open('./list/classInd.txt', 'r') as f:
        for line in f:
            content = line.strip('\n').split(' ')
            classes[content[0]] = content[1]
    cap = cv2.VideoCapture(video_path)


    while True:
        ret, img = cap.read()
        frame_num = frame_num + 1
        if type(img) == type(None):
            print('no images!')
            break
        
        if frame_num >= 1:
            frame_num = 0
            count += 1
            float_img = img.astype(np.float32)
            video_imgs.append(float_img)
            if count == 16:
                video_imgs_tensor = clip_images_to_tensor(video_imgs, 16, 112)
                video_imgs_tensor = video_imgs_tensor.transpose(0, 4, 1, 2, 3)
                video_imgs_tensor = torch.from_numpy(video_imgs_tensor)
                if cuda:
                    video_imgs_tensor = video_imgs_tensor.cuda()
                predicted_value_top5, predicted_label_top5 = predicted_clip(video_imgs_tensor, net, cuda)
                count = 0
                video_imgs = []
                flag = True
            if flag:
                for i in range(5):
                    cv2.putText(img, str(predicted_value_top5[i])+':'+classes[str(predicted_label_top5[i]+1)],
                                (10, 15*(i+1)),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, False)
        cv2.imshow('video', img)
        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()
#-------------------------------------------------------------------------------------------------


def frame_process(clip, clip_length=16, crop_size=112, channel_num=3):
    np_mean = np.load('crop_mean.npy').reshape([clip_length, 112, 112, 3])
    frames_num = len(clip)
    croped_frames = np.zeros([frames_num, crop_size, crop_size, channel_num]).astype(np.float32)
    for i in range(frames_num):
        img = Image.fromarray(clip[i].astype(np.uint8))
        if img.width > img.height:
            scale = float(crop_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width*scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height*scale+1)))).astype(np.float32)
        crop_x = int((img.shape[0]-crop_size)/2)
        crop_y = int((img.shape[1]-crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
        croped_frames[i, :, :, :] = img - np_mean[i]
    return croped_frames

def convert_images_to_clip(filename, clip_length=16, crop_size=112, channel_num=3):
    clip = []
    for parent, dirnames, filenames in os.walk(filename):
        filenames = sorted(filenames)
        if len(filenames) < clip_length:
            for i in range(0, len(filenames)):
                image_name = str(filename) + '/' + str(filenames[i])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
            for i in range(clip_length - len(filenames)):
                image_name = str(filename) + '/' + str(filenames[len(filenames) - 1])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
        else:
            s_index = random.randint(0, len(filenames) - clip_length)
            for i in range(s_index, s_index + clip_length):
                image_name = str(filename) + '/' + str(filenames[i])
                img = Image.open(image_name)
                img_data = np.array(img)
                clip.append(img_data)
    if len(clip) == 0:
       print(filename)
    clip = frame_process(clip, clip_length, crop_size, channel_num)
    return clip


def main(_):
    video_path = '../../dataset/UCF-101/UCF-101/JugglingBalls/v_JugglingBalls_g06_c01.avi'
    # model_name = 'C3D_model_pytorch.pkl'
    model_name = '../pretrained-model/resnet-101-kinetics-ucf101_split1.pth'
    real_time_recognition_video(video_path, model_name)


if __name__ == '__main__':
    tf.app.run()





