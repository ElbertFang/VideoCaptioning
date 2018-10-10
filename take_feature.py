import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torchvision import transforms
import json
import os
from PIL import Image
import numpy as np


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

class EncoderCNN(nn.Module):
    def __init__(self,):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        vgg = models.vgg16(pretrained=True)
        s = list(vgg.classifier.children())[:-3]
        vgg.classifier = nn.Sequential(*s)
        self.vgg = vgg

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.vgg(images)
        return features


if __name__ == '__main__':
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    file_list = json.load(open('./data/caption_train.json','r'))
    images_list=[]
    for i in file_list:
        images_list.append(i['file_name'])
    images_features = np.zeros((len(images_list), 4096))
    # Build Models
    encoder = EncoderCNN()
    encoder.eval()
    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
    for index,i in enumerate(images_list):
        frame = load_image('./data/resized_frames/' + i + '/0.jpg', transform)
        frame_tensor = to_var(frame, volatile=True)
        feature = encoder(frame_tensor)
        frame_feature = feature
        for j in range(1,16):
            frame = load_image('./data/resized_frames/' + i + '/%d.jpg' % j, transform)
            frame_tensor = to_var(frame, volatile=True)
            feature = encoder(frame_tensor)
            frame_feature = torch.add(frame_feature, feature)
        video_frame = torch.div(frame_feature, 16)
        images_features[index,:] = video_frame.data.cpu().numpy().copy()
        if index%1000 == 0:
            print('%d of %d file have finished'%(index, len(images_list)))
    np.save('./msvd_train_vgg16.npy', images_features)