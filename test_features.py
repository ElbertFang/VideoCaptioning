import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from data_loader_features import get_loader
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from model_features import EncoderCNN, DecoderRNN
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
import json

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def main(args)
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build Models
    encoder = EncoderCNN(args.embed_size)
    encoder.eval()
    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    image_features = np.load('msvd_test_vgg16.npy')
    encoder_path = args.encoder_path
    decoder_path = args.decoder_path

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    if torch.cuda.is_available():
        encoder.cuda()
    save = open(args.save_path,'w')
    for i in image_features:
        image = torch.Tensor(i)
        image = torch.unsqueeze(image,0)
        image_tensor = to_var(image, volatile=False)

        # Generate caption from image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids.cpu().data.numpy()

        # Decode word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption[1:-1])
        save.write(sentence+'\n')
    save.close()
    print ("All captions has been writen .")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./data/resized_images',
                         help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models/msvd-e-1-300-vgg16.pkl',
                         help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/msvd-d-1-300-vgg16.pkl',
                         help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_msvd.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--txt_path', type=str, default='/data/Dataset/coco/test_list.txt',
                        help='path for vocabulary wrapper')
    parser.add_argument('--save_path', type=str, default='./data/test_sentences/log-1-300.txt',
                        help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)