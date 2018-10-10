import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import json
import h5py
from data_loader_features import get_loader
from build_vocab import Vocabulary
from model_features import EncoderCNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from lib.metrics import calculate_bleu
from lib.metrics import score

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main(args):
    file_cap = json.load(open('data/captions_test.json', 'r'))
    file_captions = {}
    file_name = []
    number_cap = 0
    for index, i in enumerate(file_cap):
        if index == 0:
            xxx = i['file_name']
            file_name.append(i['file_name'])
            file_captions[0] = [i['caption'].encode('utf-8')]
        if i['file_name'] == xxx:
            file_captions[number_cap].append(i['caption'].encode('utf-8'))
        else:
            number_cap += 1
            xxx = i['file_name']
            file_name.append(i['file_name'])
            file_captions[number_cap] = [i['caption'].encode('utf-8')]
    len_cap = 0
    for i in file_captions:
        if len(file_captions[i]) >= len_cap:
            len_cap = len(file_captions[i])
    for i in file_captions:
        while (len(file_captions[i]) != len_cap):
            file_captions[i].append('')
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    calculate_captions = {}
    test_features = h5py.File('/home/Data/MSVD/vgg_lower_15.h5')
    write_sentences = open('data/test_score/test_captions','w')
    score_file = {}
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size,len(vocab), args.num_layers)
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    encoder.cuda()
    decoder.cuda()
    for index,k in enumerate(file_name):
        encoder.eval()
        decoder.eval()
        val_image = torch.Tensor(test_features[k])
        val_image = torch.cumsum(val_image, dim=0)
        val_image = torch.div(torch.squeeze(torch.chunk(val_image, 15, dim=0)[-1]), 15)
        val_image = torch.unsqueeze(val_image, 0)
        val_image_tensor = to_var(val_image, volatile=False)
        val_feature = encoder(val_image_tensor)
        sampled_ids = decoder.sample(val_feature)
        sampled_ids = sampled_ids.cpu().data.numpy()

        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption[1:-1])
        score_file[file_name[index]] = sentence
        calculate_captions[index]=[sentence.encode('utf-8')]
    score_now = score(file_captions,calculate_captions)
    print(score_now)
    with open('data/test_score/lstm_test_captions.json','w') as f:
        json.dump(score_file,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='./models/best-e-vgg19.pkl',
                        help='path for saving trained models')
    parser.add_argument('--decoder_path', type=str, default='./models/best-d-vgg19.pkl',
                        help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_msvd_5.pkl',
                        help='path for vocabulary wrapper')
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    args = parser.parse_args()
    print(args)
    main(args)