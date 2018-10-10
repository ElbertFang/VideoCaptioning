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
import datetime

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    file_cap = json.load(open('data/captions_val.json','r'))
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
            number_cap+=1
            xxx = i['file_name']
            file_name.append(i['file_name'])
            file_captions[number_cap] = [i['caption'].encode('utf-8')]
    len_cap = 0
    for i in file_captions:
        if len(file_captions[i]) >= len_cap:
            len_cap = len(file_captions[i])
    for i in file_captions:
        while(len(file_captions[i])!=len_cap):
            file_captions[i].append('')
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                            args.batch_size,
                             shuffle=True, num_workers=1)

    # Build the models
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)
    score_file = open('./data/scores_list.txt','w')
    best_score = 0.0
    for epoch in range(args.num_epochs):
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(nowTime)
        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            if len(images)==1:
                continue
            images = to_var(images, volatile=False)
            images = torch.cumsum(images,dim=1)
            images = torch.div(torch.squeeze(torch.chunk(images, 15, dim=1)[-1]), 15)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # Forward, Backward and Optimize
            encoder.train()
            decoder.train()
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

                # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, args.num_epochs, i, total_step,
                         loss.data[0], np.exp(loss.data[0])))
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(nowTime)
        # Save the models
        torch.save(decoder.state_dict(),
                   os.path.join(args.model_path,
                                'msvd-%d-d-vgg19.pkl' % (epoch)))
        torch.save(encoder.state_dict(),
                   os.path.join(args.model_path,
                                'msvd-%d-e-vgg19.pkl' % (epoch)))
        calculate_captions = {}
        val_features = h5py.File('../../file/vgg_lower_15.h5')
        write_sentences = open('data/sentences/epoch%d-%d.txt'%(epoch,i),'w')
        for index,k in enumerate(file_name):
            val_image = torch.Tensor(val_features[k])
            val_image = torch.cumsum(val_image, dim=0)
            val_image = torch.div(torch.squeeze(torch.chunk(val_image, 15, dim=0)[-1]), 15)
            val_image = torch.unsqueeze(val_image, 0)
            val_image_tensor = to_var(val_image, volatile=False)
            # Generate caption from image
            encoder.eval()
            decoder.eval()
            val_feature = encoder(val_image_tensor)
            sampled_ids = decoder.sample(val_feature)
            sampled_ids = sampled_ids.cpu().data.numpy()

            # Decode word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption[1:-1])
            write_sentences.write("%-25s"%(file_name[index])+' '+sentence+'\n')
            calculate_captions[index]=[sentence.encode('utf-8')]
        score_now = score(file_captions,calculate_captions)
        print(score_now)
        print(score_now['CIDEr']+score_now['Bleu_4'])
        if ((score_now['CIDEr']+score_now['Bleu_4'])>=best_score):
            best_score = (score_now['CIDEr']+score_now['Bleu_4'])
            torch.save(decoder.state_dict(),
                       os.path.join(args.model_path,
                                    'best-d-vgg19.pkl'))
            torch.save(encoder.state_dict(),
                       os.path.join(args.model_path,
                                    'best-e-vgg19.pkl'))
        score_file.write('Epoch[%d/%d]  '%(epoch,i)+str(score_now)+'\n')
    score_file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_msvd.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../../file/vgg_lower_15.h5',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/captions_train.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=500,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=385,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    args = parser.parse_args()
    print(args)
    main(args)