import argparse
import torch
import torch.nn as nn
import scipy.io.wavfile as wavfile
import numpy as np
from tqdm import tqdm
from glob import iglob
#from Model.Resnet50_1d import ResNet50 as resnet
import librosa
from sklearn import preprocessing
from plotcm import plot_confusion_matrix
import matplotlib.pyplot as plt
import cv2
from spectralcluster import SpectralClusterer
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import get_data_loader
import torch.distributed as dist
import torch.multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description='Speaker clustering')
    '''
    K-mean or Blur
    '''
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--num_of_class', type=int, default=5994)
    
    parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
    parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
    parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files');
    parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch');
    parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch');
    parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
    parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')

## Training details
    parser.add_argument('--test_interval',  type=int,   default=1,     help='Test and save every [test_interval] epochs');
    parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs');
    parser.add_argument('--trainfunc',      type=str,   default="angleproto",     help='Loss function');

## Optimizer
    parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
    parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
    parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
    parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
    parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
    parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
    parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
    parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions');
    parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
    parser.add_argument('--nPerSpeaker',    type=int,   default=2,      help='Number of utterances per speaker per batch, only for metric learning based losses');
#parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');
    parser.add_argument('--nClasses',       type=int,   default=1211,   help='Number of speakers in the softmax layer, only for softmax-based losses');

## Load and save
    parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
    parser.add_argument('--save_path',      type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/save_path", help='Path for model and logs');

## Training and test data
#parser.add_argument('--train_list',     type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/train_list.txt",  help='Train list');
    parser.add_argument('--train_list',     type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/vox1_train.txt",  help='Train list');
    parser.add_argument('--test_list',      type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/test_list.txt",   help='Evaluation list');
#parser.add_argument('--train_path',     type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/voxceleb2", help='Absolute path to the train set');
    parser.add_argument('--train_path',     type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/voxceleb1", help='Absolute path to the train set');
    parser.add_argument('--test_path',      type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/voxceleb1_test", help='Absolute path to the test set');
    parser.add_argument('--musan_path',     type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/musan_split", help='Absolute path to the test set');
    parser.add_argument('--rir_path',       type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition
    parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
    parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
    parser.add_argument('--model',          type=str,   default="ResNetSE34L",     help='Name of model definition');
    parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
    parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## For test only
    parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
    parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text');
    parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
    
    args = parser.parse_args() 
    args.specific_model = "/mnt/E/sea120424/VoxCeleb_trainer/baseline_lite_ap.model" 
    args.test_path = '/mnt/E/sea120424/data/vox1/test/unsilence/wav/'    
    args.sec = 0.5
    args.gpu = 0
    return args

def loadWAV(filename):
    sample_rate, audio  = wavfile.read(filename)
    return audio


def main(args):

    s = SpeakerNet(**vars(args));
    s = WrappedModel(s).cuda(args.gpu)

    trainer = ModelTrainer(s, **vars(args))
    trainer.loadParameters(args.specific_model)

    args.device = "cuda:1"

    test_path = args.test_path
    sec = args.sec

    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())


    f = sorted(iglob(test_path + "*/*/*.wav"))
    f = f[:10]
    logfbankA = {}

    for path in tqdm(f, desc="get logfbank feature"):
        speech = path.split('/')[-2]
        name = path.split('/')[-3]
        if name not in logfbankA:
            logfbankA[name] = {}

        #print(loadWAV(path))
        if speech not in logfbankA[name]:
            logfbankA[name][speech] = loadWAV(path)
        else:
            logfbankA[name][speech] = np.concatenate((logfbankA[name][speech], loadWAV(path)))
        

    logfbank = {}
    for name in logfbankA:
        feature = 0
        for speech in logfbankA[name]:
            if len(speech) > 150:
                continue
            else:
                if name not in logfbank:
                    logfbank[name] = logfbankA[name][speech]
                else:
                    logfbank[name] = np.concatenate((logfbank[name], logfbankA[name][speech]))
            print('feature: ', name, logfbank[name].shape)     
            print('feature: ', name, logfbank[name])     
    '''
    len_list = []
    for name in logfbank:
        for name2 in logfbank:
            len_list.append( (len(logfbank[name])+len(logfbank[name2])) / 16000)
    len_list = list(set(len_list))
    np.save('SD/data.npy', np.array(len_list))
    exit()
    ''' 
    #model = resnet(embedding_size=args.embedding_size, num_classes=args.num_of_class, poolingMethod='attentive')
    #checkpoint = torch.load(args.specific_model, map_location='cpu')
    #model.load_state_dict(checkpoint['state_dict'])
    
    clusterer = SpectralClusterer(
        min_clusters=2,
        max_clusters=2,
        p_percentile=0.95,
        gaussian_blur_sigma=1)
    
    sec2avg = []
    #sec_list = [0.5, 1, 1.2, 1.8]
    

    #for sec in sec_list:
    acc = 0
    total = 0
    feature_bank = []
    
    for i in logfbank:
        print('load', i, 'length:', logfbank[i].shape)
        emb = []
        length = len(logfbank[i])

        # 16000 for 1 sec
        t = int(16000 * sec)
        for start in range(0, length, t):
            if start + t < length:
                #print(i, logfbank[i][:,start:start+100].shape)
                inp = logfbank[i][start:start+t].astype(numpy.float)
                inp = torch.FloatTensor(inp)
                #inp = np.stack((logfbank[i][start:start+32000], logfbank[i][start:start+32000]))
                #inp = np.stack((inp, inp))                
                #print(inp.shape)
                print(inp, inp.shape)
                f = trainer.SD_test(inp)
                #f = feature2embedding(model, torch.from_numpy(logfbank[i][:,start:start+100]))
                #print(f.shape)
                #print(f.data)
                emb.append(f)
        feature_bank.append(emb)

    print('finish loading feature len is ', len(feature_bank), type(feature_bank))
    acc_record = []
    for i in range(len(feature_bank)):
        print('index i', i)
        index = i
        for j in range(index+1, len(feature_bank)):
            print('index j', j)
            
            emb1 = feature_bank[index]
            emb2 = feature_bank[j]
            ans1 = [0] * len(emb1)
            ans2 = [1] * len(emb2)
            emb = []
            emb += emb1[:100]
            emb += emb2[:100]
            emb += emb1[100:]
            emb += emb2[100:]
            ans = []
            ans += ans1[:100]
            ans += ans2[:100]
            ans += ans1[100:]
            ans += ans2[100:]
            
            X = []
            for i in emb:
                X.append(i.squeeze(0).data.numpy())
            X = np.array(X)
            
            label = clusterer.predict(X)
            tmp_acc = 0
            tmp_total = 0
            for i in range(len(label)):
                if label[i] == ans[i]:
                    tmp_acc += 1
                tmp_total += 1
            if tmp_total - tmp_acc > tmp_acc:
                tmp_acc = tmp_total - tmp_acc
            acc += tmp_acc
            total += tmp_total
            print('acc', tmp_acc/tmp_total)
            acc_record.append(tmp_acc/tmp_total)


    print('Accuracy: ', acc/total)    
    print('Number: ', total)    
    np.save(f'SD/acc_{sec}s.npy', np.array(acc_record))
    sec2avg.append(acc/total)
    
    print(sec2avg)
    np.save(f'SD/sec2avg.npy', np.array(sec2avg))

    '''
    plt.figure(figsize=matrix.shape)
    cm = np.array(matrix)
    plot_confusion_matrix(cm, ans)
    plt.show()
    '''

if __name__ == '__main__':
    args = get_args()
    main(args)



