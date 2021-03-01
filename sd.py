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
from pyannote.audio.utils.signal import Binarize


import warnings
warnings.filterwarnings('ignore')


sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')

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
    parser.add_argument('--nClasses',       type=int,   default=1211,   help='Number of speakers in the softmax layer, only for softmax-based losses');

## Load and save
    parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
    parser.add_argument('--save_path',      type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/save_path", help='Path for model and logs');

## Training and test data
    parser.add_argument('--train_list',     type=str,   default="/mnt/E/sea120424/VoxCeleb_trainer/vox1_train.txt",  help='Train list');
    
    
    #Need to set a test list
    parser.add_argument('--test_list',      type=str,   default="test_list.txt",   help='Evaluation list');
    '''
    /an/absoulte/path/00001.wav
    /an/absoutle/path/00002.wav
    '''
    
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
    return sample_rate, audio

def TimeCoverter(time):
    ele = time.split(':')
    sec = 0
    sec += float(ele[0]) * 3600
    sec += float(ele[1]) * 60
    sec += float(ele[2])
    return sec

def main(args):

    s = SpeakerNet(**vars(args));
    s = WrappedModel(s).cuda(args.gpu)

    trainer = ModelTrainer(s, **vars(args))
    trainer.loadParameters(args.specific_model)

    args.device = "cuda:1"

    test_path = args.test_path
    sec = args.sec

    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

    test_list = open(args.test_list).readlines()

    for filepath in test_list:
        filepath = filepath.strip()
        print(f"Running {filepath}")
        '''
        Doing VAD
        '''
        test_file = {'uri': filepath, 'audio': filepath}
        
        sad_scores = sad(test_file)
        binarize = Binarize(offset=0.52, onset=0.52, log_scale=True, 
                            min_duration_off=0.1, min_duration_on=0.1)
        speech = binarize.apply(sad_scores, dimension=1)
        seg = str(speech)[1:-1].split('\n')
        #print(seg)
        seg = [i.strip() for i in seg]
        seg = [i.replace("[", '') for i in seg]
        seg = [i.replace("]", '') for i in seg]
        

        sr, audio = loadWAV(filepath)
        label = {}
        emb = []
        index = 0
        clusterer = SpectralClusterer(
            min_clusters=2,
            max_clusters=2,
            p_percentile=0.95,
            gaussian_blur_sigma=1)

        for i in seg:
            time = i.split("-->")
            st = TimeCoverter(time[0].strip())        
            et = TimeCoverter(time[1].strip())        
            #print(f"s: {start_time}, e:{end_time}")
            while et - st > 1.5:
                #label[index] = f"{st} {st + 1.5}"
                label[index] = f"{st} 1.5"
                index += 1
                start = int(st * sr)
                end = int((st+1.5) * sr) 
                inp = audio[start:end].astype(numpy.float)
                inp = torch.FloatTensor(inp)
                #print(inp, inp.shape)
                f = trainer.SD_test(inp)
                emb.append(f)
                st += 2
            
            if et - st <= 0.05:
                continue 

            if et - st <= 2:
                #label[index] = f"{st} {et}"
                label[index] = f"{st} {et-st}"
                index += 1
                start = int(st * sr)
                end = int(et * sr)
                inp = audio[start:end].astype(numpy.float)
                inp = torch.FloatTensor(inp)
                f = trainer.SD_test(inp)
                emb.append(f)
            
        X = []
        for i in emb:
            X.append(i.squeeze(0).data.numpy())
        X = np.array(X)
        
        labels = clusterer.predict(X)
        for index, lab in enumerate(labels):
            #print(f"{lab}: {label[index]}")
            print(f"SPEAKER {filepath} 1 {label[index]} <NA> <NA> {lab} <NA> <NA>")


if __name__ == '__main__':
    args = get_args()
    main(args)



