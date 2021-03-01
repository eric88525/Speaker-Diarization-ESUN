# Speaker Diarization

## Acknowledgement

The code is based on voxcele\_trainer with some modification. You can view the source code from `https://github.com/clovaai/voxceleb_trainer`

## Tools

- liborsa
- torch
- sklearn
- matplotlib
- numpy
- cv2

## Other Installation

```
pip install -r requirements.txt
pip install spectralcluster
pip install pyannote.audio==1.1.1
wget http://www.robots.ox.ac.uk/~joon/data/baseline_lite_ap.model

```
pyannote is used for VAD, more specific details provided at `https://github.com/pyannote/pyannote-audio`

prepare the path of wavfile in testlist.txt

in testlist.txt
```
/path/to/00001.wav
/path/to/00002.wav
/path/to/00003.wav
```

Other models can be downloaded by
```
wget https://speechscoring.mirlab.org/deliver/required_model_SD.zip
```
There are four model in the provided url
```
CN-Celeb(interview) # Trained by the interview part of CN-Celeb
aishell1 # Trained with aishell1 dataset
3dataset # Trained by aishell1, aidatatang, Magicdata datasets
3dataset_augment # Trained by aishell1, aidatatang, Magicdata datasets and MUSAN as augmentation

```

## Run (only for SD)

```
python sd.py
```

its will output a rttm format prediction to standard output

```
SPEAKER <uri> 1 <start_time> <duration> <NA> <NA> <label> <NA> <NA>

```
