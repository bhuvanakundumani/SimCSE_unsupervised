SimCSE (implementation)[https://arxiv.org/abs/2104.08821] of unsupervised data in Pytorch.

Training using unsupervised approach

### Setup environment 
``` 
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt

```

### Data 
For unsupervised SimCSE,  1 million sentences from English Wikipedia are sampled. You can run data/download_wiki.sh to download the dataset.


``` 
source data/download_wiki.sh 

```

Make sure that data folder has wiki1m_for_simcse.txt file.

### Training:

The hyperparamers for model traing and the output directory are set in class Arguments in train.py. Change the variables in the Arguments class for epxerimentation. overwrite flag is set to True by default, so every time the train.py is run, the output directory is overwritten. 

To run, 

``` 
python train.py

```
