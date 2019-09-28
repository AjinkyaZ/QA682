# CS682 Project - Deep Learning for Question Answering  

## Contributors   
Ajinkya Zadbuke, Reetika Roy  

## Setup

### Environment and dependencies
1. Create a virtual environment
```
$ pipenv --python 3.6
```
2. Activate environment
```
$ pipenv shell
```
3. Install dependencies
```
$ pipenv install
```
4. Move to src directory
```
$ cd src
```


### Preprocess data
```
$ python prepro.py
```

### Train model
```
$ python train.py [ARGS]
```
Arguments:  
'-nt', '--num_train', default=1024, type=int  
'-nv', '--num_val', default=256, type=int  
'-bs', '--batch_size', default=32, type=int  
'-e', '--epochs', default=100, type=int  
'-lr', '--learning_rate', default=1e-3, type=float  
'-o', '--optimizer', default='Adam'  
'-nl', '--n_layers', default=1, type=int  
'-s', '--save_every', default=5, type=int  
'-hs', '--hidden_size', default=64, type=int  
'-ld', '--linear_dropout', default=0.3, type=float  
'-ls', '--seq_dropout', default=0.0, type=float  


### Test and evaluate model
```
$ python test.py [ARGS]
```
Arguments:  
'-nt', '--num_train', default=1024, type=int  
'-m', '--model', type=str  

Run evaluation script:  
```
python ../evaluation/evaluate-v1.1.py ../data/dev-v1.1.json ../data/run_<MODEL_NAME>.json
```






