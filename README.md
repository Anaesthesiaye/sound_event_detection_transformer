# Sound Event Detection Transformer
## Prepare your data
+ URBANSED Dataset  
  Download [URBAN-SED_v2.0.0](https://zenodo.org/record/1324404/files/URBAN-SED_v2.0.0.tar.gz?download=1) dataset, and 
  change urbansed_dir in config.py to your own URBAN-SED data path. To generate *.tsv file, run
    ```python
    python ./data_utils/collapse_event.py
    ```

 
+ DCASE2019 Task4 Dataset  
Download the dataset from the website of [DCASE](http://dcase.community/), and change dcase_dir in config.py to your own
 DCASE data path. 
## Environment
## 
## Train models
+ To train model on the dataset of URBANSED, run
    ```shell script
    python train_sedt.py
              --gpus $ngpu
              --fusion_strategy 1 2 3 
              --dec_at 
              --weak_loss_coef 3
              --epochs 500 # total epochs
              --epochs_ls 400 # epochs of learning stage
    ```
+ sh scripts/run_dcase.sh  
to train model on the dataset of DCASE2019

## Evaluate models

# SP-SEDT: Self-supervised pretraining  for SEDT 

## Prepare your data
+ DCASE2018 Task5 development dataset  
  Download [the dataset](https://zenodo.org/record/1247102), put the audios in $dcase_dir/audio/train/ and the *.tsv file
   in $dcase_dir/metadata/train/
## Train model
+ To pretrain SEDT, run
```shell script
python train_sedt.py  --self_self_sup
```
You can also download our [pretrained model]()
+ To fine-tune SEDT, run
```shell script
python train_sedt.py  --self_self_sup
```
## Evaluate models  
  Download our [SP-SEDT(E=6)](), then run
  ```shell script
python train_sedt.py  --self_self_sup
```