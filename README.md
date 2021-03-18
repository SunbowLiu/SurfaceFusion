# Understanding and Improving Encoder Layer Fusion in Sequence-to-Sequence Learning (ICLR 2021)

### Citation

Please cite as:

```bibtex
@inproceedings{liu2020understanding,
  title={Understanding and Improving Encoder Layer Fusion in Sequence-to-Sequence Learning},
  author={Liu, Xuebo and Wang, Longyue and Wong, Derek F and Ding, Liang and Chao, Lidia S and Tu, Zhaopeng},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```


### Requirements and Installation
This implementation is based on [fairseq(v0.9.0)](https://github.com/pytorch/fairseq/tree/v0.9.0/fairseq)

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6

```
git clone https://github.com/SunbowLiu/SurfaceFusion
cd SurfaceFusion
pip install --editable .
```

### Preprocess
Download [WMT16 En-Ro Data](https://drive.google.com/uc?id=1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl) ([Original](https://github.com/nyu-dl/dl4mt-nonauto#downloading-datasets--pre-trained-models))
```
tar -zxvf wmt16.tar.gz
PATH_TO_RAW_DATA=wmt16/en-ro
PATH_TO_DATA=wmt16/en-ro/data-bin
python preprocess.py \
    --source-lang en --target-lang ro \
    --trainpref $PATH_TO_RAW_DATA/train/corpus.bpe \
    --validpref $PATH_TO_RAW_DATA/dev/dev.bpe \
    --testpref $PATH_TO_RAW_DATA/test/test.bpe \
    --destdir $PATH_TO_DATA \
    --joined-dictionary \
    --workers 20
```

### Train (8 gpus)
```
OUTPUT=checkpoints
python train.py \
    $PATH_TO_DATA \
    --arch transformer_surface_fusion --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0005 --min-lr 1e-09 \
    --dropout 0.3  --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --save-dir $OUTPUT --seed 333 --ddp-backend=no_c10d --fp16 \
    --max-tokens 2048 --update-freq 1 --max-update 60000 --keep-last-epochs 1 \
    --surfacefusion att --sf-gate 0.8 --sf-mode hard
```
It is noted that we use 16k batch size, i.e., max-tokens * update-freq * num_of_gpus = 16k.


### Evaluation (1 gpu)
```
python generate.py \
    $PATH_TO_DATA \
    --path $OUTPUT/checkpoint_best.pt \
    --beam 4 --lenpen 1.0 --remove-bpe
```
The model can gain nearly 35.1 BLEU scores.







