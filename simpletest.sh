#!/bin/bash
# python2 run.py --runner RecurrentNetworkTorchRunner --data_path data/mimic2 --batch_size 2 --word_emb_size 32 --label_emb_size 32 --hidden_size 32 --max_note_len 10 --max_dgn_labels 50 --print_every 1
python2 run.py --runner ConvEncoderRunner --data_path data/mimic2 --batch_size 2 --word_emb_size 64 --label_emb_size 64 --hidden_size 64 --max_note_len 10 --max_dgn_labels 50 --print_every 1 --save_every -1 --max_steps 100 --optimizer nag --learning_rate 0.25 --max_grad_norm 0.1

# add dropout

#[akv245@gpu-04 ~]$ module load pytorch/intel/20170724
#[akv245@gpu-04 ~]$ python
#Python 2.7.12 (default, Nov  4 2016, 12:33:11) 
#[GCC Intel(R) C++ gcc 4.8.5 mode] on linux2
#Type "help", "copyright", "credits" or "license" for more information.
#>>> import torch
#>>> torch.__version__
#'0.2.0+4a4d884'
#>>> a = torch.zeros([4,4])
#>>> a

# 0  0  0  0
# 0  0  0  0
# 0  0  0  0
# 0  0  0  0
#[torch.FloatTensor of size 4x4]

#>>> a.cuda()

# 0  0  0  0
# 0  0  0  0
# 0  0  0  0
# 0  0  0  0
#[torch.cuda.FloatTensor of size 4x4 (GPU 0)]


#python run.py --runner ConvEncoderRunner --emb_file saved/m3p_word2vec_192.dat --save_every -1 --threads 3 --dropout 0.1 --max_note_len 4000 --max_dgn_labels 4000 --max_pcd_labels 1000 --batch_size 16 --print_every 1 --hidden_size 256 --layers 20 --curriculum
#python run.py --runner ConvEncoderRunner --emb_file saved/m3p_word2vec_192.dat --save_every -1 --threads 3 --dropout 0.1 --max_note_len 4000 --max_dgn_labels 4000 --max_pcd_labels 1000 --batch_size 16 --print_every 1 --hidden_size 128 --layers 20 --curriculum
#python run.py --runner ConvEncoderRunner --emb_file saved/m3p_word2vec_192.dat --save_every -1 --threads 3 --dropout 0.1 --max_note_len 4000 --max_dgn_labels 4000 --max_pcd_labels 1000 --batch_size 16 --print_every 1 --hidden_size 64 --layers 20 --curriculum
#python run.py --runner ConvEncoderRunner --emb_file saved/m3p_word2vec_192.dat --save_every -1 --threads 3 --dropout 0.1 --max_note_len 4000 --max_dgn_labels 4000 --max_pcd_labels 1000 --batch_size 16 --print_every 1 --hidden_size 256 --layers 10 --curriculum
#python run.py --runner ConvEncoderRunner --emb_file saved/m3p_word2vec_192.dat --save_every -1 --threads 3 --dropout 0.1 --max_note_len 4000 --max_dgn_labels 4000 --max_pcd_labels 1000 --batch_size 16 --print_every 1 --hidden_size 256 --layers 5 --curriculum

