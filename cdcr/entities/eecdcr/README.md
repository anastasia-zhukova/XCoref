# Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution

## Introduction
This code for [EECDCR](https://github.com/shanybar/event_entity_coref_ecb_plus) was originally used in the paper:

<b>"Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution"</b><br/>
Shany Barhom, Vered Shwartz, Alon Eirew, Michael Bugert, Nils Reimers and Ido Dagan. ACL 2019. 
(https://www.aclweb.org/anthology/P19-1409/)

A neural model implemented in PyTorch for resolving cross-document entity and event coreference.
The model was trained and evaluated on the ECB+ corpus.

The code belongs to [Shany Barhom](https://github.com/shanybar) (*shanyb21@gmail.com*) and was used inside the Xcoref 
as one of the baselines. Original github project is https://github.com/shanybar/event_entity_coref_ecb_plus. 
Please contact Shany for questions about the original model and code.

## Setup for the CDCR pipeline
* [PyTorch](https://pytorch.org/) 
    * PyTorch 0.4.0 with CUDA 9.0 on Linux can be installed using the command:
    `pip install https://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl`
    * torch 1.2.0 and torchvision 0.4.0 for different OS: https://pytorch.org/get-started/previous-versions/#v120
    `pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html --user` 
    (if you experience troubles, try without "user" option and then with it)
    If torchvision does not show up in a list of installed packages, run `pip install torchvision` and 
    then rerun the installation with the version above.
* [AllenNLP](https://allennlp.org/) `pip install allennlp==0.9.0`

## Download models and libraries
Download the following [archive](https://drive.google.com/file/d/197jYq5lioefABWP11cr4hy4Ohh1HMPGK/view), exract the files,
 and place the models cd_entity_best_model and cd_event_best_model into ```./resources/eecdcr_models```. 
 
The other model files will be downloaded directly from the code or can be downloaded manually: 
* ELMO 
1) [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)
2) [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json)
3) Place both files into ```./resources/word_vector_models/ELMO_Original_55B```
* BERT SRL 
1) [model](https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz) 
2) Place the model into ```./resources/word_vector_models/BERT_SRL```
