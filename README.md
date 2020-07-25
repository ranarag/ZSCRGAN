# ZSCRGAN

A GAN-based Expectation-Maximization Model for Zero-Shot Retrieval of Images from Textual Descriptions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/)


## Table of Contents

- [ZSCRGAN](#zscrgan)
  - [Table of Contents](#table-of-contents)
    - [Summary](#summary)
    - [About](#about)
    - [Citation](#citation)
    - [Dependencies](#dependencies)
    - [Hyperparameters and Options](#hyperparameters-and-options)
    - [Data for Demo](#data-for-demo)
    - [Run Demo](#run-demo)
    - [Acknowledgement](#acknowledgement)

### Summary
Implementation of the proposed algorithm in the paper **ZSCRGAN: A GAN-based Expectation-Maximization Model for Zero-Shot Retrieval of Images from Textual Descriptions**(to be presented at the *ACM International Conference on Information and Knowledge Management (CIKM2020)*) by Anurag Roy, Vinay Kumar Verma, Kripabandhu Ghosh, Saptarshi Ghosh. The proposed model performs zero-shot retrieval of images from their textual descriptions . The  following image gives a schematic view of our proposed model:
![ZSCRGAN](ZSCRGAN.png)


### About
ZSCRGAN is a novel Zero-Shot cross modal text to image retrieval model. The model does this by learning a joint probability distribution of text embeddings and relevant image embeddings, maximizing which ensures high similarity between text embeddings and relevant image embeddings. To learn this distribution we use an Expectation-Maximization based training approach of the model involving a Generative Adversarial Network(GAN) which is learnt in the E-step and a Common Space Embedding Mapper(CSEM), which is updated in the M-step. The GAN is used to generate a representative image embedding(a latent variable) and the  CSEM is used map them to a common space embedding.

### Citation
If you use the codes, please refer to the following paper:
```
  @inproceedings{roy-cikm20,
   author = {Roy, Anurag and Verma, Vinay and Ghosh, Kripabandhu and Ghosh, Saptarshi},
   title = {{ZSCRGAN: A GAN-based Expectation-Maximization Model for Zero-Shot Retrieval of Images from Textual Descriptions}},
   booktitle = {{Proceedings of the 29th ACM International Conference on Information and Knowledge Management  (CIKM)}},
   year = {2020}
  }
```

### Dependencies
python version: `python 2.7`

packages: 
- `tensorflow-gpu`
- `easydict`
- `scipy`
- `six`
- `numpy`
- `prettytensor`
- `pyYAML`
- `scikit-learn`

To install the dependencies run `pip install -r requirements.txt`

### Hyperparameters and Options
Hyperparameters and options in  `run_exp.py`:

- `batch_size` batch size used during training
- `gf_dim` hidden layer dimension of generator
- `df_dim` hidden layer dimension of discriminator
- `embed_dim` dimension of mu and sigma each
- `CSEM_lr` learning rate of CSEM
- `generator_lr` learning rate of generator
- `discriminator_lr` learning rate of discriminator
- `epochs` number of epochs
- `kl_div_coefficient` coefficient of the kl-divergence loss
- `mm_reg_coeff` coefficient of the max-margin regularizer
- `z_dim` dimension of the noise vector
- `clip_val` clipping values of the discriminator in WGAN
- `dataset` raining dataset folder name



### Data for Demo
Following are the datasets on which our experminets have been run: 

1. [AWA1](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/ERoi6WnOgA5DvwfALnPXmxMBKsx6KJsUQjNcPTpWezfMnA?e=t6qpUs)
2. [AWA2](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/EWdq-DmUmmxAnQYKNxNeBkUBX9aN1g3EMTQ8bFY8LdO5_w?e=KXtgh1)
3. [CUB](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/EZtHWPLWQRhKhpGzlMwGXg0BFKJS89AgThfCTU_d6G9Qhg?e=h1ogbX)
4. [FLO](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/EdiIbbe0g8VCoHXdG-JRHk0BxlObK9m4VYYdFXFQK_tWuA?e=X0ebBw)
5. [NAB(SCS split)](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/EckFegwRkMNJr7lEeiFdheIBrgGJVfTDy-vtyOLJAYWZzg?e=uRDLKH)
6. [NAB(SCE split)](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/EYH2h6ER_y9PqH0VR7Ex5tkB3glrYFAl0RRmiPUhEpdKcw?e=BEq8CP)
7. [WIKI](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/EY21_oBLEsBJnQCkjeXj-woBqFxUsFovpuVHfsp1-wpavA?e=lz3T1I)

Download the zip files and extract it inside the `datasets` folder.

### Run Demo
To run the model on a particular dataset use the command:

`python2 run_exp.py --dataset <dataset_folder_name>`

Precision@50 on the test set will be outputted after every 100 iterations of the E-step and the M-step.
The retrieved results can be found inside the `retrieved_res/<dataset_folder_name>_res/` folder. For example the command to run the model on the `CUB` dataset will be `python2 run_exp.py --dataset CUB`. This will create a folder `retrieved_res/CUB_res/` containing files with the the retrieval results. The name of the file will be of the form `acc<Prec@50>.pkl`. For example the retrieval result which had a Prec@50 value of 0.521 will be saved in the file `acc0.521.pkl`. The output pickle file contains a list of key value pairs with class id of the text embedding being the key and list of class ids of the retrieved images being the value. For example one element of the list will be:

```
{3: [3, 3, 3, 25, 25, 48, 3, 25, 3, 25, 25, 25, 3, 3, 3, 48, 3, 22, 25, 25, 3, 48, 25, 22, 48, 25, 3, 3, 25, 3, 48, 3, 48, 3, 3, 48, 3, 3, 48, 3, 25, 3, 3, 25, 48, 3, 3, 3, 25, 3]}
```
where the key `3` is the class id of the text embedding and the value of list of class ids correspond to those of the retrieved images.



### Acknowledgement

Some parts of the code have been borrowed from the [StackGAN](https://github.com/hanzhanggit/StackGAN) code repository. 
