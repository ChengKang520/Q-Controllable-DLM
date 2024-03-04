


# Using Quantized Embedding Vectors to Optimize Controllable Diffusion Language Models

The code of [Diffusion Language Model]() is in the improved-diffusion/improved_diffusion branch, and the code of [Control Component]() is in the improved-diffusion/control_gen branch, it could help to control the generated text according to the task requirement.


## Overview

This is the official repo for the paper: [Quantized Embedding Vectors for Controllable Diffusion Language Models](https://arxiv.org/abs/2402.10107).

> QE-Diffusion Controllable LM is based on a Controllable Diffusion Language Model whose latent space is modeled by the 
> Denoising Diffusion Probabilistic Model (DDPM), constrained by task requirement (such as, topic, grammar, length and 
> so on), and modified by the rounding process to bridge the discrete text and the continuous input. Quantization process, 
> especially fixed-quantization on embedding vectors can decease the complexity of Controllable DLM's embedding space. 
> But it cannot improve DLMs because their embedding space need higher complexity. Compared to previous controllable 
> text generation models, this method not only can decease the 
> perplexity of generated text, but also can theoretically accelerate the inference speed.  


## Requirements

```
pip install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install -e loralib/
pip install spacy
pip install datasets 
pip install huggingface_hub
pip install wandb
```

## Data Preparing

### E2E

```
│datasets/
├──e2e_data/
│  ├── captions_train2014.json
│  ├── captions_val2014.json
├──train2014/
│  ├── train2014/
│  │   ├── COCO_train2014_000000000009.jpg
│  │   ├── ......
├──val2014/
│  ├── val2014/
│  │   ├── COCO_val2014_000000000042.jpg
│  │   ├── ......
```

### ROCstory

```
│CUB-200/
├──images/
│  ├── 001.Black_footed_Albatross/
│  ├── 002.Laysan_Albatross
│  ├── ......
├──text/
│  ├── text/
│  │   ├── 001.Black_footed_Albatross/
│  │   ├── 002.Laysan_Albatross
│  │   ├── ......
├──train/
│  ├── filenames.pickle
├──test/
│  ├── filenames.pickle
```

### WikiText

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Pretrained Model
We release four text-to-image pretrained model, trained on [Conceptual Caption](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/CC_pretrained.pth?sv=2019-12-12&st=2022-03-09T01%3A55%3A06Z&se=2028-04-10T01%3A55%3A00Z&sr=b&sp=r&sig=KOklHEXv2R3cw64BQv2XmLst2pocejAZEGsxSR%2BkMDI%3D), [MSCOCO](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/coco_pretrained.pth?sv=2019-12-12&st=2022-03-09T01%3A56%3A12Z&se=2028-03-10T01%3A56%3A00Z&sr=b&sp=r&sig=1%2B9tk%2FQVOtDUn81gBDLfxtvR8lbHO0WwxdvQwO7SfMo%3D), [CUB200](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/cub_pretrained.pth?sv=2019-12-12&st=2022-03-09T01%3A56%3A38Z&se=2028-03-10T01%3A56%3A00Z&sr=b&sp=r&sig=LCVsTdNdlyTONgNuQeYJgrg%2BeWHLubD%2FSfwbv3z%2B5bI%3D), and [LAION-human](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/human_pretrained.pth?sv=2019-12-12&st=2022-03-09T01%3A56%3A57Z&se=2028-03-10T01%3A56%3A00Z&sr=b&sp=r&sig=Y%2BAxlxTQfJcUIK8GZxcDRmRixaNZgUKKxBXkOKS%2FNyg%3D) datasets. Also, we release the [ImageNet](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/imagenet_pretrained.pth?sv=2019-12-12&st=2022-03-09T01%3A57%3A25Z&se=2028-03-10T01%3A57%3A00Z&sr=b&sp=r&sig=QdrjMT7B2K3W1Vk6spjzWpFLGCTTVp5cziNp3qEHpxk%3D) pretrained model, and provide the [CLIP](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/ViT-B-32.pt?sv=2019-12-12&st=2022-03-09T01%3A57%3A52Z&se=2028-04-10T01%3A57%3A00Z&sr=b&sp=r&sig=bj5P0BbkreoGdbjDK4sZ5tis%2BwltrVAiN9DQdmzHpEE%3D) pretrained model for convenient. These should be put under OUTPUT/pretrained_model/ .
These pretrained model file may be large because they are training checkpoints, which contains gradient information, optimizer information, ema model and others.

Besides, we provide the VQVAE models on [FFHQ](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/vqgan_ffhq_f16_1024.pth?sv=2019-12-12&st=2022-03-09T01%3A58%3A54Z&se=2028-03-10T01%3A58%3A00Z&sr=b&sp=r&sig=%2BQJZYWrSdiEODji%2B86B8c7QyyWS2PBQx0ivSo8PX338%3D), [OpenImages](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/taming_f8_8192_openimages_last.pth?sv=2019-12-12&st=2022-03-09T01%3A59%3A19Z&se=2028-03-10T01%3A59%3A00Z&sr=b&sp=r&sig=T9d9A3bZVuSgGXYCYesEq9egLvMS0Gr7A4h6MCkiDcw%3D), and [imagenet](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/vqgan_imagenet_f16_16384.pth?sv=2019-12-12&st=2022-03-09T01%3A59%3A42Z&se=2028-03-10T01%3A59%3A00Z&sr=b&sp=r&sig=H%2FQ099FqkYVec7hukfzJF3w6SS%2BpjmzUpuzjsKREoug%3D) datasets, these model are from [Taming Transformer](https://github.com/CompVis/taming-transformers), we provide them here for convenient. Please put them under OUTPUT/pretrained_model/taming_dvae/ .





## Inference
To generate image from given text:
```
from inference_VQ_Diffusion import VQ_Diffusion
VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/human_pretrained.pth')
VQ_Diffusion_model.inference_generate_sample_with_condition("a beautiful smiling woman",truncation_rate=0.85, save_root="RESULT",batch_size=4)
VQ_Diffusion_model.inference_generate_sample_with_condition("a woman in yellow dress",truncation_rate=0.85, save_root="RESULT",batch_size=4,fast=2) # for fast inference
```
You may change human_pretrained.pth to other pretrained model to test different text.

To generate image from given ImageNet class label:
```
from inference_VQ_Diffusion import VQ_Diffusion
VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_imagenet.yaml', path='OUTPUT/pretrained_model/imagenet_pretrained.pth')
VQ_Diffusion_model.inference_generate_sample_with_class(407,truncation_rate=0.86, save_root="RESULT",batch_size=4)
```

## Training
First, change the data_root to correct path in configs/coco.yaml or other configs.

Train Text2Image generation on MSCOCO dataset:
```
python running_command/run_train_coco.py
```

Train Text2Image generation on CUB200 dataset:
```
python running_command/run_train_cub.py
```

Train conditional generation on ImageNet dataset:
```
python running_command/run_train_imagenet.py
```

Train unconditional generation on FFHQ dataset:
```
python running_command/run_train_ffhq.py
```

## Cite VQ-Diffusion
if you find our code helpful for your research, please consider citing:
```
@article{gu2021vector,
  title={Vector Quantized Diffusion Model for Text-to-Image Synthesis},
  author={Gu, Shuyang and Chen, Dong and Bao, Jianmin and Wen, Fang and Zhang, Bo and Chen, Dongdong and Yuan, Lu and Guo, Baining},
  journal={arXiv preprint arXiv:2111.14822},
  year={2021}
}
```
## Acknowledgement
Thanks to everyone who makes their code and models available. In particular,

- [Multinomial Diffusion](https://github.com/ehoogeboom/multinomial_diffusion)
- [Taming Transformer](https://github.com/CompVis/taming-transformers)
- [Improved DDPM](https://github.com/openai/improved-diffusion)
- [Clip](https://github.com/openai/CLIP)

### License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information
For help or issues using VQ-Diffusion, please submit a GitHub issue.
For other communications related to VQ-Diffusion, please contact Shuyang Gu (gsy777@mail.ustc.edu.cn) or Dong Chen (doch@microsoft.com).



-----------------------------------------------------
## Train Diffusion-LM:

```cd improved-diffusion; mkdir diffusion_models;```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 200000  --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data " --notes xstart_e2e```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 400000  --seed 101 --noise_schedule sqrt  --in_channel 128 --modality roc --submit no --padding_mode pad  --app "--predict_xstart True --training_mode e2e  --vocab_size 11043  --roc_train ../datasets/ROCstory " --notes xstart_e2e --bsz 64```


-------------------
## Decode Diffusion-LM:
mkdir generation_outputs 

``python scripts/batch_decode.py {path-to-diffusion-lm} -1.0 ema``


------------------- 
## Controllable Text Generation 
First, train the classsifier used to guide the generation (e.g. a syntactic parser) 

``  
python train_run.py --experiment e2e-tgt-tree  --app "--init_emb {path-to-diffusion-lm} --n_embd {16} --learned_emb yes " --pretrained_model bert-base-uncased --epoch 6 --bsz 10
``

Then, we can use the trained classifier to guide generation. 
(currently, need to update the classifier directory in scripts/infill.py. I will clean this up in the next release.)

``python 
python scripts/infill.py --model_path {path-to-diffusion-lm} --eval_task_ 'control_tree' --use_ddim True  --notes "tree_adagrad" --eta 1. --verbose pipe``



-----------------------------------------------------

For details of the methods and results, please refer to our paper. 


```bibtex
@article{kang2024quantized,
  title={Quantized Embedding Vectors for Controllable Diffusion Language Models},
  author={Kang, Cheng and Chen, Xinye and Hu, Yong and Novak, Daniel},
  journal={arXiv preprint arXiv:2402.10107},
  year={2024}
}
```

