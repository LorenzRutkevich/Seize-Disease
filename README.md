
# Seize Disease
## A project around histology with state of the art machine learning algorithms.

---
##### Seize Disease includes many models stretching out into the field of computer vision so that histology is more easily accessible all around the world. Altering some of these models has shown to be beneficial in terms of accuracy and reliability. On top of that, the Seize Disease Website includes a full analyzing tool in which one can upload and alter an histology image. In addition to that, the use and introduction of the [OpenFlexure](https://openflexure.org/) microscope into the project is planned. To use this project, one can visit the website which can be found under this [link](https://seize-disease.streamlit.app/) . 

##### The Website can be used to analyze and alter an image the following ways:
- ##### _Tumor Detection_
- ##### _Tissue Detection_
- ##### _Location Detection_
- ##### _Analyzing nuclei:_
	- ##### _Semantic Segmentation (Multi Class)_
	- ##### _Semantic Segmentation (Binary)_
	- ##### _Border Segmentation_
		- ##### _Nuclei counting_
- ##### _Microscopy Image Restoration (empathy on histology)_
- ##### _Microscopy Super Resolution (empathy on histology)_
- ##### Glandular Morphology within Colon Histopathology Images (Binary)
- ##### Glandular Morphology within Colon Histopathology Images (Object Detection)

##### Histology is a crucial field that involves the study of the structure of cells and tissues under a microscope. This area of study has proven to be invaluable in providing insights into various diseases, including cancer. By examining tissue samples under a microscope, pathologists can identify abnormal cells and tissue structures that may indicate disease, enabling early detection and treatment.

##### The potential of machine learning and Seize Disease to transform the field of histology is significant. With the increasing availability of digital pathology tools and whole-slide imaging technology, machine learning algorithms can be trained to recognize and classify tissue structures. This can lead to a more efficient and accurate diagnosis of diseases, which is the focus of my project for the national AI competition in Germany ([BWKI](https://www.bw-ki.de/)) 2023.
---
### *Results* (evaluation):
---
*first row → models shipped with this project*

###### Nuclei Segmentation:

*Multi-Class:*

| Dataset | Model          | loss   | dice score | pixel error | recall | precision | f1     | iou    | jaccard index | year |
| ------- | -------------- | ------ | ---------- | ----------- | ------ | --------- | ------ | ------ | ------------- | ---- |
| Pannuke | HrNetV2 + OCR* | 1.2188 | **0.8814** | 0.0270      | 0.9047 | 0.9302    | 0.9172 | 0.7107 | **0.7892**    |   /   | 
| ...     | SONNET         | /      | 0.824      | /           | /      | /         | /      | /      | 0.686         | 2022      |

*Binary:*

| Dataset | Model          | loss   | dice score | pixel error | recall | precision | f1     | iou    | jaccard index |
| ------- | -------------- | ------ | ---------- | ----------- | ------ | --------- | ------ | ------ | ------------- |
| Pannuke | HrNetV2 + OCR* | 0.2217 | 0.8558     | 0.07797     | 0.9117 | 0.8098    | 0.8573 | 0.7510 | 0.7484        | 

*Edge:*

| Dataset | Model          | loss   | dice score | pixel error | recall | precision | f1     | iou    | jaccard index |
| ------- | -------------- | ------ | ---------- | ----------- | ------ | --------- | ------ | ------ | ------------- |
| Pannuke | HrNetV2 + OCR* | 1.0987 | 0.9388     | 0.0141      | 0.9578 | 0.9578    | 0.9578 | 0.8612 | 0.8850        | 

<br>

###### Location Detection:

| Dataset | Model   | loss   | accuracy | f1     | precision | recall | specificity at sensitivity |
| ------- | ------- | ------ | -------- | ------ | --------- | ------ | -------------------------- |
| Pannuke | TinyVit | 0.0281 | 0.9955   | 0.9948 | 0.9966    | 0.9930 | 1.0                        | 

<br>

###### Tumor/Tissue Detection (Best Model only x/y):

*Tumor + Tissue*

| Dataset         | Model                  | loss   | accuracy   | f1         | precision  | recall | specificity at sensitivity | additional models                  | year |
| --------------- | ---------------------- | ------ | ---------- | ---------- | ---------- | ------ | -------------------------- | ---------------------------------- | ---- |
| NCT-CRC-HE-100K | EfficientNetV2B2 (1/3) | 0.0096 | **0.9975** | **0.9975** | 0.9979     | 0.9972 | **1.0**                    | EfficientNetV2B1, EfficientNetV2B3 | 2021    |
| ...             | Efficientnet-b0        | /      | 0.9559     | 0.9748     | 0.9989     | /      | 0.9945                     | /                                  | 2019     |
| ...             | ResNeXt-50-32x4d       | /      | 0.9546     | 0.9746     | **0.9991** | /      | 0.9943                     | /                                  |    2021  |

| Dataset | Model                    | loss   | accuracy | f1     | precision | recall | auc    | specificity at sensitivity | additional models |
| ------- | ------------------------ | ------ | -------- | ------ | --------- | ------ | ------ | -------------------------- | ----------------- |
| Kather  | ResAttInceptionV4* (1/2) | 0.9002 | 0.9271   | 0.9180 | 0.9449    | 0.8963 | 0.9970 | 0.9999                     | ResAttInceptionV4* (smaller)                   |

*Tumor*

| Dataset                  | Model                               | loss   | accuracy | f1     | precision | recall | specificity at sensitivity | additonal models                     | Year | 
| ------------------------ | ----------------------------------- | ------ | -------- | ------ | --------- | ------ | -------------------------- | ------------------------------------ | ---- |
| ICIAR2018_BACH_Challenge | TinyVit (1/3)                       | 0.5071 | 0.9000   | 0.9274 | 0.9583    | 0.8984 | 1.0                        | EfficientNetV2B1, EffiencientNetV2B2 | 2022     |
| ...                      | Pretrained Resnet-101; Densenet-161 | /      | 0.87     | /      | /         | /      | /                          | /                                    | 2018     |

<br>

###### Microscopy Image Restoration:

| Dataset                            | Model            | loss   | PSNR    | SSIM   | Model Config    | 
| ---------------------------------- | ---------------- | ------ | ------- | ------ | --- |
| Custom (Mixed Microscopy Images)   | NafNet (256x256) | 0.0716 | 28.4709 | 0.837  |filters = 16, middle_block_num = 2, encoder_block_nums = (1,1,1,28), decoder_block_nums= = (1,1,1,1), block_type = NAFBLOCK, drop_out_rate = 0.0     |
| ... *sligtly worse quality images* | NafNet (128x128) | 0.0986 | 25.9457 | 0.7728 | filters = 32, middle_block_num = 1, encoder_block_nums = (1,1,1,7), decoder_block_nums= = (1,1,1,1), block_type = NAFBLOCK, drop_out_rate = 0.05     |

<br>

| Dataset                   | Model            | loss   | PSNR    | SSIM   | Model Config |
| ------------------------- | ---------------- | ------ | ------- | ------ | ------------ |
| Custom (Histology Images) | NafNet (256x256) | 0.0647 | 30.9962 | 0.8412 | filters = 16, middle_block_num = 2, encoder_block_nums = (1,1,1,28), decoder_block_nums= = (1,1,1,1), block_type = NAFBLOCK, drop_out_rate = 0.0             |
| ...                       | NafNet (128x128) | 0.0587 | 29.622  | 0.8703 |filters = 32, middle_block_num = 1, encoder_block_nums = (1,1,1,7), decoder_block_nums= = (1,1,1,1), block_type = NAFBLOCK, drop_out_rate = 0.05              |


###### Microscopy Image Super Resolution

| Dataset | Model       | loss   | PSNR    | SSIM   | Additions                                                  | Resolution | Additional Models |
| ------- | ----------- | ------ | ------- | ------ | ---------------------------------------------------------- | ---------- | ----------------- |
| Custom (Mixed Microscopy Images)  | HAT (small) | 0.0382 | 29.6236 | 0.8663 | *Images were also slightly blurred, compressed and noised* | 128 -> 256 | HAT (Mid)         |
| ...  | HAT (small) | 0.0272 | 26.7771 | 0.9039 | / | 64 -> 128 | /    |
|...  | HAT (small) | 0.0623 | 23.7196 | 0.7859 | / | 64 -> 256 | /      |

<br>

###### Glandular Morphology within Colon Histopathology Images 

*Binary Segmentation*

| Dataset                                | Model     | loss   | Dice/F1 | Recall | Precison | Additional Models |  
| -------------------------------------- | --------- | ------ | ------- | ------ | -------- | ----------------- |
| Colorectal Adenocarcinoma Gland (CRAG) | HystoSeg* | 0.1567 | 0.8433  | 0.8018 | 0.8922   | VitaeV2 + OCR*    |


*Binary Object Detection*

| Dataset                                | Model     | loss   |  
| -------------------------------------- | --------- | ------ |
| Colorectal Adenocarcinoma Gland (CRAG) | FasterRCNN RegNet Y 400MF | 0.8131 |

<br>

### *References*:
----
#### Datasets used:
- ##### Kather Texture 2016 Image Tiles: Download [here](https://zenodo.org/record/53169)
- ##### ICIAR 2018 BACH: Download [here](https://iciar2018-challenge.grand-challenge.org/)
- ##### PanNuke: Download [here](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke), [Paper](https://arxiv.org/abs/2003.10778)
- ##### NCT-CRC-HE-100K: Download [here](https://zenodo.org/record/1214456)
- ##### CPM 15, CPM 17: Download [here](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK)
- ##### TNBC: Download [here](https://zenodo.org/record/1175282#.YMisCTZKgow)
- ##### Kumar: Download [here](https://monuseg.grand-challenge.org/Data/)
- ##### CoNSeP: Download [here](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/), [Paper](https://arxiv.org/pdf/1812.06499v5.pdf)
- ##### "Potato Tuber" <small></small> <small>*included in Super Resolution/Restoration Dataset only*</small> : Download [here](https://figshare.com/articles/dataset/A_large-scale_optical_microscopy_image_dataset_of_potato_tuber_for_deep_learning_based_plant_cell_assessment/12206270/1) 
- ##### Malaria Bounding Boxes <small>*included in Super Resolution/Restoration Dataset only*</small>: Download [here](https://www.kaggle.com/datasets/kmader/malaria-bounding-boxes)
- ##### Restoration/Super Resolution Dataset (self-made out of all of the above datasets)

#### Code used:
- ##### HRNetV2 Backbone: https://github.com/noelcodella/HRNetV2_keras_tensorflow_semisupervised @[Noel Codella](https://github.com/noelcodella)
- ##### OCR: https://github.com/Burf/HRNetV2-OCR-Tensorflow2 @[Burf](https://github.com/Burf)
- ##### Efficient NetV2: https://github.com/leondgarse/keras_efficientnet_v2 @[leondgarse](https://github.com/leondgarse)
- ##### Data Preparation for Merged Dataset: https://github.com/nauyan/NucleiSegmentation @ [nauyan](https://github.com/nauyan)
- ##### Data Preparation for PanNuke: https://github.com/Mr-TalhaIlyas/Prerpcessing-PanNuke-Nuclei-Instance-Segmentation-Dataset @[Mr-TalhaIlyas](https://github.com/Mr-TalhaIlyas) 
- ##### Mish Activation Function: https://www.kaggle.com/code/imokuri/mish-activation-function/notebook @[SUGIYAMA Yoshio](https://www.kaggle.com/imokuri)
- ##### Segment Anything: https://segment-anything.com/ @[Meta AI](https://ai.facebook.com/tools/), [Github](https://github.com/facebookresearch/segment-anything)
- ##### Tiny Vit: https://github.com/taki0112/vit-tensorflow/tree/main @[taki0112](https://github.com/taki0112)
- ##### StarDist: https://github.com/stardist/stardist @[StarDist](https://github.com/stardist?type=source)
- ##### NafNet: https://github.com/dslisleedh/NAFNet-tensorflow2 @[dslisleedh](https://github.com/dslisleedh)
- ##### NafNet/Restorer and SR Losses: https://github.com/TrystAI/restorers @[TrystAI](https://github.com/TrystAI)
- ##### HAT: https://github.com/XPixelGroup/HAT @ [XPixelGroup](https://github.com/XPixelGroup)
- ##### FasterRCNN: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/tree/main @[sovit123](https://github.com/sovit-123) 
- ##### VitaeV2: https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main 
- ##### HistoSeg: https://github.com/saadwazir/HistoSeg  @[saadwazir](https://github.com/saadwazir/HistoSeg)

#### Comparisions:
- ##### Pannuke Dataset Comparisions: https://paperswithcode.com/dataset/pannuke | [SONNET](https://paperswithcode.com/paper/sonnet-a-self-guided-ordinal-regression)
- ##### NCT-CRC-HE-100K Dataset Comparisions: https://paperswithcode.com/sota/medical-image-classification-on-nct-crc-he | [EfficientNet](https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for)
- ##### ICIAR 2018 BACH Dataset Comparisions: https://figshare.com/articles/dataset/Results_on_the_ICIAR_2018_challenge_dataset_/19676718/1; https://arxiv.org/pdf/1808.04277.pdf

###### Further Sources and Citations can be found in the code itself.
