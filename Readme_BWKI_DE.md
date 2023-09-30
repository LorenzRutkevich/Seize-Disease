# Seize Disease

### Ein Projekt rund um Histologie mit neusten und veränderten Machine Learning Anwendungen.

---
##### Seize Disease umfasst viele Modelle, die sich im Bereich der Computer Vision erstrecken, um die Histologie weltweit zugänglicher zu machen. Die Anpassung einiger dieser Modelle hat sich als vorteilhaft in Bezug auf Genauigkeit und Zuverlässigkeit erwiesen. Darüber hinaus enthält die Seize Disease-Website ein vollständiges Analysetool, mit dem man ein histologisches Bild hochladen und bearbeiten kann. Zusätzlich dazu ist die Verwendung und Integration des Mikroskops [OpenFlexure](https://openflexure.org/) in das Projekt geplant. Um dieses Projekt zu nutzen, kann man die Website besuchen, die unter folgendem [Link](https://lorenz-7-bwki-hystology-website-beta.streamlit.app/) zu finden ist.

##### Die Website kann verwendet werden, um ein Bild auf folgende Weise zu analysieren und zu bearbeiten:
- ##### _Tumorerkennung_
- ##### _Gewebeerkennung_
- ##### _Ortserkennung_
- ##### _Analyse von Zellkernen:_
	- ##### _Semantische Segmentierung (Mehrklassen)_
	- ##### _Semantische Segmentierung (Binär)_
	- ##### _Border Segmentierung_
		- ##### _Zählung von Zellkernen_
- ##### _Wiederherstellung von Mikroskopiebildern (Schwerpunkt auf Histologie)_
- ##### _Superresolution von Mikroskopiebildern (Schwerpunkt auf Histologie)_
- ##### _Binäre Drüsensegmentierung innerhalb des Dickdarms_
- ##### _Object Detection von Drüsen innerhalb des Dickdarms_
##### Histologie ist ein entscheidendes Fachgebiet, das sich mit der Untersuchung der Struktur von Zellen und Geweben unter dem Mikroskop befasst. Dieses Forschungsgebiet hat sich als unschätzbar erwiesen, um Einblicke in verschiedene Krankheiten, einschließlich Krebs, zu gewinnen. Durch die Untersuchung von Gewebeproben unter dem Mikroskop können Pathologen abnormale Zellen und Gewebestrukturen identifizieren, die auf eine Krankheit hinweisen können, und so eine frühzeitige Erkennung und Behandlung ermöglichen.

##### Das Potenzial von maschinellem Lernen und Seize Disease, das Fachgebiet der Histologie zu transformieren, ist bedeutend. Mit der zunehmenden Verfügbarkeit von digitalen Pathologie-Werkzeugen und Whole-Slide-Imaging-Technologie können maschinelle Lernalgorithmen darauf trainiert werden, Gewebestrukturen zu erkennen und zu klassifizieren. Dies kann zu einer effizienteren und genaueren Diagnose von Krankheiten führen, was der Schwerpunkt meines Projekts für den [BWKI](https://www.bw-ki.de/) 2023 ist.

---
### *Erste Schritte*

#### Lokal:

##### Erstelle zunächst eine Conda- oder Pip-Umgebung, um die erforderlichen Anforderungen zu installieren. Anschließend kannst du die Datensätze herunterladen und sie bei Bedarf vorbereiten. Die Links zu diesen Datensätzen findest du im [Quellen](#quellen)-Abschnitt.

```python
# Mit Pip
pip install -r requirements.txt

# Mit Conda
conda create --name <env_name> --file requirements.txt
```

#### Colab:

##### Wenn du Google Colab verwendest, öffne eine beliebige Projektdatei und stelle sicher, dass die Datei requirements.txt darin vorhanden ist. Führe dann den folgenden Befehl aus:

```
!pip install -r requirements.txt
```

### *Ergebnisse* (Evaluation):
---
*erste Zeile → Modelle mit diesem Projekt mitgeliefert*
**veränderte/angepasste Modelle*

###### Zellkern Segmentation:

*Mehrklassen:*

| Dataset | Model          | loss   | dice score | pixel error | recall | precision | f1     | iou    | jaccard index | year |
| ------- | -------------- | ------ | ---------- | ----------- | ------ | --------- | ------ | ------ | ------------- | ---- |
| Pannuke | HrNetV2 + OCR* | 1.2188 | **0.8814** | 0.0270      | 0.9047 | 0.9302    | 0.9172 | 0.7107 | **0.7892**    |   /   | 
| ...     | SONNET         | /      | 0.824      | /           | /      | /         | /      | /      | 0.686         | 2022      |

*Binär:*

| Dataset | Model          | loss   | dice score | pixel error | recall | precision | f1     | iou    | jaccard index |
| ------- | -------------- | ------ | ---------- | ----------- | ------ | --------- | ------ | ------ | ------------- |
| Pannuke | HrNetV2 + OCR* | 0.2217 | 0.8558     | 0.07797     | 0.9117 | 0.8098    | 0.8573 | 0.7510 | 0.7484        | 

*Instanz:*

| Dataset | Model          | loss   | dice score | pixel error | recall | precision | f1     | iou    | jaccard index |
| ------- | -------------- | ------ | ---------- | ----------- | ------ | --------- | ------ | ------ | ------------- |
| Pannuke | HrNetV2 + OCR* | 1.0987 | 0.9388     | 0.0141      | 0.9578 | 0.9578    | 0.9578 | 0.8612 | 0.8850        | 

<br>

###### Ortserkennung:

| Dataset | Model   | loss   | accuracy | f1     | precision | recall | specificity at sensitivity |
| ------- | ------- | ------ | -------- | ------ | --------- | ------ | -------------------------- |
| Pannuke | TinyVit | 0.0281 | 0.9955   | 0.9948 | 0.9966    | 0.9930 | 1.0                        | 

<br>

###### Tumor/Gewebe Erkennng (Nur bestes Modell x/y):

*Tumor + Gewebe*

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

##### Wiederherstellung von Mikroskopiebildern :

| Dataset                            | Model            | loss   | PSNR    | SSIM   | Model Config    | 
| ---------------------------------- | ---------------- | ------ | ------- | ------ | --- |
| Custom (Mixed Microscopy Images)   | NafNet (256x256) | 0.0716 | 28.4709 | 0.837  |filters = 16, middle_block_num = 2, encoder_block_nums = (1,1,1,28), decoder_block_nums= = (1,1,1,1), block_type = NAFBLOCK, drop_out_rate = 0.0     |
| ... *sligtly worse quality images* | NafNet (128x128) | 0.0986 | 25.9457 | 0.7728 | filters = 32, middle_block_num = 1, encoder_block_nums = (1,1,1,7), decoder_block_nums= = (1,1,1,1), block_type = NAFBLOCK, drop_out_rate = 0.05     |

<br>

| Dataset                   | Model            | loss   | PSNR    | SSIM   | Model Config |
| ------------------------- | ---------------- | ------ | ------- | ------ | ------------ |
| Custom (Histology Images) | NafNet (256x256) | 0.0647 | 30.9962 | 0.8412 | filters = 16, middle_block_num = 2, encoder_block_nums = (1,1,1,28), decoder_block_nums= = (1,1,1,1), block_type = NAFBLOCK, drop_out_rate = 0.0             |
| ...                       | NafNet (128x128) | 0.0587 | 29.622  | 0.8703 |filters = 32, middle_block_num = 1, encoder_block_nums = (1,1,1,7), decoder_block_nums= = (1,1,1,1), block_type = NAFBLOCK, drop_out_rate = 0.05              |


###### Superresolution von Mikroskopiebildern 

| Dataset | Model | loss   | PSNR   | SSIM   |
| ------- | ----- | ------ | ------ | ------ |
| Custom (Mixed Microscopy Images) | HAT   | 0.0321 | 35.502 | 0.9407 | 

<br>

###### Glandula Morphologie innerhalb des Dickdarms

*Binary Segmentation*

| Dataset                                | Model            | loss   | Dice/F1    | Recall     | Precison   | Additional Models | Model Config          |
| -------------------------------------- | ---------------- | ------ | ---------- | ---------- | ---------- | ----------------- | --------------------- |
| Colorectal Adenocarcinoma Gland (CRAG) | HystoSeg*        | 0.1567 | 0.8433     | 0.8018     | 0.8922     | VitaeV2 + OCR*    | backbone = "xception" |
| ...                                    | Custom Model L*   | 0.2533 | **0.8865** | **0.8722** | **0.9013** | /                 | /                     |
| ...                                    | Custom Model S/M* | 0.2724 | 0.854      | 0.8275     | 0.8849     | /                 | /                     |

| ... | HystoSeg DP* | 0.6872 | 0.7306 | 0.6843 | 0.7836 | /   | based on HystoSeg mobilenetv2 | 
| --- | ------------ | ------ | ------ | ------ | ------ | --- | ----------------------------- |
| ... | HystoSeg*    | 0.5615 | 0.701  | 0.6589 | 0.7514 | /   | backbone = "mobilenetv2"      |     

*Object Detection*

| Dataset                                | Model     | loss   |  
| -------------------------------------- | --------- | ------ |
| Colorectal Adenocarcinoma Gland (CRAG) | FasterRCNN RegNet Y 400MF | 0.8131 |


### *Quellen*:
----
#### Benutzte Datensätze:
- ##### Kather Texture 2016 Image Tiles: Download [here](https://zenodo.org/record/53169)
- ##### ICIAR 2018 BACH: Download [here](https://iciar2018-challenge.grand-challenge.org/)
- ##### PanNuke: Download [here](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke), [Paper](https://arxiv.org/abs/2003.10778)
- ##### NCT-CRC-HE-100K: Download [here](https://zenodo.org/record/1214456)
- ##### CPM 15, CPM 17: Download [here](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK)
- ##### TNBC: Download [here](https://zenodo.org/record/1175282#.YMisCTZKgow)
- ##### Kumar: Download [here](https://monuseg.grand-challenge.org/Data/)
- ##### CoNSeP: Download [here](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/), [Paper](https://arxiv.org/pdf/1812.06499v5.pdf)
- ##### "Potato Tuber" <small></small> <small>*Nur im Super Resolution/Restoration-Datensatz enthalten.*</small> : Download [here](https://figshare.com/articles/dataset/A_large-scale_optical_microscopy_image_dataset_of_potato_tuber_for_deep_learning_based_plant_cell_assessment/12206270/1) 
- ##### Malaria Bounding Boxes <small>*Nur im Super Resolution/Restoration-Datensatz enthalten.*</small>: Download [here](https://www.kaggle.com/datasets/kmader/malaria-bounding-boxes)
- ##### Restoration/Super Resolution Dataset (Selbst erstellt aus allen oben genannten Datensätzen)
###### _Die selbst erstellten Datensätze können von meiner Drive runtergeladen werden:_
- ###### [Restoration Dataset](https://drive.google.com/file/d/1jPz7W1MWWE0iGf6PyadSJRkClB1pdeQ_/view?usp=sharing)
- ###### [Super Resolution 256/128 Mixed Microscopy](https://drive.google.com/file/d/1aJYnvFXh-m-HjA7YMT6J_mECRKPGZbt6/view?usp=sharing)
- ###### [Super Resolution 256/128 Histology Only](https://drive.google.com/file/d/1imWuRFz7uZuS1lB5DZUajBL2AcXT5tRY/view?usp=sharing)

#### Benutzer Code:
- ##### HRNetV2 Backbone: https://github.com/noelcodella/HRNetV2_keras_tensorflow_semisupervised @[Noel Codella](https://github.com/noelcodella)
- ##### OCR: https://github.com/Burf/HRNetV2-OCR-Tensorflow2 @[Burf](https://github.com/Burf)
- ##### Efficient NetV2: https://github.com/leondgarse/keras_efficientnet_v2 @[leondgarse](https://github.com/leondgarse)
- ##### Data Preparation for Merged Dataset: https://github.com/nauyan/NucleiSegmentation @ [nauyan](https://github.com/nauyan)
- ##### Data Preparation for PanNuke: https://github.com/Mr-TalhaIlyas/Prerpcessing-PanNuke-Nuclei-Instance-Segmentation-Dataset @[Mr-TalhaIlyas](https://github.com/Mr-TalhaIlyas) 
- ##### Mish Activation Function (TF): https://www.kaggle.com/code/imokuri/mish-activation-function/notebook @[SUGIYAMA Yoshio](https://www.kaggle.com/imokuri)
- ##### Segment Anything: https://segment-anything.com/ @[Meta AI](https://ai.facebook.com/tools/), [Github](https://github.com/facebookresearch/segment-anything)
- ##### Tiny Vit: https://github.com/taki0112/vit-tensorflow/tree/main @[taki0112](https://github.com/taki0112)
- ##### StarDist: https://github.com/stardist/stardist @[StarDist](https://github.com/stardist?type=source)
- ##### NafNet: https://github.com/dslisleedh/NAFNet-tensorflow2 @[dslisleedh](https://github.com/dslisleedh)
- ##### NafNet/Restorer and SR Losses: https://github.com/TrystAI/restorers @[TrystAI](https://github.com/TrystAI)
- ##### HAT: https://github.com/XPixelGroup/HAT @ [XPixelGroup](https://github.com/XPixelGroup)
- ##### FasterRCNN: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/tree/main @[sovit123](https://github.com/sovit-123) 
- ##### VitaeV2: https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main 
- ##### HistoSeg: https://github.com/saadwazir/HistoSeg  @[saadwazir](https://github.com/saadwazir/HistoSeg)

#### Vergleiche:
- ##### Pannuke: https://paperswithcode.com/dataset/pannuke | [SONNET](https://paperswithcode.com/paper/sonnet-a-self-guided-ordinal-regression)
- ###### NCT-CRC-HE-100K: https://paperswithcode.com/sota/medical-image-classification-on-nct-crc-he | [EfficientNet](https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for)
- ###### ICIAR 2018 BACH: https://figshare.com/articles/dataset/Results_on_the_ICIAR_2018_challenge_dataset_/19676718/1; https://arxiv.org/pdf/1808.04277.pdf

###### Weitere Zitate und Quellen können in dem Code selber aufgefunden werden

