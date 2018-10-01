# xray14 thoracic disease classification

Implement and evaluation of xception keras model with xray14 dataset.
After 23 epochs training with vanilla xception model ,I get a roc score close to current second score.

## example of xray14
![example](https://raw.githubusercontent.com/doublechenching/xray14-keras/master/results/xray14_sample.png)


## vanilla xception model roc score
1. roc score comparing with known publish result
![example](https://raw.githubusercontent.com/doublechenching/xray14-keras/master/results/roc_score.png)
2. roc curve
![example](https://raw.githubusercontent.com/doublechenching/xray14-keras/master/results/vanilla_xeception_roc.png)


## problems I found
### 1. offical train_val and test dataset splitting have a heavy bias.when i plot 14 diseases distribution,
there is a heavy unbias.There are total  86523 images in train and val dataset and  25595 images in test dataset, dataset spliting ratio close to 3:1, but disease splited ratio close to 2:1.Thus, if you train with this splitting without any weighted loss function, you will find generalization performance drop sharply.    
In experiment, I used two blocks of attention model integrated in xception model.
I got lower score, but still close to naive xception model, I get average roc score is 0.87 in validation daset.I think my mode is sota. But thing is more than that.When I evaluate test dataset, there is 0.7 roc score decrease. Maybe my model is not rubust enough, but dataset spliting is really problem. I think i will get better result if I split dataset in patient level making distribution more blanced.
#### 14 disease distribution in full dataset
There are total  112120  images.
Infiltration        ---19894  
Effusion            ---13317  
Atelectasis         ---11559  
Nodule              ---6331  
Mass                ---5782  
Pneumothorax        ---5302  
Consolidation       ---4667  
Pleural_Thickening  ---3385  
Cardiomegaly        ---2776  
Emphysema           ---2516  
Edema               ---2303  
Fibrosis            ---1686  
Pneumonia           ---1431  
Hernia              ---227
#### 14 disease distribution in train val dataset
There are total  86523  images.  
Infiltration        ---13782  
Effusion            ---8659  
Atelectasis         ---8280  
Nodule              ---4708  
Mass                ---4033  
Consolidation       ---2852  
Pneumothorax        ---2637  
Pleural_Thickening  ---2242  
Cardiomegaly        ---1707  
Emphysema           ---1423  
Edema               ---1378  
Fibrosis            ---1251  
Pneumonia           ---875  
Hernia              ---141
#### 14 disease distribution in test dataset
There are total  25595  images.  
Infiltration        ---6112  
Effusion            ---4658  
Atelectasis         ---3279  
Pneumothorax        ---2665  
Consolidation       ---1815  
Mass                ---1748  
Nodule              ---1623  
Pleural_Thickening  ---1143  
Emphysema           ---1093  
Cardiomegaly        ---1069  
Edema               ---925  
Pneumonia           ---555  
Fibrosis            ---435  
Hernia              ---86

### 2. attention model focus wired feature
In experiment, i use two blocks of attention model, but found saliency feature is so wired and i still got decent roc sorce
#### Saliency map
![example](https://raw.githubusercontent.com/doublechenching/xray14-keras/master/results/org_image.png)
![example](https://raw.githubusercontent.com/doublechenching/xray14-keras/master/results/saliency_map1.png)
![example](https://raw.githubusercontent.com/doublechenching/xray14-keras/master/results/saliency_map2.png)