# NSCLC_ResNet
 Haowen Zhou, Mark Watson, Cory T. Bernadt, Steven (Siyu) Lin, Chieh-yu Lin, Jon. H. Ritter, Alexander Wein, Simon Mahler, Sid Rawal, Ramaswamy Govindan, Changhuei Yang*, and Richard J. Cote*. "AI-guided histopathology predicts brain metastasis in lung cancer patients." Journal of Pathology (In review).

## Abstract
Brain metastases can occur in nearly half of patients with early and locally advanced (stage I-III) non-small cell lung cancer (NSCLC). There are no reliable histopathologic or molecular means to identify those who are likely to develop brain metastases. We sought to determine if deep learning (DL) could be applied to routine hematoxylin and eosin (H&E) stained primary tumor tissue sections from Stage I-III NSCLC patients to predict the development of brain metastasis. Diagnostic slides from 158 patients with Stage I to III NSCLC followed for at least 5 years for development of brain metastases (met+, 65 pts) vs. no progression (Met–, 93 pts) were subjected to whole slide imaging (WSI). Three separate iterations were performed by first selecting 118 cases (45 met+, 73 Met–) to train and validate the DL algorithm, while 40 separate cases (20 met+, 20 Met–) were used as the test set. The results using the DL algorithm were compared to a blinded review by 4 expert pathologists. The DL-based algorithm was able to distinguish eventual development of brain metastases with an accuracy of 87% (p<0.0001) vs. an average of 57% by pathologists (p=NS). The DL algorithm appears to focus on a complex set of histological features. DL based algorithms using routine H&E-stained slides may identify patients likely to develop brain metastases vs. those that will remain disease free over extended (>5 year) follow-up and may thus be spared systemic therapy. 

## Dataset
The dataset is available at CaltechData: https://doi.org/10.22002/dw66e-mbs82

If the downloaded dataset is placed in this GitHub repo main folder. Then, the code does not need any changes.

If the dataset is placed at other directory. Please replace "Cpath" in the code with your directory.

## Environment
The Anaconda environment yaml file is provided to run this code. The code can be adapted to Linux or Windows systems. The python depedencies can be installed by creating the anaconda environment from environment.yaml file.

## scripts
To run the script in python editors, click run.

To run the script in terminal:
```
python xxxxx.py
```

"Training_with_3dfold_cross_validation.py" is to perform training with 3-fold cross validation. 

"Train_on_entire_training_set.py": Once the train-validation is done, the train-validation sets are combined as training sets. This script performs the three train-test splits.

"Testing.py" is to get the testing results.

## Trained models
The trained models for the three train-test splits are available at CaltechData: https://doi.org/10.22002/dw66e-mbs82

For loading and using the trained models, only "Testing.py" needs to be run. 

## Citation (Bibitex)

To be released.
