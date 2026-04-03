# Image Style Transfer with Unpaired Data Using GANs

## Dataset
https://www.kaggle.com/datasets/lyndialu/cyclegan-oilpainting-dataset.
Download the dataset, and place the four subsets under data.

## Train
Run cyclegan_train.py or munit_train.py.
You can change epoch number by setting the EPOCH variable at the top.
You can continue training from a specific checkpoint by setting the RESUME_PATH variable at the top.

## Test
Run cyclegan_test.py or munit_test.py.
You can change the model by setting the CHECKPOINT_PATH variable at the top.
MUNIT has an additional test script, munit_test_paired.py, that tests images in data/test_pairs with reference style images.

## Evaluation
Run evaluate.py to input scores.
Each run will only check one test image.
Scores are saved at scores.txt.
