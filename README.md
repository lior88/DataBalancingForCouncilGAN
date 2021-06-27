### in order to train the model run the following line:
    python train.py --config ./Eyeglasses_train.yaml --output_path ./outputs/council_glasses_128_128 --resume

#### change the .yaml file in accordance with the type of domain transfer you wish to train, specifically notice the following 4 lines:
* attributes_root: location of the attributes csv file.
* data_root: location of the dataset images folder.
* different: True if you want to train using the dataset balancing method, false if you want to run test_on_folder or do not want to use the dataset balancing method.
* m2f: True if you want the m2f domain transfer, False if you want the Eyeglasses domain transfer.


### in order to use the model on an images folder run the following line:
    python test_on_folder.py --config ./Eyeglasses_output.yaml --output_path ./outputs/glasses_male_balanced --checkpoint ./outputs/pretrained_glasses_balanced/pretrained_glasses_balanced/checkpoints/01000000 --input_folder ./glasses_male --a2b 1


### in order to use the classifier you need to separate the pictures into 2 different according to the relevent attribute, if we will take the Male Female classifier as an example then we will have the following folders for the training and testing:
  * train/Male
  * train/Female
  * test/Male
  * test/Female

for each set of pictures we would like to check we need to organize them as explained.

### you can use the following line in order to organize the images for you:
    python making_classifier_dataset.py --m2f 1

change the value of m2f to choose which classifier to train. notice that you must create the empty folders yourself prior to running the program.



### to train the classifier use the following line:
    python classifier.py --m2f 1

change the value of m2f to choose which classifier to train.


### to check the output of the classifier use the following line:
    python check_output.py --m2f 1

change the value of m2f to choose which classifier to train.
