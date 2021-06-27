import pandas as pd
import shutil
import argparse
dAttributes = pd.read_csv('/home/danlior/Council-GAN-master/list_attr_celeba.csv')  # change to new location
im_location = '/home/danlior/DatesetBalancing/img_align_celeba/'  # change to new location

parser = argparse.ArgumentParser()
parser.add_argument('--m2f', type=int, default=1, help="1 for m2f classifier 0 for BlondHair classifier ")
opts = parser.parse_args()
m2f = opts.m2f

if m2f:
    folder = '/home/danlior/Council-GAN-master/MaleFemale_classifier/'  # change to new location
    dMaleP = dAttributes[dAttributes['Male'] == 1]
    dMaleN = dAttributes[dAttributes['Male'] == -1]
    dMalePTrain = dMaleP.iloc[:round(0.8 * dMaleP.shape[0])]
    dMalePTest = dMaleP.iloc[round(0.8 * dMaleP.shape[0]):]
    dMaleNTrain = dMaleN.iloc[:round(0.8 * dMaleN.shape[0])]
    dMaleNTest = dMaleN.iloc[round(0.8 * dMaleN.shape[0]):]

    for i in range(len(dMalePTrain)):
        image = dMalePTrain.image_id.iloc[i]
        shutil.copy((im_location + image), (folder + 'train/Male'))

    for i in range(len(dMalePTest)):
        image = dMalePTest.image_id.iloc[i]
        shutil.copy((im_location + image), (folder + 'test/Male'))

    for i in range(len(dMaleNTrain)):
        image = dMaleNTrain.image_id.iloc[i]
        shutil.copy((im_location + image), (folder + 'train/Female'))

    for i in range(len(dMaleNTest)):
        image = dMaleNTest.image_id.iloc[i]
        shutil.copy((im_location + image), (folder + 'test/Female'))
else:
    folder = '/home/danlior/Council-GAN-master/Hair_classifier/'  # change to new location
    dBlondHair = dAttributes[dAttributes['Blond_Hair'] == 1]
    dNotBlondHair = dAttributes[dAttributes['Blond_Hair'] == -1]
    dBlondHairTrain = dBlondHair.iloc[:round(0.8*dBlondHair.shape[0])]
    dBlondHairTest = dBlondHair.iloc[round(0.8*dBlondHair.shape[0]):]
    dNotBlondHairTrain = dNotBlondHair.iloc[:round(0.8*dNotBlondHair.shape[0])]
    dNotBlondHairTest = dNotBlondHair.iloc[round(0.8*dNotBlondHair.shape[0]):]

    for i in range(len(dBlondHairTrain)):
        image = dBlondHairTrain.image_id.iloc[i]
        shutil.copy((im_location + image), (folder + 'train/Blond_Hair'))

    for i in range(len(dBlondHairTest)):
        image = dBlondHairTest.image_id.iloc[i]
        shutil.copy((im_location + image), (folder + 'test/Blond_Hair'))

    for i in range(len(dNotBlondHairTrain)):
        image = dNotBlondHairTrain.image_id.iloc[i]
        shutil.copy((im_location + image), (folder + 'train/Not_Blond_Hair'))

    for i in range(len(dNotBlondHairTest)):
        image = dNotBlondHairTest.image_id.iloc[i]
        shutil.copy((im_location + image), (folder + 'test/Not_Blond_Hair'))

