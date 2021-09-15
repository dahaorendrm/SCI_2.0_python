
'''
*step 1 import data
*step 2 compressive_model input: 25 bands output:8 led bands, a snapshot measurement
*step 3 gap-tv
*step 4 chasti improve
 step 4.5 chasti result

 Multi-thread data preparing
step 5 spectral convertion model
*    5.1 setup model
    5.2 load weight
    5.3 run model

'''

    dataset = Imgdataset(path)
    batch_size = 2
    train_dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    for data in train_dataloader:
        break

def spectral_convertion(inputs, gts):
