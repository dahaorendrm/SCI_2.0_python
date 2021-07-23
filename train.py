from imgdataset import Imgdataset
import torch
import torch.nn as nn
from networks.chasti_network import CHASTINET
import scipy.io as scio
from torch.utils.data import DataLoader
import numpy as np
import os
num_epochs = 100
batch_size = 4 
learning_rate = 0.0005


#test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

# data transfer?
# transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
#                                            root_dir='data/faces/',
#                                            transform=transforms.Compose([
#                                                Rescale(256),
#                                                RandomCrop(224),
#                                                ToTensor()
#                                            ]))

#for ind,(gts,inputs) in enumerate(train_dataloader):
#    print(f'Inter {ind} ,shape of gt is {gts.size()}, shape of inputs is {inputs.size()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
model = CHASTINET(4,128,5).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def normalizer(imgs,masks):
    mask_s = torch.sum(masks,3)
    #mask_s = masksi
    #print(f'normalizer mask_s size is {mask_s.size()}; size of imgs is {imgs.size()}')
    mask_s[mask_s==0] = 1
    imgs = imgs/mask_s
    return imgs
# Train the model
def train(data_loader):
    MASK = scio.loadmat('./data/lesti_mask.mat')['mask']
    total_step = len(data_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for ind_batch, (gts, inputs) in enumerate(data_loader): # batch,width,height,channel
            mea = inputs[...,0]
            img_ns = inputs[...,1:]
            masks = torch.tensor(MASK[...,:img_ns.size()[-1]]).float()
            masks = masks.repeat(img_ns.size()[0],1,1,1)
            #print(f'test:{masks.size()}')
            img_n_codes = img_ns*masks # may be not with mask?????????????????????????????????????????????
            output = []
            mea = normalizer(mea,masks)
            mea = torch.unsqueeze(mea,3)

            gts_ = []
            ind_c = 0
            for ind in range(gts.size()[-1]):
                temp = gts[...,ind_c,ind]
                gts_.append(temp)
                ind_c = ind_c+1 if ind_c<2 else 0
            gts = torch.stack(gts_,1)/255.
            gts = gts.to(device)

            for ind in range(img_ns.size()[-1]): # iterative over channel 
                #gt = gts[...,ind]
                img_n = img_ns[...,ind:ind+1]
                img_n_code = torch.sum(img_n_codes[...,:ind],-1) + torch.sum(img_n_codes[...,ind+1:],-1) # sum this two line together, then normalize
                mask_code = torch.sum(masks[...,:ind],-1, keepdim=True) + torch.sum(masks[...,ind+1:],-1, keepdim=True)
                #print(f'size of img_n_code is {img_n_code.size()}; size of mask_code is {mask_code.size()}')
                img_n_code_s = normalizer(img_n_code,mask_code)
                img_n_code_s = torch.unsqueeze(img_n_code_s,-1)
                mask = masks[...,ind:ind+1]
                #print(f'Shape of img_n: {img_n.size()}, img_n_code_begin: {img_n_code_begin.size()}, /nimg_n_code_end: {img_n_code_end.size()}, mask: {mask.size()}, mea: {mea.size()}')
                cat_input = torch.cat((img_n,mea,mask,img_n_code_s),dim=3)
                # Forward pass
                cat_input = np.array(cat_input)
                cat_input = np.moveaxis(cat_input,-1,1)
                cat_input = torch.from_numpy(cat_input).float()
                #print(f'Shap of cat_input is {cat_input.size()}')
                #cat_input = torch.movedim(cat_input,-1,1)
                cat_input = cat_input.to(device)
                output_ = model(cat_input)
                output.append(output_)
            # Backward and optimize
            output = torch.cat(output,1)
            loss = criterion(output, gts) # probably loss per frame/ add all frames loss together
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (ind_batch) % 4 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                       .format(epoch+1, num_epochs, ind_batch+1, total_step, loss.item()))
        save_path = './train/epoch_more_depth_wf/'

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), save_path + str(epoch) +".pth")

        # Decay learning rate
        if (epoch+1) % 3 == 0:
            curr_lr /= 2
            update_lr(optimizer, curr_lr)


if __name__ == '__main__':
    path = './data/data/train'
    dataset = Imgdataset(path)
    train_dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    train(train_dataloader)
