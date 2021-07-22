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

path = './train/data'
dataset = Imgdataset(path)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
model = CHASTINET(4,128).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def normalizer(imgs,masks):
    mask_s = torch.sum(masks,3)
    mask_s[mask_s==0] = 1
    imgs = imgs/mask_s
    return imgs
# Train the model
def train(data_loader):
    MASK = scio.loadmat('./train/lesti_mask.mat')['mask']
    total_step = len(data_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for ind_batch, (gts, inputs) in enumerate(data_loader): # batch,weight,height,channel
            mea = inputs[...,0]
            img_ns = inputs[...,1:]
            masks = torch.tensor(MASK[...,:img_ns.size()[-1]]).float()
            masks = masks.repeat(img_ns.size()[0],1,1,1)
            img_n_codes = img_ns*masks # may be not with mask?????????????????????????????????????????????
            output = []
            mea = normalizer(mea,masks)
            mea = torch.unsqueeze(mea,3)
            for ind in range(img_ns.size()[-1]):
                #gt = gts[...,ind]
                img_n = img_ns[...,ind:ind+1]
                img_n_code = torch.sum(img_n_codes[...,:ind],-1) + torch.sum(img_n_codes[...,ind+1:],-1) # sum this two line together, then normalize
                masks_code = [masks[...,ind_m] for ind_m in range(img_ns.size()[-1]) if ind_m != ind]
                masks_code = torch.stack(masks_code,3)
                img_n_code_s = normalizer(img_n_code,masks_code)
                img_n_code_s = torch.unsqueeze(img_n_code_s,3)
                mask = masks[...,ind:ind+1]
                #print(f'Shape of img_n: {img_n.size()}, img_n_code_s: {img_n_code_s.size()}, mask: {mask.size()}, mea: {mea.size()}')
                cat_input = torch.cat((img_n,mea,mask,img_n_code_s),dim=3)
                # Forward pass
                cat_input = np.array(cat_input)
                cat_input = np.moveaxis(cat_input,-1,1)
                cat_input = torch.from_numpy(cat_input).float()
                #print(f'Shap of cat_input is {cat_input.size()}')
                #cat_input = torch.movedim(cat_input,-1,1)
                cat_input = cat_input.to(device)
                output_ = model(cat_input)
                #print(f'Shap of single output is {output_.size()}')
                output.append(output_)
            output = torch.cat(output,1)
            #output = torch.moveaxis(output,0,-1)
            #print(f'output shape is {output.size()}')
            gts_ = []
            ind_c = 0
            for ind in range(gts.size()[-1]):
                temp = gts[...,ind_c,ind]
                gts_.append(temp)
                ind_c = ind_c+1 if ind_c<2 else 0
            gts = torch.stack(gts_,1)/255.
            #gts = torch.moveaxis(gts,0,-1)
            #print('gts shape is' + str(gts.size()))

            gts = gts.to(device)
            loss = criterion(output, gts) # probably loss per frame/ add all frames loss together

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (ind_batch) % 10 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                       .format(epoch+1, num_epochs, ind_batch+1, total_step, loss.item()))
        save_path = './train/epoch_normalize2/'

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), save_path + str(epoch) +".pth")
        # Decay learning rate
        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

def test():
    # Test the model #############################################################
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'resnet.ckpt')
if __name__ == '__main__':
    path = './data/data/train'
    dataset = Imgdataset(path)
    train_dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    train(train_dataloader)
