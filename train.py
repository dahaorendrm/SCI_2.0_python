from imgdataset import Imgdataset
import torch
import torch.nn as nn
from networks.chasti_network import CHASTINET
import scipy.io as scio
from torch.utils.data import DataLoader
import numpy as np
num_epochs = 100
batch_size = 4
learning_rate = 0.003

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
model = CHASTINET().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
def train():
    MASK = scio.loadmat('./train/lesti_mask.mat')['mask']
    total_step = len(train_dataloader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for ind_batch, (gts, inputs) in enumerate(train_dataloader): # batch,weight,height,channel
            mea = inputs[...,0] # mea normalize???????????????????????????????????????????? from birnet
            img_ns = inputs[...,1:]
            masks = torch.tensor(MASK[...,:img_ns.size()[-1]]).double()
            masks = masks.repeat(img_ns.size()[0],1,1,1)
            img_n_codes = img_ns*masks
            output = []
            for ind in range(img_ns.size()[-1]):
                #gt = gts[...,ind]
                mea = torch.unsqueeze(mea,3)
                img_n = img_ns[...,ind:ind+1]
                img_n_code_begin = img_n_codes[...,:ind]
                img_n_code_end = img_n_codes[...,ind+1:]
                mask = masks[...,ind:ind+1]
                print(f'Shape of img_n: {img_n.size()}, img_n_code_begin: {img_n_code_begin.size()}, /nimg_n_code_end: {img_n_code_end.size()}, mask: {mask.size()}, mea: {mea.size()}')
                cat_input = torch.cat((img_n,mea,mask,img_n_code_begin,img_n_code_end),dim=3)
                # Forward pass
                print(f'Shap of cat_input is {cat_input.size()}')
                cat_input = np.array(cat_input)
                cat_input = np.moveaxis(cat_input,-1,1)
                cat_input = torch.from_numpy(cat_input).double()
                #cat_input = torch.movedim(cat_input,-1,1)
                cat_input = cat_input.to(device)
                output.append(model(cat_input))
            output = torch.Tensor(output)
            output = torch.moveaxis(output,0,-1)

            gts_ = []
            ind_c = 0
            for ind in range(gts.size()[-1]):
                temp = gts[...,ind_c,ind]
                gts_.append(temp)
                ind_c = ind_c+1 if ind_c<2 else 0
            gts = torch.Tensor(gts_)
            gts = torch.moveaxis(gts,0,-1)
            print('gts shape is' + str(gts.size()))

            gts = gts.to(device)
            loss = criterion(output, gts) # probably loss per frame/ add all frames loss together

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

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
    train()
