from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
from torch.utils.data import RandomSampler, DataLoader, random_split
from torchvision import datasets, transforms
from model import *
import matplotlib.pyplot as plt

model = Net()
model.cuda()
model.load_state_dict(torch.load('ASL Model\\final_model_ALS.pt'))
model.eval()

train_dir = 'ASL Model\\Data\\asl_alphabet_train\\asl_alphabet_train'
data_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])
seed = torch.random.initial_seed()
train_data = datasets.ImageFolder(train_dir, transform=data_transform)
train_data, valid_data = torch.utils.data.random_split(train_data, [79832, 7168], generator=torch.Generator().manual_seed(seed))

num_workers = 0
batch_size = 128

validating_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

y_pred = []
y_true = []

# iterate over test data
for data, target in validating_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        target = target.data.cpu().numpy()
        y_true.extend(target) # Save Truth

# constant for classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Build confusion matrix

cf_matrix = confusion_matrix(y_true, y_pred)

# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                    #  columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(cf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix of Validation Set")
plt.savefig('output.png')
