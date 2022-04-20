import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import *
import matplotlib.pyplot as plt

model = Net()
model.cuda()
model.load_state_dict(torch.load('ASL Model\\final_model_ALS.pt'))
model.eval()

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

batch_size=29
test_dir = 'ASL Model\\Data\\asl_alphabet_test\\asl_alphabet_test'
data_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
test_data = datasets.ImageFolder(test_dir, transform=data_transform)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()
images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.cpu().numpy())

imgs = images.cpu()
imgs.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
plt.axis('off')
plt.title('Visualization of Test Data Classification', pad=30)
for idx in np.arange(29):
    ax = fig.add_subplot(4, 8, idx+1, xticks=[], yticks=[])
    image_png = np.transpose(imgs[idx], (1, 2, 0))
    plt.imshow(image_png)
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
plt.show()
