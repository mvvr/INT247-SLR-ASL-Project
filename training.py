import torch
import torch.optim as optim
import cv2
from model import *
from data_processing import *

model = Net()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.NAdam(model.parameters(), lr=0.0005)

n_epochs = 30

min_loss = np.Inf

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    
    model.train()
    count = 0
    for data, target in training_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*data.size(0)
        count += 1
        # print(f'Epoch {epoch}: Batch {count}')
        
    model.eval()
    
    train_loss = train_loss/len(training_loader.sampler)

    valid_loss = 0.0
    class_correct = list(0. for i in range(29))
    class_total = list(0. for i in range(29))

    for data, target in validating_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)

        loss = criterion(output, target)

        valid_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())

        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    valid_loss = valid_loss/len(validating_loader.dataset)
    print('Valid Loss: {:.6f}\n'.format(valid_loss))

    if valid_loss < min_loss:
        torch.save(model.state_dict(), 'ASL Model\\model_ALS.pt')
        print("Improvement. Model Saved")
        min_loss = valid_loss

    for i in range(29):
        if class_total[i] > 0:
            print('Valid Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Valid Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nValid Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
