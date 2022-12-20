import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CatDogDataset
from utils.eval import evaluation
from utils.params import arg

from model import get_model

import os


traindata = CatDogDataset(rootdir=arg.root_dir, is_training=True)
evaldata = CatDogDataset(rootdir=arg.root_dir, is_training=False)

trainloader = DataLoader(traindata, batch_size=arg.batch_size, num_workers=arg.num_workers, shuffle=True)
evalloader = DataLoader(evaldata, batch_size=arg.batch_size, num_workers=arg.num_workers, shuffle=False)

device = torch.device(arg.device)

model = get_model(model_name=arg.model_name, num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)

epochs = 1 if arg.debug else arg.epochs
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    for step, (imgs, labs) in enumerate(trainloader):
        imgs = imgs.to(device).float()
        labs = labs.to(device)

        logits = model(imgs)
        losses = criterion(logits, labs)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if step % arg.verbose_step==0:
            print(f'epoch: {epoch}, step: {step}, loss: {losses.item()}')

    acc = evaluation(model, evalloader, device)
    print(f'epoch: {epoch}--acc: {acc}')

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(arg.export_dir, 'model.pth'))
        print(f'best accuracy: {best_acc} update checkpoint...')






