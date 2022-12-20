import torch
from torch.utils.data import DataLoader
from time import time
from utils.params import arg
import os
from dataset import CatDogDataset
from model import get_model
from utils.eval import evaluation

device = torch.device(arg.device)
torch_model = os.path.join(arg.export_dir, 'model.pth')
model = get_model(model_name=arg.model_name, num_classes=2)
model = model.to(device)
model.load_state_dict(torch.load(torch_model))


evaldata = CatDogDataset(rootdir=arg.root_dir, is_training=False)
evalloader = DataLoader(evaldata, batch_size=1, num_workers=arg.num_workers, shuffle=True)

y_true = []
y_pred = []

acc = evaluation(model, evalloader, device)

print(f'acc: {acc}')



