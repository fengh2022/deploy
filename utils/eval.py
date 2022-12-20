import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from time import time

def evaluation(model, dataloader, device):
    labels = []
    preds = []
    model.eval()
    time_total = 0.0
    cnt = 0
    with torch.no_grad():
        for imgs, labs in tqdm(dataloader):
            imgs = imgs.to(device).float()
            labs = labs.tolist()

            torch.cuda.synchronize()
            tap = time()
            logits = model(imgs)
            torch.cuda.synchronize()
            time_total += time()-tap

            logits = logits.argmax(dim=1)

            labels += labs
            preds += logits.cpu().numpy().tolist()
            cnt += 1

    print(f'fps: {cnt/time_total}')

    return accuracy_score(labels, preds)



