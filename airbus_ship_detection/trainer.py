import random
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from airbus_ship_detection import configs
from airbus_ship_detection.metrics import Metrics

def get_torch_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# main train routine
# Implementation from  https://github.com/ternaus/robot-surgery-segmentation
def train(lr, model, model_name, criterion, train_loader, valid_loader, init_optimizer, train_batch_sz=16, valid_batch_sz=4, n_epochs=1, fold=1, verbose=True):

    model_path = configs.DIR_MODELS / f"{model_name}_{fold}.pt"
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 50
    log_path = configs.DIR_LOGS / f"{model_name}_{fold}.log"
    log = open(log_path,'at', encoding='utf8')

    device = get_torch_device()
    if verbose:
        print(f"Using device {device}")
    model = model.to(device)
    optimizer = init_optimizer(lr)

    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) *  train_batch_sz)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        valid_metrics = Metrics(batch_size=valid_batch_sz)  # for validation
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            
            # Validation
            comb_loss_metrics = validation(model, criterion, valid_loader, valid_metrics)
            write_event(log, step, **comb_loss_metrics)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
        
def validation(model: nn.Module, criterion, valid_loader, metrics):
    print("Validation")
    
    losses = []
    device = get_torch_device()
    model.eval()
    
    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        metrics.collect(outputs.detach().cpu(), targets.detach().cpu()) # get metrics 
    
    valid_loss = np.mean(losses)  # float
    valid_iou, valid_dice, valid_jaccard = metrics.get() # float

    print('Valid loss: {:.5f}, IoU: {:.5f}, Jaccard: {:.5f}, Dice: {:.5f}'.format(valid_loss, valid_iou, valid_jaccard, valid_dice))
    comb_loss_metrics = {'valid_loss': valid_loss, 'iou': valid_iou.item(), 'jaccard': valid_jaccard.item(), 'dice': valid_dice.item()}

    return comb_loss_metrics

def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()