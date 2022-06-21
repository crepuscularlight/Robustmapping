import torch
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import time

def get_trainer():
    return {"SGPRTrainer":SGPRTrainer}

class SGPRTrainer:
    def __init__(self,cfg,model,optimizer=None,scheduler=None,device=None):
        self.model=model.to(device)
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.device=device

        self.cfg=cfg

    def train_step(self, data):

        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self,data):

        prediction, _, _ = self.model(data)
        gt=data["target"].type(torch.FloatTensor).cuda(self.device)
        print("prediction shape", prediction)
        print("ground truth", gt)

        losses = torch.nn.functional.binary_cross_entropy(prediction, gt)

        return losses

    def evaluate(self,val_loader):
        pred_db = []
        gt_db = []
        losses=0
        for batch in tqdm(val_loader):
            loss_score, pred_b, gt_b = self.eval_step(batch)
            losses += loss_score
            pred_db.extend(pred_b)
            gt_db.extend(gt_b)

        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)

        # F1_score=metrics.f1_score(np.array(gt_db,dtype=int),pred_db)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        F1_max_score = np.max(F1_score)
        print("test",F1_score)
        print("\nModel " +  " F1_max_score: " + str(F1_max_score) + ".")
        model_loss = losses / len(val_loader)
        print("\nModel " + " loss: " + str(model_loss) + ".")

        return model_loss , F1_max_score

    def eval_step(self, data):

        self.model.eval()
        prediction,_,_=self.model(data)
        gt = data["target"].type(torch.FloatTensor)#to(self.device)

        losses = torch.nn.functional.binary_cross_entropy(prediction,gt.cuda(self.device))

        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = data["target"].cpu().detach().numpy().reshape(-1)

        return losses.item(),pred_batch,gt_batch


