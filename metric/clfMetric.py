# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np 

# gt\pred      P     N 
# ---------------------
# P         |  TP    FN
# N         |  FP    TN 
 
class ClfMetric(object):
    """
    Metric for classification
    """
    def __init__(self, k, dtype=np.int32):
        self.k = k  # number of class
        self.dtype = dtype
        self.reset()
    
    def reset(self):
        # confusion matrix
        self.cm = np.zeros((self.k,)*2).astype(self.dtype)

    def get_accuracy(self):
        """
        get current pixel accuracy
        """
        # acc = (TP + TN) / (TP + TN + FP + TN)
        pixelAcc = np.diag(self.cm).sum() / self.cm.sum()
        return pixelAcc

    def get_precision(self):
        """
        get current accuracy for each class
        """
        # precision = TP / (TP + FP)
        precision = np.diag(self.cm) / self.cm.sum(axis=1)
        return precision

    def get_meanPrecision(self):
        """
        get mean precision  
        """
        # mean class acc = 1/k * sum(class acc), usually 1/(k+1) here use 1/k
        precision = self.get_precision()
        meanPrecision = np.nanmean(precision)
        return meanPrecision 
    
    def get_recall(self):
        """
        get current recall
        """
        recall = np.diag(self.cm) / self.cm.sum(axis=0)
        return recall

    def get_F1score(self, e=1e-8):
        """
        get current f1 score 
        """
        precision = self.get_precision()
        recall = self.get_recall()
        f1 = 2 * precision * recall / (precision + recall + e)
        return f1

    def _get_cm(self, pred, label):
        """
        get current confusion matrix
        """
        assert pred.shape == label.shape, f"Predicted label {pred.shape} \
            and True label shape {label.shape} do not match."
        pred = pred.astype(self.dtype)
        label = label.astype(self.dtype)
        mask = (label >= 0) & (label < self.k)
        finLabel = self.k * label[mask] + pred[mask]
        count = np.bincount(finLabel, minlength=self.k**2)
        cm = count.reshape(self.k, self.k)
        return cm

    def update(self, pred, label):
        """
        add new batches and update the current confusion matrix
        """
        self.cm += self._get_cm(pred, label)


# if __name__ == '__main__':
#     k = 3
#     batch_size = 32 
#     np.random.seed(4)
#     pred = np.random.randint(0,k,(batch_size, 1))
#     label = np.random.randint(0,k,(batch_size, 1))

#     metric = ClfMetric(k)
#     metric.update(pred, label)
#     acc = metric.get_accuracy()
#     f1 = metric.get_F1score()
#     from sklearn.metrics import accuracy_score, f1_score
#     print(accuracy_score(label, pred) == acc)
#     print(f1_score(label, pred, average=None))
#     print(f1)