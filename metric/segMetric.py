# Copyright (c) 2020. All Rights Reserved.
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
 
class SegMetric(object):
    """
    Metric for segmentation
    """
    def __init__(self, k, dtype=np.int32):
        self.k = k  # number of class
        self.dtype = dtype
        self.reset()
    
    def reset(self):
        # confusion matrix
        self.cm = np.zeros((self.k,)*2).astype(self.dtype)

    def get_pixelAccuracy(self):
        """
        get current pixel accuracy
        """
        # acc = (TP + TN) / (TP + TN + FP + TN)
        pixelAcc = np.diag(self.cm).sum() / self.cm.sum()
        return pixelAcc

    def get_precision(self):
        """
        get current pixel accuracy for each class
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

    def get_IoU(self):
        """
        get intersection over union 
        """
        intersection = np.diag(self.cm)
        union = self.cm.sum(axis=1) + self.cm.sum(axis=0) - np.diag(self.cm)
        # IoU = TP / (TP + FP + FN)
        IoU = intersection / union
        return IoU

    def get_meanIoU(self):
        """
        get mean intersection over union 
        """
        IoU = self.get_IoU()
        mIou = np.nanmean(IoU)
        return mIou

    def get_FWIoU(self):
        """
        get frequency weighted Intersection over union 
        """
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] * [TP/(TP+FP+FN)]
        IoU = self.get_IoU()
        freq = self.cm.sum(axis=1) / np.sum(self.cm)
        FWIoU = (freq[freq > 0] * IoU[freq > 0]).sum()
        return FWIoU

    def _get_cm(self, imgPred, imgTrue):
        """
        get current confusion matrix
        """
        assert imgPred.shape == imgTrue.shape, f"Predicted image shape {imgPred.shape} \
            and True image shape {imgTrue.shape} do not match."
        imgPred = imgPred.astype(self.dtype)
        imgTrue = imgTrue.astype(self.dtype)
        mask = (imgTrue >= 0) & (imgTrue < self.k)
        finLabel = self.k * imgTrue[mask] + imgPred[mask]
        count = np.bincount(finLabel, minlength=self.k**2)
        cm = count.reshape(self.k, self.k)
        return cm

    def update(self, imgPred, imgTrue):
        """
        add new batches and update the current confusion matrix
        """
        self.cm += self._get_cm(imgPred, imgTrue)


# if __name__ == '__main__':
#     k = 3
#     batch_size = 4 
#     np.random.seed(4)
#     imgPred = np.random.randint(0,k,(batch_size, 3, 3))
#     imgTrue = np.random.randint(0,k,(batch_size, 3, 3))

#     metric = SegMetric(k)
#     metric.update(imgPred, imgTrue)
#     acc = metric.get_pixelAccuracy()
#     mIoU = metric.get_meanIoU()
#     print(acc)
#     print(mIoU)