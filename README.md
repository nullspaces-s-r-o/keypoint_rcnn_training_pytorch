# How to Train a Custom Keypoint Detection Model with PyTorch

### Detailed explanation
https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da

## Known issuse

###  issue #1, operands could not be broadcast together with shapes (2,) (17,)
[Reference](https://detectron2.readthedocs.io/en/stable/modules/evaluation.html)

.torch/lib/python3.8/site-packages/pycocotools/cocoeval.py, function setKpParams() sets 

```
self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
```

which is 17 in lenght, compared to 2 keypoionts in this tutorial

solution:
```
class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            if iou_type == 'keypoints':
                self.coco_eval[iou_type].params.kpt_oks_sigmas = np.array(2 * [0.5])/10.0

```
