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

#How to label
install labelme using 
```pip install labelme```

run labelme using  
```
labelme.exe --nodata --autosave
```

složku s fotkama olelujeme (pro každý objekt vytvoříme "rectangle" a dva body)
na složku pustíme skript "labelme_2_COCO.py" -> to nám vytvoří json soubory pro COCO formát.
Je potřeba manutálně přetahat vytvořené JSONY a fotky do takovéto datové strukury:
<dataset_name>/train/annotations - sem dát trénovací JSON soubory
<dataset_name>/train/images - sem dát trénovací snímky
<dataset_name>/test/annotations - sem dít testovací JSON soubory
<dataset_name>/test/images - sem dít testovací snímky




## Převod ipynb do *.py skriptu
```
pip install nbconvert 
jupyter nbconvert --to script KeypointRCNN_training.ipynb
```
-> vznikne `KeypointRCNN_training.py`

# Potential improvements
- počet anchors a počet aspect_ratios můžeme snížit protože známe vzdálenost blady od kamery

# Dataset
```
mkdir -p $HOME/keypoints
sshfs mendel@192.168.0.249:/home/mendel/sd/DetectionData/Dataset/keypoints $HOME/keypoints
```