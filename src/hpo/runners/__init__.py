# src/hpo/runners/__init__.py
from .yolov8 import yolov8_train_eval
from .legacy import yolov3_train_eval
from .utils import kfold_iter
