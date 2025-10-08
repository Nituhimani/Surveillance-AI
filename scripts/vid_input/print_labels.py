#to check the labels of a model
#python print_labels.py path/to/your/best.pt

from ultralytics import YOLO
import sys
m = YOLO(sys.argv[1])
print("Model labels:", m.names)
