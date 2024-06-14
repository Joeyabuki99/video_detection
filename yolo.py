from IPython.display import display, Image
from ultralytics import YOLO

ROOT_DIR_YOLO = "/path/dataset_other/"
model_path = "/path/yolov8s.pt"

model = YOLO(model_path)

results_m = model.train(data= ROOT_DIR_YOLO + "data.yaml", single_cls=False, imgsz=640, epochs = 25)    
