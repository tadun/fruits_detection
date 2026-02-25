from ultralytics import YOLO
import os, sys
print("Python:", sys.executable)
print("CWD:", os.getcwd())

model = YOLO("yolov8n.pt")  ## switch the models here can find the appr names to call on the webpage (dont mistake the name or else it will use a random model)

## dont chnage the parameters
model.train(
    data="./fruits/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    workers=4, 
    verbose=True
)
