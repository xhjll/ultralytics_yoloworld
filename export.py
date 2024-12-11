from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-worldv2.pt')  # load an official model
# model = YOLO('runs/obb/train8/weights/last.pt')  # load a custom trained model
# model = YOLO('runs/obb/train22/weights/best.pt') 
# model = YOLO('runs/export/best_update_fisheye_person.pt') 
# Export the model
model.export(format='onnx',simplify=True, opset=12,imgsz = [640, 640])