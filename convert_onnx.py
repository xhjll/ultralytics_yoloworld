from ultralytics import YOLO

model = YOLO("/home/chenmiaomiao/work_program/Detection/Detection/ultralytics_yoloworld/custom_yolov8s-worldv2.pt")   # 加载第一步生成的指定类别的权重文件（生成onnx）