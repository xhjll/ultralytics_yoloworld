from ultralytics import YOLO

model = YOLO("/home/chenmiaomiao/work_program/Detection/Detection/ultralytics_yoloworld/yolov8s-worldv2.pt")
model.set_classes(["person", "bus", "truck", "bicycle", "dog"])           # 指定导出类别（以两个类别为例）
model.save("/home/chenmiaomiao/work_program/Detection/Detection/ultralytics_yoloworld/custom_yolov8s-worldv2.pt")        # 保存指定类别的权重文件
