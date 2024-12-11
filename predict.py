from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
# model = YOLOWorld("yoloworld_v2_cmm_sim.onnx") 
# model = YOLOWorld("yolov8s-worldv2.onnx") 

# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict("test_onnx_1head/test.jpg", device=3, line_width=3, save=True, save_txt=True, iou=0.4)

# Show results
# results[0].show()