def train_detector(data, project, name="yolo26n_1280", model="yolo26n.pt", epochs=30, imgsz=1280, batch=8):
    from ultralytics import YOLO

    return YOLO(model).train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, seed=0, workers=2, project=str(project), name=name)
