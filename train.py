from ultralytics import YOLO

model = YOLO("yolo11s.pt")

if __name__=='__main__':
    results = model.train(
        data="PersonDefective/data.yaml",   
        epochs=50,                    
        batch=8,                       
        imgsz=640,                     
        device=0,                      
        name='person-defection',       
        pretrained=False,             
        optimizer='SGD',             
        verbose=True,                 
    )

    val_results = model.val()
    print(val_results)
