# AutoSmartLock

## Set up
```
git clone https://github.com/AutoSmartLock/AutoSmartLock.git
cd AutoSmartLock/
pip install -r requirements.txt
```

## Load dataset
...

## Run
```
Run inference on images, videos, directories, streams, etc.

Usage - weight:
    $ python path/to/detect.py --weights        yolov5s.pt                      # basic
                                                training-result/weights/best.pt # AutoSmartLock custom model
                                        	path/*.pt                       # your custom model

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - model:
    $ python path/to/detect.py --model   mymodel.pkl                # AutoSmartLock custom model 
                                         path/*.pkl                 # your custom model 
```
