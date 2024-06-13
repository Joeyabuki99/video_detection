# video_detection
HI! I need help with this system. It consists of a YOLOv8 detector and an EfficientNetB3 classifier for a video detection task. 
The dataset is composed as follows:
- for the detection, some videos of people littering were recorded, they were annotated with a bounding box around plastic bags, cardboard boxes, and all other things that are not one of these two (I have three classes: 'plastic bags', 'cardboard boxes', 'other').
- For the classifier, bounding boxes were extracted from these videos and used as pictures for classification training.

As can be seen, the training of the detection phase gives good results. I cannot say the same for the classifier, which at most reaches 0.40 accuracy. 

But if I try the system, in addition to the classification problem, I also see that the detection task is not done correctly, but I do not understand why. 
