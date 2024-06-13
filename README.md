# video_detection
HI!! I need help with this system. It is composed by a YOLOv8 detector and by an EfficientNetB3 classifier for a video detection task. 
The dataset is composed as follow:
- for the detection have been recorded some video of people dumping garbage, have been annotated with bounding box around plastic_bags, cardboard_box, and all the other things that are not one of these two (i have three classes: 'plastic_bag', 'cardboard_box', 'other')
- for the classifier, from these video have been extracted the bounding boxes and used as images for the training of the classification.

As you can see the training of the detection phase gives me some good results. I cannot say the same for the classifier, which reach the 0.40 accuracy at his best. 

But if I try the system, other than the classsification problem, I also see that the detection task is not done correctly, but I don't understand why. 
