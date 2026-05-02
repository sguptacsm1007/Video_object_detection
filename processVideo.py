import cv2
import streamlit as st
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from parameters import get_model_path
#load model from weights locally

def load_model():
    model_path=get_model_path()
    image_processor = RTDetrImageProcessor.from_pretrained(model_path)
    model = RTDetrForObjectDetection.from_pretrained(model_path)
    return model,image_processor




#process the video and return the results
# input parameters video_path,model, streamlitoutputframe  
def process_video(video,model,image_processor,streamlitoutputframe,device):
    cap=cv2.VideoCapture(video)
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        inputs=image_processor(images=image,return_tensors="pt").to(device)
         
        #predict from model 
        with torch.no_grad():
            outputs=model(**inputs)
        
        # draw bounding box and label in image frame
        #store target size
    
        target_sizes=torch.tensor([[height, width]],device=device)
        
             
        # draw bounding box and label in image frame
        results=image_processor.post_process_object_detection(outputs,target_sizes=target_sizes,threshold=0.3)[0]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy().astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_name = model.config.id2label[label.item()]
            cv2.putText(frame, label_name, (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- KEY FOR INSTANT FEEDBACK ---
        # Convert BGR (OpenCV) to RGB for Streamlit and update the SAME placeholder
        streamlitoutputframe.image(frame, channels="BGR")
    cap.release()
    return


