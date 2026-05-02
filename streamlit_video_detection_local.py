from processVideo import load_model,process_video

import streamlit as st
import torch

model,image_processor=load_model()
#device checking strategy mps->CUDA->cpu
device=torch.device( "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)

video_file=st.file_uploader("Upload Video",type=["mp4"])
temp_video_file_name="video_file.mp4"
if video_file is not None:
    with open(temp_video_file_name,"wb") as f:
        f.write(video_file.read())
    streamlitframe=st.empty()
    process_video(temp_video_file_name,model=model,image_processor=image_processor,streamlitoutputframe=streamlitframe,device=device)
    

