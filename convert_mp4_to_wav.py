# learn from https://github.com/monirome/AphasiaBank/blob/main/convert_mp4_to_wav.py

##############################################
### Define Variables
###############################################

audiopath = "../../../data/aphasia/English/Aphasia/Kempler" 
output_path = "../data/audio_wav/English/Aphasia/Kempler" 

# audiopath = "../../../data/aphasia/English/Control/Kempler" 
# output_path = "../data/audio_wav/English/Control/Kempler" 

################################################
### Library
################################################
import os
import glob

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Convert mp4 format to wav format 
lst = glob.glob(audiopath + "/*.mp4", recursive=True) 
       
for file in lst:
    # Generate the output file path
    output_file = os.path.join(output_path, os.path.basename(file)[:-4] + ".wav")
    # Convert mp4 to wav using ffmpeg
    os.system(f"""ffmpeg -i "{file}" -ar 16000 -ac 1 "{output_file}" """)
    # os.system(f"""ffmpeg -i {file} -ar 16000 -ac 1 {file[:-4]}.wav""") 