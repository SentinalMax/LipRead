import imageio
import numpy as np
import PIL.Image as Image
from PIL import ImageFilter
from PIL import ImageEnhance
import pytesseract
from pytesseract import Output
import spacy
from spacy.matcher import Matcher
import cv2
from tqdm import tqdm
import time 

#python3 -m pip install pytesseract
#python3 -m pip install imageio
#python3 -m pip install imageio[ffmpeg]
#python3 -m pip install spacy
#python3 -m pip install imageio[pyav]
#python3 -m pip install opencv-python
#python3 -m pip install tesseract
#python3 -m pip install tesseract-ocr
#python3 -m spacy download en_core_web_sm

#Variables
threshold1 = 100
threshold2 = 200

# Step 1 : Read the Video
reader = imageio.get_reader('video2.mp4')
fps = reader.get_meta_data()['fps']

print("Step #1 [read the video] COMPLETE")

# Step 2 : Extract Frames from Video as Images
frame_list = list()
progress_bar = tqdm(total=len(frame_list), unit='I') #create progress bar
for I, frame in enumerate(reader):
    
    # Convert the frame to an RGB-image
    frame = Image.fromarray(frame).convert('RGB')
    frame_list.append(frame)
    progress_bar.update(1)

print("\nStep #2 [Extract Frames from Video as Images] COMPLETE")

# Step 3 : Apply filters to highlight lips
progress_bar = tqdm(total=len(frame_list), unit='frame')
for frame in frame_list:
    
    progress_bar.update(1)
    frame_filtered = frame.filter(ImageFilter.EDGE_ENHANCE_MORE)
    frame_sharpened = ImageEnhance.Sharpness(frame_filtered).enhance(2)
    frame_contrast = ImageEnhance.Contrast(frame_sharpened).enhance(3)
    frame_filtered = frame_contrast

print("\nStep #3 [Apply filters to highlight lips] COMPLETE")
# Step 4 : Apply Edge Detection to detect mouth contours
for frame in frame_list:
    
    
    frame_filtered = np.asarray(frame_filtered)
    frame_gray = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
    #edged = frame_gray.filter(ImageFilter.FIND_EDGES)
    edged = cv2.Canny(frame_gray, threshold1, threshold2)
    # Ensure the 'edged' image is in the correct format
    edged = np.uint8(edged)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    progress_bar.update(1)
    
print("\nStep #4 [Apply Edge Detection to detect mouth contours] COMPLETE")
# Step 5 : Use OCR to read words from lips
for frame in frame_list:
    
    d = pytesseract.image_to_data(frame, lang='eng', output_type=Output.DICT)
    text = d['text'][0]
    speech = ' '.join([w for w in text.split() if w])
    progress_bar.update(1)
    print(speech + "-t")

print("\nStep #5 [Use OCR to read words from lips] COMPLETE")
# Step 6 : Use NLP to understand the meaning of the words


nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
pat_1 = [{'LOWER': 'hello'}, {'IS_PUNCT': True}, {'LOWER': 'world'}]
matcher.add('hello_world', [pat_1])

doc = nlp(speech)
matches = matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print(span.text)