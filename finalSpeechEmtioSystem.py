import streamlit as st
#from audio_recorder_streamlit import audio_recorder
import librosa
import librosa.display
#import IPython.display as ipd
#from IPython.display import display
import numpy as np
import sounddevice as sd
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import base64
#matplotlib inline
#config InlineBackend.figure_format='retina'

 #page settings
st.set_page_config(page_title="SER web-app", page_icon=":speech_balloon:", layout="wide")

from keras.models import load_model
model = load_model('SpeechEmotion.h5')
 
fs = 44100
sd.default.samplerate = fs
sd.default.channels = 2

trStd = np.array(np.loadtxt("trStd.txt"))
trMean = np.array(np.loadtxt("trMean.txt"))

with st.sidebar:
    st.image("choko.jpg")
    

# TITLE and Creator information
st.title('Speech Emotion Recognition System')
st.markdown('Implemented by '
        '[Mayowa Choko](https://www.linkedin.com) - '
        'view project source code on '
        '[GitHub](https://github.com/Oloyedetoby/Speech-Emotion-Detection)')
st.write('\n\n')

#####################################################

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
##add_bg_from_local('emotion1.jpg')
#########################################################

#############################################################
def show_audio(y , sr):
    # create sublots
    fig, axs = plt.subplots(nrows=1,ncols=3, figsize=(30,5))
    
    # load audio file:
    #yy, srr = librosa.load(emotion, sr=16000)
    
    # Show waveform
    librosa.display.waveshow(y=y, sr=sr, ax=axs[0])
    axs[0].set_title('Waveform')
    ''''''
    # Extract fundamental frequency (f0) using a probabilistic approach
    f0, _, _ = librosa.pyin(y=y, sr=sr, fmin=50, fmax=1500, frame_length=2048)

    # Establish timepoint of f0 signal
    timepoints = np.linspace(0, 3.6, num=len(f0), endpoint=False)
    
    # Compute short-time Fourier Transform
    x_stft = np.abs(librosa.stft(y))
    
    # Apply logarithmic dB-scale to spectrogram and set maximum to 0 dB
    x_stft = librosa.amplitude_to_db(x_stft, ref=np.max)
    
    # Plot STFT spectrogram
    librosa.display.specshow(x_stft, sr=sr, x_axis="time", y_axis="log", ax=axs[1])
    
    # Plot fundamental frequency (f0) in spectrogram plot
    axs[1].plot(timepoints, f0, color="cyan", linewidth=4)
    axs[1].set_title('Spectrogram with fundamental frequency')
    
    # Extract 'n_mfcc' numbers of MFCCs components - in this case 30
    x_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Plot MFCCs
    librosa.display.specshow(x_mfccs, sr=sr, x_axis="time", norm=Normalize(vmin=-50, vmax=50), ax=axs[2])
    axs[2].set_title('MFCCs')
    
    # Show metadata in title
    #plt.tight_layout()
    #plt.show()
    
    # Display media player for the selected file
   # display(ipd.Audio(y, rate=sr))
    st.pyplot(fig)
#############################################################


######################################################################
def predict(mfss): 
    mfccs = []
    y, sr = librosa.load(mfss, sr=16000)
    mfccs.append(librosa.feature.mfcc(y=y, sr=sr, fmin=50, n_mfcc=30)) 
    # Create a variable to store the new resized mfccs and apply function for all the extracted mfccs
    resized_mfccs = []

    for mfcc in mfccs:
        resized_mfccs.append(resize_array(mfcc)) 
    X = resized_mfccs.copy()
    X = np.array(X)
    X = (X - trMean)/trStd
    print(X.shape)
    data = X[..., None]  
    print(data.shape)
    prediction = model.predict(data)   
    print(prediction)   
    class_label = np.argmax(prediction)     
    return class_label,prediction, y, sr

class_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

def resize_array(array):
    new_matrix = np.zeros((30,150))   # Initialize the new matrix shape with an array 30X150 of zeros
    for i in range(30):               # Iterate rows
        for j in range(150):          # Iterate columns
            try:                                 # the mfccs of a sample will replace the matrix of zeros, then cutting the array up to 150
                new_matrix[i][j] = array[i][j]
            except IndexError:                   # if mfccs of a sample is shorter than 150, then keep looping to extend lenght to 150 with 0s
                pass
    return new_matrix
################################################################

file = st.file_uploader("Please Upload Wav Audio File Here or Use Demo Of App Below using Preloaded Music",type=["wav","mp3"])


if file is None:  
    st.text("")
else:  
    class_label, prediction, y, sr= predict(file)
    show_audio(y, sr)    
    st.write("## The Emotion Detected is "+class_labels[class_label])
    #st.write("## The Genre of Song is "+class_labels[class_label])