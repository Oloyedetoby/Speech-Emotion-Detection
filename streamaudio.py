import streamlit as st
from audio_recorder_streamlit import audio_recorder
import librosa
import librosa.display
import numpy as np
import os
import sounddevice as sd
from scipy.io.wavfile import write
import streamlit.components.v1 as components
from io import BytesIO
from pydub import AudioSegment
import io
import wave
import scipy.io.wavfile
import soundfile as sf


fs = 44100
sd.default.samplerate = fs
sd.default.channels = 2

from keras.models import load_model
model = load_model('SpeechEmotion.h5')

trStd = np.array(np.loadtxt("trStd.txt"))
trMean = np.array(np.loadtxt("trMean.txt"))

print(trStd.shape, trMean.shape)

# TITLE and Creator information
st.title('Speech Emotion Recognition System')
st.markdown('Implemented by '
        '[OLF TObi](https://www.linkedin.com/in/stefanrmmr/) - '
        'view project source code on '
        '[GitHub](https://github.com/stefanrmmr/streaml)')
st.write('\n\n')


def predict(mfss): 
    mfccs = []
    y, sr = librosa.load(file, sr=16000)
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
    return class_label,prediction

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

file = st.sidebar.file_uploader("Please Upload Wav Audio File Here or Use Demo Of App Below using Preloaded Music",type=["wav"])


def audiorec_demo_app():

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Custom REACT-based component for recording client audio in browser
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    # specify directory and initialize st_audiorec object functionality
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    # STREAMLIT AUDIO RECORDER Instance
    val = st_audiorec()
    # web component returns arraybuffer from WAV-blob
    st.write('Audio data received in the Python backend will appear below this message ...')

    if isinstance(val, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            val = np.array(val)             # convert to np array
            sorted_ints = val[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()
            output = convert_bytearray_to_wav_ndarray(input_bytearray = wav_bytes)
            scipy.io.wavfile.write("output1.wav", 16000, output)
    predict('output1.wav')


def convert_bytearray_to_wav_ndarray(input_bytearray: bytes, sampling_rate=16000):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sampling_rate, np.frombuffer(input_bytearray, dtype=np.int16))
    output_wav = byte_io.read()
    output, samplerate = sf.read(io.BytesIO(output_wav))
    return output



if __name__ == '__main__':
    # call main function
    audiorec_demo_app()


if file is None:  
  st.text("Please upload an mp3 file")
else:   
  class_label,prediction = predict(file)   
  st.write("## The Genre of Song is "+class_labels[class_label])
  #st.write("## The Genre of Song is "+class_labels[class_label])