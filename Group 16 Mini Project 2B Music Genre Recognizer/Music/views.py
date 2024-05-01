from django.shortcuts import render
import librosa
import numpy as np
from django.http import JsonResponse
import json
import tensorflow
from .models import CNNModel

# Create your views here.
def home(request):
    return render(request, 'Music/index.html')

def main(request):
    return render(request, 'Music/upload.html')

def about(request):
    return render(request, 'Music/about.html')


import numpy as np
import librosa
from keras.models import load_model
import math
import os
import tempfile

mapping= [
        "reggae",
        "rock",
        "metal",
        "jazz",
        "hiphop",
        "disco",
        "blues",
        "classical",
        "pop",
        "country"
    ]

# Load the pre-trained CNN model
model = CNNModel.load_model()

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
# Inside your preprocess_audio function
def preprocess_audio(file_path,num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # # # Load the audio file using librosa
    file_path = os.path.join(file_path)

    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

  # ... (rest of your function to process the signal using librosa)


    for d in range(num_segments):

      # calculate start and finish sample for current segment
      start = samples_per_segment * d
      finish = start + samples_per_segment

      # extract mfcc
      mfcc = librosa.feature.mfcc(y=signal[start:finish],sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
      mfcc = mfcc.T

      # store only mfcc feature with expected number of vectors
      if len(mfcc) == num_mfcc_vectors_per_segment:
          data["mfcc"].append(mfcc.tolist())

    # save MFCCs to json file
    return np.array(data['mfcc'])



def upload_audio(request):
    if request.method == 'POST':
        audio_data = request.FILES['audio_file']
        # Save the uploaded audio file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
            for chunk in audio_data.chunks():
                temp_audio_file.write(chunk)
            temp_audio_file_path = temp_audio_file.name

        # Preprocess the audio file
        x = preprocess_audio(temp_audio_file_path)
        x = x[0][..., np.newaxis]
        x = x[np.newaxis, ...]
        # Predict the genre
        # perform prediction
        prediction = model.predict(x)
        predicted_index = np.argmax(prediction)
        predicted_genre=mapping[predicted_index]
       
        return render(request, 'Music/upload.html', {'genre': predicted_genre})

    return render(request, 'upload.html')