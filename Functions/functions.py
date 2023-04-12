import os
import pandas as pd
import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn
import seaborn as sns
import IPython.display as ipd

sr = 48000
window_length_samples = 2560
hop_length_samples = 694
n_mels = 128
stft_window_seconds = window_length_samples/sr
stft_hop_seconds = hop_length_samples/sr
fft_length =2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

log_mel_params = {
    "fs" : sr,
    "stft_window_seconds" : stft_window_seconds,
    "stft_hop_seconds" : stft_hop_seconds,
    "n_mels" :  n_mels,
    "window_length_samples" :  window_length_samples,
    "hop_length_samples" :  hop_length_samples,
    "fft_length" : fft_length
}

duration = 10

def get_params():

    sr = 48000
    window_length_samples = 2560
    hop_length_samples = 694
    n_mels = 128
    stft_window_seconds = window_length_samples/sr
    stft_hop_seconds = hop_length_samples/sr
    fft_length =2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    log_mel_params = {
        "fs" : sr,
        "stft_window_seconds" : stft_window_seconds,
        "stft_hop_seconds" : stft_hop_seconds,
        "n_mels" :  n_mels,
        "window_length_samples" :  window_length_samples,
        "hop_length_samples" :  hop_length_samples,
        "fft_length" : fft_length
    }

    return log_mel_params

# Generate 10 second snippets of the audio files
def generate_snippets(audio_file, sr, duration):
    y, sr = lib.load(audio_file, sr=sr)
    n_samples = len(y)
    n_samples_per_snippet = int(sr * duration)
    n_snippets = int(np.ceil(n_samples / n_samples_per_snippet))
    snippets = []
    for i in range(n_snippets):
        start = i * n_samples_per_snippet
        end = start + n_samples_per_snippet
        snippet = y[start:end]
        # When decided whats the preprocessing for the best model here will be added that preprocessing
        snippets.append(snippet)
    return snippets

def generate_mel_spec(audio_signal, sr, dict_params):
    '''
    This function generates the mel spectrogram of an audio signal with sample rate sr and using the parameters indicated by the dict_params dictionary.
    '''

    D = np.abs(lib.stft(audio_signal,
                            n_fft=dict_params["fft_length"],
                            hop_length= dict_params["hop_length_samples"],
                            win_length = dict_params["window_length_samples"],
                            window='hann',
                            pad_mode = 'constant'
                           ))**2
    S = lib.feature.melspectrogram(S=D, sr=sr,
                                    win_length = dict_params["window_length_samples"], #nsamples = sr [samples/second] * time [seconds]
                                    n_fft= dict_params["fft_length"], 
                                    hop_length=dict_params["hop_length_samples"], 
                                    n_mels=dict_params["n_mels"], 
                                    fmax = sr/2,
                                    window='hann'
                                      )
    log_S = lib.power_to_db(S, ref=np.max)
    return log_S

def analyze_audios(audios):

  analyzed_data = []

  model = tf.keras.models.load_model('./pfc-serena-demo/Model/model.h5')

  for audio in audios:
      audio_file = "./pfc-serena-demo/Audios/"+audio
      snippets = generate_snippets(audio_file, sr, duration)
      for s in range(len(snippets)):
        snippet = snippets[s]
        log_mel = generate_mel_spec(snippet, sr, log_mel_params)
        pred = model.predict(np.array([log_mel]), verbose = 0)[0]
        
        analyzed_data.append((audio, s*10,  (s*10+10), snippet, log_mel, pred[0], pred[1]))

  analyzed_data = pd.DataFrame(analyzed_data, columns=["Nombre del audio","Comienzo (s)", "Fin (s)",'Audio','log-mel', 'Score Vehiculos Acuaticos y Terrestres', 'Score Vehiculos Aereos'])
  return analyzed_data

def plot_audio_labels(audio_name, dataframe, inicio, fin):
    information = dataframe[dataframe["Nombre del audio"]==audio_name]
    information = dataframe[dataframe["Comienzo (s)"]>=inicio]
    information = dataframe[dataframe["Fin (s)"]<=fin]

    # sort by begin
    information = information.sort_values(by=['Comienzo (s)'], ascending = True)

    pred0 = []
    pred1 = []

    x_axis = []
    for i, row in information.iterrows():

        pred0.append(row['Score Vehiculos Acuaticos y Terrestres'])
        pred0.append(row['Score Vehiculos Acuaticos y Terrestres'])
        pred1.append(row['Score Vehiculos Aereos'])
        pred1.append(row['Score Vehiculos Aereos'])

        x_axis.append(row["Comienzo (s)"])
        x_axis.append(row["Fin (s)"]+0.001)

    plt.figure(figsize=(20, 5), dpi=300)
    plt.title("Evolucion de los Scores para el audio: " + audio_name)
    plt.plot(np.array(x_axis), pred0, color='green', linestyle ='dashdot', label = "VEHICULOS ACUATICOS Y TERRESTRES Score")
    plt.plot(np.array(x_axis), pred1, color='orange', linestyle ='dashdot', label = "VEHICULOS AEREOS Score")

    for i in np.arange(0,310,step=10):
            plt.axvline(x=i, color='silver', linestyle=':', linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.ylim(-0.01, 1.01)
    plt.xlim(inicio,fin)
    plt.legend()
    plt.show()

    # Display a widget to listen to the audio
    return information

def check_information(information, inicio):
    '''
    
    Given a start time, this function return the score in that chunk of time and make posible to listen to the audio.

    '''

    # Modify inicio to be a multiple of 10
    inicio = inicio - inicio%10

    # Get the information of the audio
    information = information[information["Comienzo (s)"]==inicio]


    # Get the audio signal
    audio_signal = information["Audio"].values[0]

    # Print scores and listen to the audio
    print("Score Vehiculos Acuaticos y Terrestres: ", information["Score Vehiculos Acuaticos y Terrestres"].values[0])
    print("Score Vehiculos Aereos: ", information["Score Vehiculos Aereos"].values[0])

    return audio_signal

