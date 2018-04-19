import os
import keras
import librosa
import numpy as np
import matplotlib.pyplot as plt
import ast

# @Class: MelSpectrogram
# @Description: 
#  Class to read .au files and export the songs as MelSpectrograms
class AudioStruct(object):
  def __init__(self, file_path,config):
    # Constants
    self.save_spec_img = ast.literal_eval(config['FILE_READ']['SAVE_SPEC_IMG'])
    self.save_spec_img_dir = config['FILE_READ']['SAVE_SPEC_IMG_DIR']
    self.song_samples = 660000
    self.file_path = file_path
    self.n_fft = 2048
    self.hop_length = 512
    self.genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
      'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
   
  # @Method: getdata
  # @Description:
  #  Retrieve data from .au files and return then as numpy arrays
  def getdata(self):
    # Structure for the array of songs
    song_data = []
    genre_data = []
        
    # Read files from the folders
    for x,_ in self.genres.items():
      for root, subdirs, files in os.walk(self.file_path + x):
        for file in files:
          # Read the audio file
            file_name = self.file_path + x + "/" + file
            print(file_name)
            signal, sr = librosa.load(file_name)
          
            # Calculate the melspectrogram of the audio and use log scale
            melspec = librosa.feature.melspectrogram(signal[:self.song_samples],
              sr = sr, n_fft = self.n_fft, hop_length = self.hop_length).T[:1280,]

            if self.save_spec_img == True:
                plt.pcolormesh(np.transpose(melspec))
                plt.title('{} sampled melspec '.format(file))
                img_out_dir = '{}/{}.png'.format(self.save_spec_img_dir,file)
                print(img_out_dir + " was generated.")
                plt.savefig(img_out_dir)

            # Append the result to the data structure
            song_data.append(melspec)
            genre_data.append(self.genres[x])
    return np.array(song_data), keras.utils.to_categorical(genre_data, len(self.genres))