import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
import keyboard 

class wav_homework:
 '''
    프레임 저하 문제로 전체로 돌리는 것에 대해서는 받는 소리의 1초에 한번씩만 처리 했습니다.
    2개로 나누어서 돌렸습니다. 
    (105)wav_homework(data_np,fs).show_result_first() : original signal , fft, filter back, rms bar
    (106)wav_homework(data_np,fs).show_result_second(): 6 channel plotting
    q 입력시 중지 
 '''
 def __init__(self,array,fs):
   self.array=array
   self.fs=fs

 def FFT(self): # FFT function
   dc_removal_signal=self.array-np.mean(self.array)
   NFFT=self.array.size
   k=np.arange(NFFT)
   f0=k*self.fs/NFFT
   f0=f0[range(math.trunc(NFFT/2))]
   y=np.fft.fft(dc_removal_signal)/NFFT
   y=y[range(math.trunc(NFFT/2))]
   amplitude_hz=2*abs(y)
   return f0,amplitude_hz

 def bandpass_filter(self, lowcut, highcut,data): 
   nyq= 0.5 * self.fs
   b,a = signal.butter(3,[lowcut/nyq,highcut/nyq],btype='band')
   w,h=signal.freqz(b,a,self.fs)
   w_hz=w*self.fs/(2*math.pi)
   after_data=signal.filtfilt(b,a,data, axis=0)
   return w_hz,abs(h),after_data

 def filter_bank(self):
    w_hz_list,filter_bank_list=[],[]
    aff=[]
    number=[200, 373, 629, 1006, 1561, 2381,3590]; d_number=[self.array]
    for j in range (len(d_number)):
      for i in range(0,len(number)-1):
        w_hz,fr_w,after_data=wav_homework.bandpass_filter(self,number[i],number[i+1],d_number[j]) # bandpassfilter
        w_hz_list = np.concatenate([w_hz_list, w_hz]) # filter bank x
        filter_bank_list=np.concatenate([filter_bank_list,fr_w]) # filter bank y
        aff.append(after_data)
    return w_hz_list, filter_bank_list, aff
 
 def make_envelope(self,data):
   da=data**2
   rms_data= math.sqrt(np.mean(da))
   return rms_data

 def show_result_first(self):
   fft_freq,data_fft=wav_homework.FFT(self)
   w_hz,fr_h,after_ndata=wav_homework.filter_bank(self) # filter bank
   rms_bar=[];x=[1,2,3,4,5,6]
   for i in after_ndata:
     rms_bar.append(wav_homework.make_envelope(self,i))
   plt.subplot(4,1,1);plt.plot(self.array,color="black")
   plt.subplot(4,1,2);
   marker, stemlines, baseline =plt.stem(fft_freq,data_fft);plt.ylim([0,30])
   stemlines.set_linewidth(2);stemlines.set_edgecolor("black") ## 라인 색깔
   marker.set_visible(False);baseline.set_visible(False) 
   plt.subplot(4,1,3);plt.plot(w_hz,fr_h,color="red");plt.xlim([0,4000])
   plt.subplot(4,1,4);plt.bar(x,rms_bar,color="blue")
   plt.tight_layout();plt.pause(0.001)
   plt.show();plt.clf();
   
   
 def show_result_second(self):
   w_hz,fr_h,after_ndata=wav_homework.filter_bank(self) # filter bank
   plt.subplot(3,2,1);plt.title("1 Channel Signal");plt.xlabel("Time");plt.ylabel("Amplitude");
   plt.plot(after_ndata[0],'black')
   plt.subplot(3,2,2);plt.title("2 Channel Signal");plt.xlabel("Time");plt.ylabel("Amplitude");
   plt.plot(after_ndata[1],'black')
   plt.subplot(3,2,3);plt.title("3 Channel Signal");plt.xlabel("Time");plt.ylabel("Amplitude");
   plt.plot(after_ndata[2],'black')
   plt.subplot(3,2,4);plt.title("4 Channel Signal");plt.xlabel("Time");plt.ylabel("Amplitude");
   plt.plot(after_ndata[3],'black')
   plt.subplot(3,2,5);plt.title("5 Channel Signal");plt.xlabel("Time");plt.ylabel("Amplitude");
   plt.plot(after_ndata[4],'black')
   plt.subplot(3,2,6);plt.title("6 Channel Signal");plt.xlabel("Time");plt.ylabel("Amplitude");
   plt.plot(after_ndata[5],'black')
   plt.tight_layout();plt.pause(0.00001)
   plt.show();plt.clf();
   
if __name__=="__main__":
    NBLOCKS = 100; DURATION = 0.1 
    FORMAT = pyaudio.paInt16; CHANNELS = 1 
    RATE = 8000; CHUNK = int(RATE * DURATION)
    fs = 8000; ch = 6 
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,
                    input=True,frames_per_buffer=CHUNK)
    frame_count = 0
    plt.figure(figsize=(8,6))
    while frame_count < NBLOCKS + 1:
        data = stream.read(CHUNK)
        data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
        data_np = np.array(data_int, dtype='b')[::2] + 128
        wav_homework(data_np,fs).show_result_first()
        #wav_homework(data_np,fs).show_result_second()
        if keyboard.is_pressed("q"):
            break
