import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import signal
import datetime

chunk = 1024
format_input = pyaudio.paInt16
fs = 44100
f = 440
duration = 0.5

p = pyaudio.PyAudio()

#p.get_device_count()
#p.get_default_input_device_info()
#info = p.get_host_api_info_by_index(0)
#numdevices = info.get('deviceCount')
#for i in range(0, numdevices):
#        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#            print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
   
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)   

            
#stream_input = p.open(format = pyaudio.paInt16,
#                channels = 1,
#                rate = fs,
#                input = True,
#                frames_per_buffer = chunk,
#                stream_callback=callback_input
#)


stream_output = p.open(format=pyaudio.paFloat32,
                channels = 1,
                rate = fs,
                output = True,
)

#stream_input.start_stream()
tic = datetime.datetime.now()
stream_output.write(samples)
toc = datetime.datetime.now()

print ((toc-tic).total_seconds())

               
#all = []
#for i in range(0, int(fs / chunk * duration)):
#    data = stream_input.read(chunk, exception_on_overflow = False)
#    all.append(data)
#
#
#decoded = np.frombuffer(data, dtype=np.int16)
#
#plt.hist(decoded,bins=7)


#stream_input.close()
stream_output.close()
p.terminate()

