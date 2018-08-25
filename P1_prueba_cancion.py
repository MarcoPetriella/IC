import pyaudio
import wave
import sys
import datetime

CHUNK = 10*44100


wf = wave.open('cancion.wav', 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# read data
data = wf.readframes(CHUNK)

# play stream (3)
while len(data) > 0:
    stream.write(data)
    tic = datetime.datetime.now()
    data = wf.readframes(CHUNK)
    toc = datetime.datetime.now()
    print ((toc-tic).total_seconds()*1000)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()






