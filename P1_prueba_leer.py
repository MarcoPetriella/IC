"""PyAudio Example: Play a wave file."""

import pyaudio
import numpy as np
import wave
import sys

CHUNK = 1024

p = pyaudio.PyAudio()
volume = 0.8

fs = 44100
duration = 30.
f = 50.
