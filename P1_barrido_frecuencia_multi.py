# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
"""


"""
Este script realiza un barrido en frecuencias utilizando la salida y entrada de audio como emisor-receptor.
Utiliza la libreria pyaudio para generar las señales de salida y la entrada de adquisición.
El script lanza dos thread o hilos que se ejecutan contemporaneamente: 
uno para la generación de la señal (producer) y otro para la adquisicón (consumer).
El proceso está programado para que el hilo productor envie una señal y habilite en ese momento la adquisición del hilo consumidor.
Se utilizan dos semaforos para la interección entre threads: 
    - semaphore1 indica el comienzo de cada paso en el barrido de frecuencia y avisa al consumidor que puede comenzar la adquisición
    - semaphore2 indica que el hilo consumidor ya ha adquirido la señal y por lo tanto se puede comenzar la adquisición de siguiente paso
La señal enviada se guarda en el array data_send, donde cada fila indica un paso del barrido 
La señal adquirida se guarda en el array data_acq, donde cada fila indica un paso del barrido 

Algunas dudas:
    - el buffer donde se lee la adquisición guarda datos de la adquisicón correspondiente al paso anterior. Para evitar esto se borran 
    los primeros datos del buffer, pero no es muy profesional. La cantidad de datos agregados parece ser independiente del tiempo de adquisición
    o la duración de la señal enviada.
    - la señal digital que se envia debe ser cuatro veces mas larga que la que definitivamente se envia analógicamente. No entiendo porqué.

Falta:
    - mejorar la interrupción del script por el usuario. Por el momento termina únicamente cuando termina la corrida.

Al final de script se agregan dos secciones para verificar el correcto funcionamiento del script y para medir el retardo
entre mediciones iguales.


Parametros
----------
fs = 44100 # frecuencia de sampleo en Hz
frec_ini_hz = 440 # frecuencia inicial de barrido en Hz
steps = 10 # cantidad de pasos del barrido
delta_frec_hz = 50 # paso del barrido en Hz
duration_sec_send = 2 # duracion de la señal de salida de cada paso en segundos
duration_sec_acq = 0.2 # duracion de la adquisicón de cada paso en segundos
A = 0.1 # Amplitud de la señal de salida

"""

#%%
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import numpy.fft as fft
import datetime
import time
import matplotlib.pylab as pylab
from scipy import signal

params = {'legend.fontsize': 'medium',
     #     'figure.figsize': (15, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)


fs = 44100 # frecuencia de sampleo en Hz
frec_ini_hz = 840 # frecuencia inicial de barrido en Hz
steps = 10 # cantidad de pasos del barrido
delta_frec_hz = -40 # paso del barrido en Hz
duration_sec_send = 0.5 # duracion de la señal de salida de cada paso en segundos
duration_sec_acq = 0.70# duracion de la adquisicón de cada paso en segundos
A = 0.1 # Amplitud de la señal de salida

chunk_acq = int(fs*duration_sec_acq)
chunk_send = int(fs*duration_sec_send)

p = pyaudio.PyAudio()

   
# Defino el stream del microfono
stream_input = p.open(format = pyaudio.paInt16,
                channels = 1,
                rate = fs,
                input = True,
                frames_per_buffer = chunk_acq,
)

# defino el stream del parlante
stream_output = p.open(format=pyaudio.paFloat32,
                channels = 1,
                rate = fs,
                output = True,
                #input_device_index = 4,
                
)

#print (stream_output.get_output_latency()*1000)


# Defino los semaforos para sincronizar la señal y la adquisicion
semaphore1 = threading.Semaphore() # Este semaforo es para asegurar que la adquisicion este siempre dentro de la señal enviada
semaphore2 = threading.Semaphore() # Este semaforo es para asegurar que no se envie una nueva señal antes de haber adquirido y guardado la anterior
semaphore1.acquire() # Inicializa el semaforo, lo pone en cero.

# Defino el thread que envia la señal
data_send = np.zeros([steps,chunk_send],dtype=np.float32)  # aqui guardo la señal enviada
frecs_send = np.zeros(steps)   # aqui guardo las frecuencias

tic = datetime.datetime.now()
def producer(steps, delta_frec):  
    global producer_exit
    i = 0
    while(i<steps):
        f = frec_ini_hz + delta_frec_hz*i
        
        samples = (A*np.sin(2*np.pi*np.arange(1*chunk_send)*f/fs)).astype(np.float32) 
        samples = np.append(samples, np.zeros(3*chunk_send).astype(np.float32))
        #samples = np.append(samples, (A/2*np.sin(2*np.pi*np.arange(1*int(fs*0.1))*f/fs)).astype(np.float32))
        #samples = np.append(samples, np.zeros(3*chunk_send).astype(np.float32))
        #samples = A*signal.square(2*np.pi*np.arange(np.dtype(np.float32).itemsize*chunk_send)*f/fs).astype(np.float32)  
        data_send[i][:] = samples[0:chunk_send]
        frecs_send[i] = f
        semaphore2.acquire() # Se da por avisado que terminó el step anterior
        semaphore1.release() # Avisa al consumidor que comienza la adquisicion

        print ('Frecuencia: ' + str(f) + ' Hz')
        print ('Empieza Productor: '+ str(i))
        i = i + 1
        
        # Envia la señal y la guarda en el array
        stream_output.start_stream()
        stream_output.write(samples)
        stream_output.stop_stream()

    producer_exit = True  
        
        
        
# Defino el thread que adquiere la señal        
data_acq = np.zeros([steps,chunk_acq],dtype=np.int16)  # aqui guardo la señal adquirida
def consumer():
    global consumer_exit
    j = 0
    while(j<steps):
        semaphore1.acquire() # Se da por avisado que que el productor comenzó un nuevo step
        
        # Adquiere la señal y la guarda en el array
        stream_input.start_stream()
        data_i = stream_input.read(int(fs*stream_output.get_output_latency())) # esto lo pongo porque el buffer parece quedar lleno de la medicion anterior
        data_i = stream_input.read(chunk_acq)  
        stream_input.stop_stream()      
        
        data_acq[j][:] = np.frombuffer(data_i, dtype=np.int16)
        
        print ('Termina Consumidor: '+ str(j))
        print ('')
        j = j + 1
        semaphore2.release() # Avisa al productor que terminó de escribir los datos y puede comenzar con el próximo step

    consumer_exit = True  
       

producer_exit = False   
consumer_exit = False 
        
# Inicio los threads    
t1 = threading.Thread(target=producer, args=[steps,delta_frec_hz])
t2 = threading.Thread(target=consumer, args=[])
t1.start()
t2.start()

     
while(not producer_exit or not consumer_exit):
    time.sleep(np.max([duration_sec_acq,duration_sec_send]))


    
stream_input.close()
stream_output.close()
p.terminate()   
 


#%%

### ANALISIS de la señal adquirida

# Elijo la frecuencia
ind_frec = 1


### Muestra la serie temporal de las señales enviadas y adquiridas
t_send = np.linspace(0,np.size(data_send,1)-1,np.size(data_send,1))/fs
t_adq = np.linspace(0,np.size(data_acq,1)-1,np.size(data_acq,1))/fs

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.15, .15, .8, .8])
ax1 = ax.twinx()
ax.plot(t_send,data_send[ind_frec,:], label=u'Señal enviada: ' + str(frecs_send[ind_frec]) + ' Hz')
ax1.plot(t_adq,data_acq[ind_frec,:],color='red', label=u'Señal adquirida')
ax.set_xlabel('Tiempo [seg]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()

### Realiza la FFT de la señal enviada y adquirida
fft_send = abs(fft.fft(data_send[ind_frec,:]))
fft_send = fft_send[0:int(chunk_send/2+1)]
fft_acq = abs(fft.fft(data_acq[ind_frec,:]))
fft_acq = fft_acq[0:int(chunk_acq/2+1)]

frec_send = np.linspace(0,int(chunk_send/2),int(chunk_send/2+1))
frec_send = frec_send/duration_sec_send
frec_acq = np.linspace(0,int(chunk_acq/2),int(chunk_acq/2+1))
frec_acq = frec_acq/duration_sec_acq

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.1, .1, .8, .8])
ax1 = ax.twinx()
ax.plot(frec_send,fft_send, label='Frec enviada: ' + str(frecs_send[ind_frec]) + ' Hz')
ax1.plot(frec_acq,fft_acq,color='red', label=u'Señal adquirida')
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()

#%%

## Estudo del retardo en caso que delta_frec = 0

retardos = np.array([])
for i in range(steps):
    
    data_acq_i = data_acq[i,:]     
    corr = np.correlate(data_acq[0,:] - np.mean(data_acq[0,:]),data_acq_i - np.mean(data_acq_i),mode='full')
    pos_max = np.argmax(corr) - len(data_acq_i)
    retardos = np.append(retardos,pos_max/fs)


    
fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.15, .15, .8, .8])
ax.hist(1000*retardos,bins=10, rwidth =0.99)    
ax.set_xlabel(u'Retardo [ms]')
ax.set_ylabel('Frecuencia [eventos]')
ax.set_title(u'Histograma de retardo respecto a la primera medición')
ax1.legend(loc=4)
plt.show()    


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.15, .15, .8, .8])
ax.hist(retardos*frec_ini_hz,bins=10, rwidth =0.99)    
ax.set_xlabel(u'Retardo relativo [periodo]')
ax.set_ylabel('Frecuencia [eventos]')
ax.set_title(u'Histograma de retardo relativo a la duración del período respecto a la primera medición')
ax1.legend(loc=4)
plt.show()    