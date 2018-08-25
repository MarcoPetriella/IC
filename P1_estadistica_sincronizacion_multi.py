# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
"""

#%% Seccion1

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import numpy.fft as fft
import datetime
import time


fs = 44100 # frecuencia de sampleo en Hz
frec_ini_hz = 440 # frecuencia inicial de barrido en Hz
steps = 100 # cantidad de pasos del barrido
duration_sec_send = 1 # duracion de cada paso en segundos
duration_sec_acq = 0.4
delta_frec_hz = 0 # paso del barrido en Hz
A = 0.1

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


# Defino los semaforos para sincronizar la señal y la adquisicion
semaphore1 = threading.Semaphore() # Este semaforo es para asegurar que la adquisicion este siempre dentro de la señal enviada
semaphore2 = threading.Semaphore() # Este semaforo es para asegurar que no se envie una nueva señal antes de haber adquirido y guardado la anterior
semaphore1.acquire() # Inicializa el semaforo

# Defino el thread que envia la señal
data_send = np.zeros([steps,chunk_send],dtype=np.float32)  # aqui guardo la señal enviada
frecs_send = np.zeros(steps)   # aqui guardo las frecuencias

tic = datetime.datetime.now()
def producer(steps, delta_frec):  
    global producer_exit
    for i in range(steps):
        f = frec_ini_hz + delta_frec_hz*i
        samples = (A*np.sin(2*np.pi*np.arange(np.dtype(np.float32).itemsize*chunk_send)*f/fs)).astype(np.float32)   
        data_send[i][:] = samples[0:int(len(samples)/np.dtype(np.float32).itemsize)]
        frecs_send[i] = f
        semaphore2.acquire() # Se da por avisado que terminó el step anterior
        semaphore1.release() # Avisa al consumidor que comienza la adquisicion

        print ('Frecuencia: ' + str(f) + ' Hz')
        print ('Empieza Productor: '+ str(i))
        
        # Envia la señal y la guarda en el array
        stream_output.start_stream()
        stream_output.write(samples)
        stream_output.stop_stream()


    producer_exit = True  
        
        
        
# Defino el thread que adquiere la señal        
data_acq = np.zeros([steps,chunk_acq],dtype=np.int16)  # aqui guardo la señal adquirida
def consumer():
    global consumer_exit
    count = 0
    while(count<steps):
        semaphore1.acquire() # Se da por avisado que que el productor comenzó un nuevo step
        
        # Adquiere la señal y la guarda en el array
        stream_input.start_stream()
        data_i = stream_input.read(int(fs*stream_output.get_output_latency())) # esto lo pongo porque el buffer parece quedar lleno de la medicion anterior
        data_i = stream_input.read(chunk_acq)   
        stream_input.stop_stream()      
        
        data_acq[count][:] = np.frombuffer(data_i, dtype=np.int16)
        
        print ('Termina Consumidor: '+ str(count))
        count = count + 1
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
# Elijo la frecuencia
ind_frec = 0

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


retardos = np.array([])
for i in range(steps):
    
    data_acq_i = data_acq[i,:]     
    corr = np.correlate(data_acq[0,:] - np.mean(data_acq[0,:]),data_acq_i - np.mean(data_acq_i),mode='full')
    pos_max = np.argmax(corr) - len(data_acq_i)
    retardos = np.append(retardos,pos_max/fs)


    
fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.15, .15, .8, .8])
ax.hist(retardos,bins=10, rwidth =0.99)    
ax.set_xlabel(u'Retardo [seg]')
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


#retardos = np.array([])
#for i in range(steps):
#    
#    data_acq_i = data_acq[i,:]
#    a = np.abs(data_acq_i) > 0.02*np.max(data_acq_i)
#    b = a.nonzero()
#    retardos = np.append(retardos,b[0][0]/fs)
#
#
#    
#fig = plt.figure(figsize=(14, 7), dpi=250)
#ax = fig.add_axes([.15, .15, .8, .8])
#ax.hist(retardos,bins=10)    
#ax.set_xlabel(u'Retardo [seg]')
#ax.set_ylabel('Frecuencia [eventos]')
#ax1.legend(loc=4)
#plt.show()   


## Grafico para estudiar el retardo entre la señal enviada y medida
#tiempo_ini_seg = 0.00
#tiempo_fin_seg = 0.08
#retardos = np.array([])
#for i in range(steps):
#    
#    data_send_i = data_send[i,int(tiempo_ini_seg*fs):int(tiempo_fin_seg*fs)]
#    data_acq_i = data_acq[i,int(tiempo_ini_seg*fs):int(tiempo_fin_seg*fs)]
#    
#    corr = np.correlate(data_send_i-np.mean(data_send_i),data_acq_i-np.mean(data_acq_i),mode='full')
#    pos_max = np.argmax(corr) - len(data_send_i)
#    retardos = np.append(retardos,pos_max)
#    
#
#fig = plt.figure(figsize=(14, 7), dpi=250)
#ax = fig.add_axes([.15, .15, .8, .8])
#ax.hist(retardos/fs*frec_ini_hz,bins=50)    
#ax.set_xlabel(u'Retardo/Periodo señal')
#ax.set_ylabel('Frecuencia [eventos]')
#ax1.legend(loc=4)
#plt.show()

   



