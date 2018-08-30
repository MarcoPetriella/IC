# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
"""


"""
Descripción:
------------
Este script realiza un barrido en frecuencias utilizando la salida y entrada de audio como emisor-receptor.
Utiliza la libreria pyaudio para generar las señales de salida y la entrada de adquisición.
El script lanza dos thread o hilos que se ejecutan contemporaneamente: 
uno para la generación de la señal (producer) y otro para la adquisicón (consumer).
El script está programado para que el hilo productor envie una señal y habilite en ese momento la adquisición del hilo consumidor.
Se utilizan dos semaforos para la interección entre threads: 
    - semaphore1: señala el comienzo de cada paso del barrido de frecuencias, y avisa al consumidor que puede comenzar la adquisición.
    - semaphore2: señala que el hilo consumidor ya ha adquirido la señal y por lo tanto se puede comenzar la adquisición del siguiente paso del barrido.
La señal enviada se guarda en el array data_send, donde cada fila indica un paso del barrido 
La señal adquirida se guarda en el array data_acq, donde cada fila indica un paso del barrido 

Al final del script se agregan dos secciones para verificar el correcto funcionamiento del script y para medir el retardo
entre mediciones iguales (en este caso es necesario que delta_frec_hz = 0). En mi pc de escritorio el retardo entre señales medidas está dentro
de +/- 3 ms, que puede considerarse como la variabilidad del retardo entre el envío de la señal y la adquisición.

Algunas dudas:
--------------
    - El buffer donde se lee la adquisición guarda datos de la adquisicón correspondiente al paso anterior. Para evitar esto se borran 
    los primeros datos del buffer, pero no es muy profesional. La cantidad de datos agregados parece ser independiente del tiempo de adquisición
    o la duración de la señal enviada.
    - La señal digital que se envia debe ser cuatro veces mas larga que la que se envía analógicamente. No entiendo porqué.
    - Se puede mejorar la variabilidad en el retardo entre señal enviada y adquirida? Es decir se puede mejorar la sincronización entre los dos procesos?

Falta:
------
    - Mejorar la interrupción del script por el usuario. Por el momento termina únicamente cuando termina la corrida.

Notas:
--------
- Cambio semaforo por lock. Mejora la sincronización en +/- 1 ms. 
- Define un chunk_acq_eff que tiene en cuenta el delay inicial
- Cambiar Event por Lock no cambia mucho
- Cuando duration_sec_send > duration_sec_adq la variabilidad del retardo entre los procesos es aprox +/- 1 ms, salvo para la primera medición
- Cuando duration_sec_send < duration_sec_adq la variabilidad del retardo es muchas veces nula, salvo para la primera medición
- Obligo a que la duración de adquisición  > duración de la señal enviada para mejorar la sincronización.


Parametros:
-----------
fs = 44100*8 # frecuencia de sampleo en Hz
frec_ini_hz = 10 # frecuencia inicial de barrido en Hz
frec_fin_hz = 40000 # frecuencia inicial de barrido en Hz
steps = 50 # cantidad de pasos del barrido
duration_sec_send = 0.3 # duracion de la señal de salida de cada paso en segundos
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
import os

params = {'legend.fontsize': 'medium',
     #     'figure.figsize': (15, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)


def function_generator(parametros):
    
    fs = parametros['fs']
    frec = parametros['frec']
    amplitud = parametros['amplitud']
    duracion = parametros['duracion']
    tipo = parametros['tipo']
        
    if tipo is 'sin':
        output_signal = (amplitud*np.sin(2*np.pi*np.arange(int(duracion*fs))*frec/fs)).astype(np.float32)  
    elif tipo is 'square':
        output_signal = amplitud*signal.square(2*np.pi*np.arange(int(duracion*fs))*frec/fs, duty=0.5).astype(np.float32) 
    elif tipo is 'ramp':
        output_signal = amplitud*signal.sawtooth(2*np.pi*np.arange(int(duracion*fs))*frec/fs, width=0.5).astype(np.float32) 

    return output_signal


def play_rec(parametros):
    
    # Cargo parametros comunes a los dos canales
    fs = parametros['fs']
    duration_sec_send = parametros['duration_sec_send'] 
    steps_frec = parametros['steps_frec'] 
    input_channels = parametros['input_channels']
    output_channels = parametros['output_channels']
    
    # Estos parametros son distintos par cada canal
    frec_ini_hz = []
    frec_fin_hz = []
    amplitud = []
    delta_frec_hz = []
    tipo = []
    for i in range(output_channels):
        
        frec_ini_hz.append(parametros['frec_ini_hz_ch' + str(i)])
        frec_fin_hz.append(parametros['frec_fin_hz_ch' + str(i)])     
        amplitud.append(parametros['amplitud_ch' + str(i)])
        tipo.append(parametros['tipo_ch' + str(i)])
   
        if steps_frec == 1: 
            delta_frec_hz.append(0.)
            frec_fin_hz[i] = frec_ini_hz[i]
        else:
            delta_frec_hz.append((frec_fin_hz[i]-frec_ini_hz[i])/(steps_frec-1)) # paso del barrido en Hz
            
    # Obligo a la duracion de la adquisicion > a la de salida    
    duration_sec_acq = duration_sec_send + 0.1 # duracion de la adquisicón de cada paso en segundos
    
    # Inicia pyaudio
    p = pyaudio.PyAudio()
    
    # Defino los buffers de lectura y escritura
    chunk_send = int(fs*duration_sec_send)
    chunk_acq = int(fs*duration_sec_acq)
    
    # Defino el stream del parlante
    stream_output = p.open(format=pyaudio.paFloat32,
                    channels = output_channels,
                    rate = fs,
                    output = True,
                    
    )
    
    # Defino un buffer de lectura efectivo que tiene en cuenta el delay de la medición
    chunk_delay = int(fs*stream_output.get_output_latency()) 
    chunk_acq_eff = chunk_acq + chunk_delay
    
    # Defino el stream del microfono
    stream_input = p.open(format = pyaudio.paInt16,
                    channels = input_channels,
                    rate = fs,
                    input = True,
                    frames_per_buffer = chunk_acq_eff*p.get_sample_size(pyaudio.paInt16),
    )
    
    # Defino los semaforos para sincronizar la señal y la adquisicion
    lock1 = threading.Lock() # Este lock es para asegurar que la adquisicion este siempre dentro de la señal enviada
    lock2 = threading.Lock() # Este lock es para asegurar que no se envie una nueva señal antes de haber adquirido y guardado la anterior
    lock1.acquire() # Inicializa el lock, lo pone en cero.
    
    # Defino el thread que envia la señal
    data_send = np.zeros([steps_frec,chunk_send,output_channels],dtype=np.float32)  # aqui guardo la señal enviada
    frecs_send = np.zeros([steps_frec,output_channels])   # aqui guardo las frecuencias
    
    # Guardo los parametros de la señal de salida por canal
    parametros_output_signal_chs = []
    for i in range(output_channels):
        para = {}
        para['fs'] = fs
        para['amplitud'] = amplitud[i]
        para['duracion'] = parametros['duration_sec_send']
        para['tipo'] = tipo[i]
        
        parametros_output_signal_chs.append(para)
          
    
    def producer(steps_frec, delta_frec):  
        for i in range(steps_frec):
            
            # Genero las señales de salida para los canales
            samples = np.zeros([output_channels,4*chunk_send],dtype = np.float32)
            for j in range(output_channels):
                
                f = frec_ini_hz[j] + delta_frec_hz[j]*i       
                                       
                parametros_output_signal = parametros_output_signal_chs[j]
                parametros_output_signal['frec'] = f
                samples[j,0:chunk_send] = function_generator(parametros_output_signal)
                
                # Guardo las señales de salida
                data_send[i,:,j] = samples[j,0:chunk_send]
                frecs_send[i,j] = f
            
            # Paso la salida a un array de una dimension
            samples_out = np.reshape(samples,4*chunk_send*output_channels,order='F')
            
            
            for j in range(output_channels):
                print ('Frecuencia ch'+ str(j) +': ' + str(frecs_send[i,j]) + ' Hz')
            
            print ('Empieza Productor: '+ str(i))
            
            # Se entera que se guardó el paso anterior (lock2), avisa que comienza el nuevo (lock1), y envia la señal
            lock2.acquire() 
            lock1.release() 
            stream_output.start_stream()
            stream_output.write(samples_out)
            stream_output.stop_stream()        
    
        producer_exit[0] = True  
            
            
            
    # Defino el thread que adquiere la señal        
    data_acq = np.zeros([steps_frec,chunk_acq,input_channels],dtype=np.int16)  # aqui guardo la señal adquirida
    
    def consumer(steps_frec):
        for i in range(steps_frec):
            
            # Toma el lock, adquiere la señal y la guarda en el array
            lock1.acquire()
            stream_input.start_stream()
            stream_input.read(chunk_delay)  
            data_i = stream_input.read(chunk_acq)  
            stream_input.stop_stream()   
            
            data_i = np.frombuffer(data_i, dtype=np.int16)
            
            for j in range(input_channels):
                data_acq[i,:,j] = data_i[j::input_channels]
            
            print ('Termina Consumidor: '+ str(i))
            print ('')
            lock2.release() # Avisa al productor que terminó de escribir los datos y puede comenzar con el próximo step
    
        consumer_exit[0] = True  
           
    
    producer_exit = [False]   
    consumer_exit = [False] 
            
    # Inicio los threads    
    t1 = threading.Thread(target=producer, args=[steps_frec,delta_frec_hz])
    t2 = threading.Thread(target=consumer, args=[steps_frec])
    t1.start()
    t2.start()
    
         
    while(not producer_exit[0] or not consumer_exit[0]):
        time.sleep(0.2)
    
    
        
    stream_input.close()
    stream_output.close()
    p.terminate()   
    
    return data_acq, data_send, frecs_send
 



#%%
    
## Realiza medición y grafica
parametros = {}
parametros['fs'] = 44100 
parametros['steps_frec'] = 50 
parametros['duration_sec_send'] = 0.3
parametros['input_channels'] = 2
parametros['output_channels'] = 2
parametros['tipo_ch0'] = 'sin' 
parametros['amplitud_ch0'] = 0.1 
parametros['frec_ini_hz_ch0'] = 500 
parametros['frec_fin_hz_ch0'] = 500 
parametros['tipo_ch1'] = 'ramp' 
parametros['amplitud_ch1'] = 0.1 
parametros['frec_ini_hz_ch1'] = 500 
parametros['frec_fin_hz_ch1'] = 500 

data_acq, data_send, frecs_send = play_rec(parametros)

#%%
plt.plot(np.transpose(data_acq[:,:,1]))


#%%
### ANALISIS de la señal adquirida. Cheque que la señal adquirida corresponde a la enviada

fs = parametros['fs']

ch_acq = 0
ch_send = 0
ind_frec = 2

### Realiza la FFT de la señal enviada y adquirida
fft_send = abs(fft.fft(data_send[ind_frec,:,ch_send]))/int(data_send.shape[1]/2+1)
fft_send = fft_send[0:int(data_send.shape[1]/2+1)]
fft_acq = abs(fft.fft(data_acq[ind_frec,:,ch_acq]))/int(data_acq.shape[1]/2+1)
fft_acq = fft_acq[0:int(data_acq.shape[1]/2+1)]

frec_send = np.linspace(0,int(data_send.shape[1]/2),int(data_send.shape[1]/2+1))
frec_send = frec_send*(fs/2+1)/int(data_send.shape[1]/2+1)
frec_acq = np.linspace(0,int(data_acq.shape[1]/2),int(data_acq.shape[1]/2+1))
frec_acq = frec_acq*(fs/2+1)/int(data_acq.shape[1]/2+1)

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.1, .1, .75, .8])
ax1 = ax.twinx()
ax.plot(frec_send,fft_send, label='Frec enviada: ' + str(frecs_send[ind_frec,ch_send]) + ' Hz')
ax1.plot(frec_acq,fft_acq,color='red', label=u'Señal adquirida')
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()

#%%

### Estudo del retardo en caso que delta_frec = 0

ch_acq = 0
i_comp = 5

retardos = np.array([])
for i in range(data_acq.shape[0]):
    
    data_acq_i = data_acq[i,:,ch_acq]     
    corr = np.correlate(data_acq[i_comp,:,ch_acq] - np.mean(data_acq[i_comp,:,ch_acq]),data_acq_i - np.mean(data_acq_i),mode='full')
    pos_max = np.argmax(corr) - len(data_acq_i)
    retardos = np.append(retardos,pos_max)


    
fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.15, .15, .8, .8])
ax.hist(retardos,bins=1000, rwidth =0.99)    
ax.set_xlabel(u'Retardo')
ax.set_ylabel('Frecuencia [eventos]')
ax.set_title(u'Histograma de retardo respecto a la i = ' + str(i_comp) + ' medición')
ax1.legend(loc=4)
plt.show()    




#%%

Is = 1.0*1e-12
Vt = 26.0*1e-3
n = 1.

Vd = np.linspace(-1,1,1000)
Id = Is*(np.exp(Vd/n/Vt)-1)

Rs = 100
Vs = 1
Ir = Vs/Rs - Vd/Rs


plt.plot(Vd,Id)
plt.plot(Vd,Ir)

##%%
## Parametros
#parametros = {}
#parametros['fs'] = 44100*8 
#parametros['frec_ini_hz'] = 500 
#parametros['frec_fin_hz'] = 500 
#parametros['steps'] = 21 
#parametros['amplitud'] = 0.1 
#
#frec_inis = [0,100,1000,15000]
#frec_fins = [100,1000,15000,19000]
#duraciones = [2,1,0.5,0.5]
#
#carpeta_resultdos = 'respuesta_emisor_receptor'
#os.mkdir(carpeta_resultdos)
#
#
#for i in range(4):
#    
#    parametros['frec_ini_hz'] = frec_inis[i]
#    parametros['frec_fin_hz'] = frec_fins[i]
#    parametros['duration_sec_send'] = duraciones[i] 
#
#
#    data_acq, data_send, frecs_send = play_rec(parametros)
#    
#    np.save(os.path.join(carpeta_resultdos, 'data_acq_rango_'+str(i)),data_acq)
#    np.save(os.path.join(carpeta_resultdos, 'data_send_rango_'+str(i)),data_send)
#    np.save(os.path.join(carpeta_resultdos, 'frecs_send_rango_'+str(i)),frecs_send)
#    np.save(os.path.join(carpeta_resultdos, 'parametros_rango_'+str(i)),parametros)
#
#
#i = 0
#dd = np.load(os.path.join(carpeta_resultdos, 'data_acq_rango_' + str(i) + '.npy'))  
#params = np.load(os.path.join(carpeta_resultdos, 'parametros_rango_' + str(i) + '.npy'))
#
#plt.plot(np.transpose(dd))
#
#
#plt.plot(dd[5,:])