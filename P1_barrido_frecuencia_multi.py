# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
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
from sys import stdout

    
    
params = {'legend.fontsize': 'medium',
     #     'figure.figsize': (15, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)


def barra_progreso(paso,pasos_totales,leyenda,tiempo_ini):

    """
    Esta función genera una barra de progreso
    
    Parametros:
    -----------
    paso : int, paso actual
    pasos_totales : int, cantidad de pasos totales
    leyenda : string, leyenda de la barra
    tiempo_ini : datetime, tiempo inicial de la barra en formato datetime   
    
    Autores: Leslie Cusato, Marco Petriella
    """    
    n_caracteres = 30    
    barrita = int(n_caracteres*(paso+1)/pasos_totales)*chr(9632)    
    ahora = datetime.datetime.now()
    eta = (ahora-tiempo_ini).total_seconds()
    eta = (pasos_totales - paso - 1)/(paso+1)*eta
    
    if paso == 0:
        print("\n")        
       
    stdout.write("\r %s %s %3d %s |%s| %s %6.1f %s" % (leyenda,':',  int(100*(paso+1)/pasos_totales), '%', barrita.ljust(n_caracteres),'ETA:',eta,'seg'))
    
    
    
    

def function_generator(parametros):
    
    """
    Esta función genera señales de tipo seno, cuadrada y rampa.
    
    Parametros:
    -----------
    Para el ingreso de los parametros de adquisición se utiliza un diccionario.

    fs : int, frecuencia de sampleo de la placa de audio. Valor máximo 44100*8 Hz. [Hz] 
    frec : float, frecuencia de la señal. [Hz] 
    amplitud : float, amplitud de la señal.
    duracion : float, tiempo de duración de la señal. [seg]
    tipo : {'square', 'sin', 'ramp', 'constant'}, tipo de señal.   
    
    Salida (returns):
    -----------------
    output_signal : numpy array, señal de salida.
    
    Autores: Leslie Cusato, Marco Petriella
    """

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
    elif tipo is 'constant':
        output_signal = amplitud*(int(duracion*fs)).astype(np.float32) 

    return output_signal


def play_rec(parametros):
    
    
    """
    Descripción:
    ------------
    Esta función permite utilizar la placa de audio de la pc como un generador de funciones / osciloscopio
    con dos canales de entrada y dos de salida. Para ello utiliza la libreria pyaudio y las opciones de write() y read()
    para escribir y leer los buffer de escritura y lectura. Para realizar el envio y adquisición simultánea de señales, utiliza
    un esquema de tipo productor-consumidor que se ejecutan en thread o hilos diferenntes. Para realizar la comunicación 
    entre threads y evitar overrun o sobreescritura de los datos del buffer de lectura se utilizan dos variables de tipo block.
    El block1 se activa desde proceso productor y avisa al consumidor que el envio de la señal ha comenzado y que por lo tanto 
    puede iniciar la adquisición. 
    El block2 se activa desde el proceso consumidor y aviso al productor que la lesctura de los datos ha finalizado y por lo tanto
    puede comenzar un nuevo paso del barrido. 
    Teniendo en cuenta que existe un retardo entre la señal enviada y adquirida, y que existe variabilidad en el retardo; se puede
    utilizar el canal 0 de entrada y salida para el envio y adquisicón de una señal de disparo que permita sincronizar las mediciones.
    Notar que cuando se pone output_channel = 1, en la segunda salida pone la misma señal que en el channel 1 de salida.
    
    Parámetros:
    -----------
    Para el ingreso de los parametros de adquisición se utiliza un diccionario.
    
    parametros = {}
    parametros['fs'] : int, frecuencia de sampleo de la placa de audio. Valor máximo 44100*8 Hz. [Hz]
    parametros['steps_frec'] : int, cantidad de pasos del barrido de frecuencias.
    parametros['steps_amplitud'] : int, cantidad de pasos del barrido de amplitudes.
    parametros['duration_sec_send'] : float, tiempo de duración de la adquisición. [seg]
    parametros['input_channels'] : int, cantidad de canales de entrada.
    parametros['output_channels'] : int, cantidad de canales de salida.
    parametros['tipo_ch0'] : {'square', 'sin', 'ramp', 'constant'}, tipo de señal enviada en el canal 0.
    parametros['amplitud_ini_ch0'] : float, amplitud inicial de la señal del canal 0. [V]. Máximo valor 1 V.
    parametros['amplitud_fin_ch0'] : float, amplitud final de la señal del canal 0. [V]. Máximo valor 1 V.
    parametros['frec_ini_hz_ch0'] : float, frecuencia inicial del barrido del canal 0. [Hz] 
    parametros['frec_fin_hz_ch0'] : float, frecuencia final del barrido del canal 0. [Hz] 
    parametros['tipo_ch1'] : {'square', 'sin', 'ramp'}, tipo de señal enviada en el canal 1.
    parametros['amplitud_ini_ch1'] : float, amplitud inicial de la señal del canal 1. [V]. Máximo valor 1 V.
    parametros['amplitud_fin_ch1'] : float, amplitud final de la señal del canal 1. [V]. Máximo valor 1 V.
    parametros['frec_ini_hz_ch1'] : float, frecuencia inicial del barrido del canal 1. [Hz] 
    parametros['frec_fin_hz_ch1'] : float, frecuencia final del barrido del canal 1. [Hz] 
    parametros['corrige_retardos'] = {'si','no'}, corrige el retardo utilizando la función sincroniza_con_trigger
    
    Salida (returns):
    -----------------
    data_acq: numpy array, array de tamaño [steps_frec][steps_amplitud][muestras_por_pasos_input][input_channels]
    data_send: numpy array, array de tamaño [steps_frec][steps_amplitud][muestras_por_pasos_output][output_channels]
    frecs_send: numpy array, array de tamaño [steps_frec][steps_amplitud][output_channels]
    amplitud_send: numpy array, array de tamaño [steps_frec][steps_amplitud][output_channels]
    retardos: numpy_array, array de tamaño [steps_frec*steps_amplitud] con los retardos entre trigger enviado y adquirido
    
    Las muestras_por_pasos está determinada por los tiempos de duración de la señal enviada y adquirida. El tiempo entre 
    muestras es 1/fs.
    
    Ejemplo:
    --------
    
    parametros = {}
    parametros['fs'] = 44100
    parametros['steps_frec'] = 10 
    parametros['steps_amplitud'] = 10 
    parametros['duration_sec_send'] = 0.01
    parametros['input_channels'] = 2
    parametros['output_channels'] = 2
    
    parametros['tipo_ch0'] = 'square' 
    parametros['amplitud_ini_ch0'] = 0.1 
    parametros['amplitud_fin_ch0'] = 0.1 
    parametros['frec_ini_hz_ch0'] = 500 
    parametros['frec_fin_hz_ch0'] = 500 
    
    parametros['tipo_ch1'] = 'sin' 
    parametros['amplitud_ini_ch1'] = 0.01 
    parametros['amplitud_fin_ch1'] = 0.1 
    parametros['frec_ini_hz_ch1'] = 500 
    parametros['frec_fin_hz_ch1'] = 5000
    
    parametros['corrige_retardos'] = 'si'
    
    data_acq, data_send, frecs_send, amplitudes_send, retardos = play_rec(parametros)

    
    
    Autores: Leslie Cusato, Marco Petriella    
    """    
    
    # Cargo parametros comunes a los dos canales
    fs = parametros['fs']
    duration_sec_send = parametros['duration_sec_send'] 
    steps_frec = parametros['steps_frec'] 
    steps_amplitud = parametros['steps_amplitud'] 
    input_channels = parametros['input_channels']
    output_channels = parametros['output_channels']
    corrige_retardos = parametros['corrige_retardos']
    
    # Estos parametros son distintos par cada canal
    frec_ini_hz = []
    frec_fin_hz = []
    amplitud_ini = []
    amplitud_fin = []
    delta_frec_hz = []
    delta_amplitud = []
    tipo = []
    for i in range(output_channels):
        
        frec_ini_hz.append(parametros['frec_ini_hz_ch' + str(i)])
        frec_fin_hz.append(parametros['frec_fin_hz_ch' + str(i)])     
        amplitud_ini.append(parametros['amplitud_ini_ch' + str(i)])
        amplitud_fin.append(parametros['amplitud_fin_ch' + str(i)])
        tipo.append(parametros['tipo_ch' + str(i)])
   
        if steps_frec == 1: 
            delta_frec_hz.append(0.)
            frec_fin_hz[i] = frec_ini_hz[i]
        else:
            delta_frec_hz.append((frec_fin_hz[i]-frec_ini_hz[i])/(steps_frec-1)) # paso del barrido en Hz

        if steps_amplitud == 1: 
            delta_amplitud.append(0.)
            amplitud_fin[i] = amplitud_ini[i]
        else:
            delta_amplitud.append((amplitud_fin[i]-amplitud_ini[i])/(steps_amplitud-1)) # paso del barrido en Hz
            
    # Obligo a la duracion de la adquisicion > a la de salida    
    duration_sec_acq = duration_sec_send + 0.2 
    
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
       
    # Guardo los parametros de la señal de salida por canal, para usarlos en la funcion function_generator
    parametros_output_signal_chs = []
    for i in range(output_channels):
        para = {}
        para['fs'] = fs
        para['duracion'] = parametros['duration_sec_send']
        para['tipo'] = tipo[i]
        
        parametros_output_signal_chs.append(para)

    # Defino el thread que envia la señal
    data_send = np.zeros([steps_frec,steps_amplitud,chunk_send,output_channels],dtype=np.float32)  # aqui guardo la señal enviada
    frecs_send = np.zeros([steps_frec,steps_amplitud,output_channels])
    amplitudes_send = np.zeros([steps_frec,steps_amplitud,output_channels])
          
    def producer(steps_frec,steps_amplitud):  
        for i in range(steps_frec):
            
            for k in range(steps_amplitud):
                            
                # Genero las señales de salida para los canales
                samples = np.zeros([output_channels,4*chunk_send],dtype = np.float32)
                for j in range(output_channels):
                    
                    f = frec_ini_hz[j] + delta_frec_hz[j]*i  
                    amplitud = amplitud_ini[j] + delta_amplitud[j]*k
                                           
                    parametros_output_signal = parametros_output_signal_chs[j]
                    parametros_output_signal['frec'] = f
                    parametros_output_signal['amplitud'] = amplitud
                    samples[j,0:chunk_send] = function_generator(parametros_output_signal)                    
                    
                    # Guardo las señales de salida
                    data_send[i,k,:,j] = samples[j,0:chunk_send]
                    frecs_send[i,k,j] = f
                    amplitudes_send[i,k,j] = amplitud
                
                # Paso la salida a un array de una dimension
                samples_out = np.reshape(samples,4*chunk_send*output_channels,order='F')
                
                
                
                # Se entera que se guardó el paso anterior (lock2), avisa que comienza el nuevo (lock1), y envia la señal
                lock2.acquire() 
                lock1.release() 
                stream_output.start_stream()
                stream_output.write(samples_out)
                stream_output.stop_stream()        
    
        producer_exit[0] = True  
            
            
    # Donde se guardan los resultados                     
    data_acq = np.zeros([steps_frec,steps_amplitud,chunk_acq,input_channels],dtype=np.int16)  
        
    # Defino el thread que adquiere la señal   
    def consumer(steps_frec,steps_amplitud):
        cont = 0
        tiempo_ini = datetime.datetime.now()
        for i in range(steps_frec):
            
            for k in range(steps_amplitud):
                cont = cont + 1
            
                # Toma el lock, adquiere la señal y la guarda en el array
                lock1.acquire()
                stream_input.start_stream()
                stream_input.read(chunk_delay)  
                data_i = stream_input.read(chunk_acq)  
                stream_input.stop_stream()   
                
                data_i = -np.frombuffer(data_i, dtype=np.int16)                            
                
                # Guarda la salida                   
                for j in range(input_channels):
                    data_acq[i,k,:,j] = data_i[j::input_channels]                   
                    
                # Barra de progreso
                barra_progreso(cont-1,steps_frec*steps_amplitud,'Progreso barrido',tiempo_ini)          
                                
                lock2.release() # Avisa al productor que terminó de escribir los datos y puede comenzar con el próximo step
    
        consumer_exit[0] = True  
           
    # Variables de salida de los threads
    producer_exit = [False]   
    consumer_exit = [False] 
            
    # Inicio los threads    
    t1 = threading.Thread(target=producer, args=[steps_frec,steps_amplitud])
    t2 = threading.Thread(target=consumer, args=[steps_frec,steps_amplitud])
    t1.start()
    t2.start()
             
    while(not producer_exit[0] or not consumer_exit[0]):
        time.sleep(0.2)
         
    stream_input.close()
    stream_output.close()
    p.terminate()   
    
    retardos = np.array([])
    if corrige_retardos is 'si':
        parametros_retardo = {}
        parametros_retardo['data_send'] = data_send
        parametros_retardo['data_acq']  = data_acq        
        data_acq, retardos = sincroniza_con_trigger(parametros_retardo)       
    
    return data_acq, data_send, frecs_send, amplitudes_send, retardos
 

def sincroniza_con_trigger(parametros):
    
    """
    Esta función corrige el retardo de las mediciones adquiridas con la función play_rec. Para ello utiliza la señal de 
    trigger enviada y adquirida en el canal 0 de la placa de audio, y sincroniza las mediciones de todos los canales de entrada. 
    El retardo se determina a partir de realizar la correlación cruzada entre la señal enviada y adquirida, y encontrando la posición
    del máximo del resultado.
    
    
    Parámetros:
    -----------
    data_acq: numpy array, array de tamaño [steps_frec][steps_amplitud][muestras_por_pasos_input][input_channels]
    data_send: numpy array, array de tamaño [steps_frec][steps_amplitud][muestras_por_pasos_output][output_channels]
    
    Salida (returns):
    -----------------
    data_acq_corrected : numpy array, señal de salida con retardo corregido de tamaño [steps_frec][steps_amplitud][muestras_por_pasos_input][input_channels]. 
                         El tamaño de la segunda dimensión es la misma que la de data_send.
    retardos : numpy array, array con los retardos de tamaño [steps_frec*steps_amplitud].
    
    Autores: Leslie Cusato, Marco Petriella   
    """
    
    data_send = parametros['data_send']
    data_acq = parametros['data_acq']   
    
    extra = 0
        
    data_acq_corrected = np.zeros([data_send.shape[0],data_send.shape[1],data_send.shape[2]+extra,data_acq.shape[3]])
    retardos = np.array([])
    
    cont = 0
    tiempo_ini = datetime.datetime.now()
    for k in range(data_acq.shape[1]):
        
        trigger_send = data_send[:,k,:,0]
        trigger_acq = data_acq[:,k,:,0]            
    
        for i in range(data_acq.shape[0]):
            cont = cont + 1
            barra_progreso(cont-1,data_acq.shape[1]*data_acq.shape[0],u'Progreso corrección',tiempo_ini) 

            corr = np.correlate(trigger_send[i,:] - np.mean(trigger_send[i,:]),trigger_acq[i,:] - np.mean(trigger_acq[i,:]),mode='full')
            pos_max = trigger_acq.shape[1] - np.argmax(corr)-1
            retardos = np.append(retardos,pos_max)
            
            for j in range(data_acq.shape[3]):
                data_acq_corrected[i,k,:,j] = data_acq[i,k,pos_max:pos_max+trigger_send.shape[1]+extra,j]
        
        
    return data_acq_corrected, retardos

#%%
    
## Realiza medición y grafica
parametros = {}
parametros['fs'] = 44100*8
parametros['steps_frec'] = 10
parametros['steps_amplitud'] = 10
parametros['duration_sec_send'] = 0.01
parametros['input_channels'] = 2
parametros['output_channels'] = 2

parametros['tipo_ch0'] = 'sin' 
parametros['amplitud_ini_ch0'] = 0.1 
parametros['amplitud_fin_ch0'] = 0.1 
parametros['frec_ini_hz_ch0'] = 500 
parametros['frec_fin_hz_ch0'] = 500 

parametros['tipo_ch1'] = 'sin' 
parametros['amplitud_ini_ch1'] = 0.01 
parametros['amplitud_fin_ch1'] = 0.1 
parametros['frec_ini_hz_ch1'] = 500 
parametros['frec_fin_hz_ch1'] = 5000

parametros['corrige_retardos'] = 'si'

data_acq, data_send, frecs_send, amplitudes_send, retardos = play_rec(parametros)


plt.plot(np.transpose(data_acq[:,0,:,1]))


#%%

### Corrige retardo y grafica
#parametros_retardo = {}
#parametros_retardo['data_send'] = data_send
#parametros_retardo['data_acq']  = data_acq
#
#data_acq, retardos = sincroniza_con_trigger(parametros_retardo)

#%%
ch = 1
ind_frec = 2
ind_amplitud = 0

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax1 = ax.twinx()
ax.plot(np.transpose(data_acq[ind_frec,ind_amplitud,:,ch]),'-',color='r', label='señal adquirida')
ax1.plot(np.transpose(data_send[ind_frec,ind_amplitud,:,ch]),'-',color='b', label='señal enviada')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.1, .1, .75, .8])
ax.hist(retardos, bins=100)
ax.set_title(u'Retardos')
ax.set_xlabel('Retardos')
ax.set_ylabel('Frecuencia')


#%%
### ANALISIS de la señal adquirida. Cheque que la señal adquirida corresponde a la enviada

fs = parametros['fs']

ch_acq = 1
ch_send = 1
ind_frec = 5
ind_amplitud = 0

### Realiza la FFT de la señal enviada y adquirida
fft_send = abs(fft.fft(data_send[ind_frec,ind_amplitud,:,ch_send]))/int(data_send.shape[2]/2+1)
fft_send = fft_send[0:int(data_send.shape[2]/2+1)]
fft_acq = abs(fft.fft(data_acq[ind_frec,ind_amplitud,:,ch_acq]))/int(data_acq.shape[2]/2+1)
fft_acq = fft_acq[0:int(data_acq.shape[2]/2+1)]

frec_send = np.linspace(0,int(data_send.shape[2]/2),int(data_send.shape[2]/2+1))
frec_send = frec_send*(fs/2+1)/int(data_send.shape[2]/2+1)
frec_acq = np.linspace(0,int(data_acq.shape[2]/2),int(data_acq.shape[2]/2+1))
frec_acq = frec_acq*(fs/2+1)/int(data_acq.shape[2]/2+1)

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax1 = ax.twinx()
ax.plot(frec_send,fft_send,'-' ,label='Frec enviada: ' + str(frecs_send[ind_frec,ind_amplitud,ch_send]) + ' Hz - Amplutud enviada: ' + str(amplitudes_send[ind_frec,ind_amplitud,ch_send]),alpha=0.7)
ax1.plot(frec_acq,fft_acq,'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
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

    
