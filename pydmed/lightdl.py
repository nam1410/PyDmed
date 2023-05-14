


'''
General TODO:s
    - replace os.nice and taskset with cross-platform counterparts.
'''


import numpy as np
import matplotlib.pyplot as plt
import os, sys
import psutil
from pathlib import Path
import re
import time
import random
import multiprocessing as mp
import subprocess
from abc import ABC, abstractmethod
import openslide
import torch
import torchvision
import torchvision.models as models
from multiprocessing import Process, Queue
import pydmed.utils.multiproc
from pydmed.utils.multiproc import *

'''
Global enumerations. 
As enum is only supported for python 3.4+, pydmed uses some global variables.
'''
PYDMEDRESERVED_HALTDL = "PYDMEDRESERVED_HALTDL"
PYDMEDRESERVED_DLRETURNEDLASTINSTANCE = "PYDMEDRESERVED_DL_RETURNED_LAST_INSTANCE"

def get_default_constglobinf():
    '''
    returns a dictionary with default values for various parameters, such as the number of big chunk loaders, maximum length of the small chunk and light DL queues, rescheduling interval, and core assignments
    '''
    toret = {
        "num_bigchunkloaders":10,
        "maxlength_queue_smallchunk":100,
        "maxlength_queue_lightdl":10000,
        "interval_resched": 10,
        "core-assignment":{"lightdl":None,
                           "smallchunkloaders":None,
                           "bigchunkloaders":None}
    }
    return toret


class BigChunk:
    def __init__(self, data, dict_info_of_bigchunk, patient):
        '''
        represents a large chunk of data
        data-  the data part of the big chunk, represented as a numpy ndarray.
        dict_info_of_bigchunk- a dictionary containing additional information about the big chunk, such as its position within the patient's image
        patient- an instance of the utils.data.Patient class, which represents a patient's data
        '''
        '''
        Implementation of a "Big Data Chunk".
        Inputs:
            - data: the data part (e.g., a patch of size 1000x1000), an instance of numpy.ndarray.
            - dict_info_of_bigchunk: a dictionary containing information about the bigchunk. 
                                  It may include, e.g., the (left,top) position of the big patch, etc.
            - patient: an instance of `utils.data.Patient`.
        '''
        #grab privates
        self.data = data
        self.dict_info_of_bigchunk = dict_info_of_bigchunk
        self.patient = patient

class SmallChunk:
    '''
    represents a small chunk of data
    data-  the data part of the small chunk, represented as a numpy ndarray
    dict_info_of_smallchunk - a dictionary containing additional information about the small chunk
    dict_info_of_bigchunk- a dictionary containing additional information about the big chunk, such as its position within the patient's image
    patient- an instance of the utils.data.Patient class, which represents a patient's data
    '''
    def __init__(self, data, dict_info_of_smallchunk, dict_info_of_bigchunk, patient):
        '''
        Implementation of a "Big Data Chunk".
        Inputs:
            - data: the data part (e.g., a patch of size 1000x1000), an instance of numpy.ndarray.
            - dict_info_of_smallchunk: a dictionary containing information about the bigchunk. 
                                  It may include, e.g., the (left,top) position of the small patch, etc.
            - dict_info_of_bigchunk: a dictionary containing information about the bigchunk. 
                                  It may include, e.g., the (left,top) position of the big patch, etc.
            - patient: an instance of `utils.data.Patient`.
        '''
        #grab privates
        self.data = data
        self.dict_info_of_smallchunk = dict_info_of_smallchunk
        self.dict_info_of_bigchunk = dict_info_of_bigchunk
        self.patient = patient


class BigChunkLoader(mp.Process):
    '''
    inherits from mp.Process
    represents a process that loads big chunks of data for a patient. Its main purpose is to extract a big chunk of data for a patient and put it into a queue for further processing
    '''
    def __init__(self, patient, queue_bigchunk, const_global_info, queue_logs, old_checkpoint, last_message_from_root):
        '''
        Inputs:
            - patient: an instance of `Patient`.
            - queue_bigchunk: the queue to place the extracted bigchunk,
                    an instance of multiprocessing.Queue.
            - const_global_info: global information visible by all subprocesses, 
                        a dictionary. It can contain, e.g., the lenght of the queues, 
                        waiting times. etc.
            - path_logfiles: a path where the logfiles are saved, a string.
                             If provided, you can log to the file with the funciton self.log(" some string ").
            - last_message_from_root: TODO:adddoc for last_message_from_root.
        '''
        super(BigChunkLoader, self).__init__()
        self.patient = patient
        self.queue_bigchunk = queue_bigchunk
        self.const_global_info = const_global_info
        self._queue_logs = queue_logs
        self.old_checkpoint = old_checkpoint
        self.last_message_from_root = last_message_from_root
        #make internals
        # ~ if(self.path_logfiles != None):
            # ~ self.logfile = open(self.path_logfiles + "patient_{}.txt".format(self.patient.int_uniqueid),"a")
    
    def get_checkpoint(self):
        return self.old_checkpoint
        
    def run(self):
        '''
        method executed when the process starts
        The method first sets a random seed using the current time, then assigns the process to a specific CPU core if the core-assignment dictionary in const_global_info contains a value for "bigchunkloaders".
        Next, the method calls the extract_bigchunk method to extract a bigchunk and puts it in the queue_bigchunk. 
        '''
        """Loads a bigchunk and waits for the patchcollector to enqueue the bigchunk."""
        
        #set random seed using time ====
        np.random.seed(int(time.time()))
        
        #assign the bigchunkloader to core
        if(self.const_global_info["core-assignment"]["bigchunkloaders"] != None):
            os.system("taskset -cp {} {}".format(self.const_global_info["core-assignment"]["bigchunkloaders"], os.getpid()))
            print(" taskset called for bigchunkloader")
            
        #extract a bigchunk =======
        bigchunk = self.extract_bigchunk(self.last_message_from_root)
        #place the bigchunk in the queue
        self.queue_bigchunk.put_nowait(bigchunk)
        # ~ if(self.path_logfiles != None):
            # ~ self.logfile.flush()
        
        # ~ self.logfile.close()
        #wait untill being terminated
        #TODO:make the waiting more efficient
#         while(True):
#             pass
    
    def log(self, str_input):
        '''
        logs a string to the log file by putting the string into the multiprocessing queue _queue_logs.
        '''
        '''
        Logs to the log file, i.e. the file with `fname_logfile`.
        '''
        self._queue_logs.put_nowait(str_input)
    
    @abstractmethod     
    def extract_bigchunk(self, last_message_fromroot):
        '''
         abstract method - needs to be implemented in the subclass of BigChunkLoader. 
         extracts and returns a bigchunk, where the bigchunk is a large piece of data (e.g., a patch of size 1000x1000) from a patient's data. 
         takes last_message_fromroot as input, which is the last message sent to patient using the function lightdl.send_message. The method has access to the attributes self.patient and self.const_global_info.
        '''
        '''
        Extract and return a bigchunk.
        Inputs:
            - `last_message_fromroot`: The last messsage sent to this patient. Indeed, this is the message sent by calling the function
                                       `lightdl.send_message`.
        Note that in this function you have access to `self.patient` and `self.const_global_info`.
        '''
        pass




class SmallChunkCollector(mp.Process):
    '''
    class inherits from mp.Process
    represents a process that is intended to extract and enqueue small chunks from a Patient instance. 
    process extracts small chunks of data from the Patient instance, and it enqueues them into a queue. To do this, it instantiates a BigChunkLoader instance of the type specified by the type_bigchunkloader argument and uses it to extract a big chunk. It then splits the big chunk into smaller chunks and enqueues them into the queue_smallchunks.
    The class has several internal variables such as _cached_checkpoint, _queue_status, _cached_status, and _queue_bigchunkloader_terminated that are used to cache the checkpoint, status, and big chunk loader's termination status. These variables are also used to communicate with other processes that use the SmallChunkCollector.
    '''
    def __init__(self, patient, queue_smallchunks, const_global_info,\
                type_bigchunkloader, queue_logs, old_checkpoint, queue_checkpoint, last_message_from_root):
        '''
        Inputs:
            - patient: an instance of `Patient`.
            - queue_smallchunks: the queue to place the extracted small chunks,
                    an instance of multiprocessing.Queue.
            - const_global_info: global information visible by all subprocesses, 
                        a dictionary. It can contain, e.g., the lenght of the queues, 
                        waiting times. etc.
            - type_bigchunkloader: the type (i.e. Class) of bigchunkloader to instantiate from,
                        a subclass of BigChunkLoader.
            - queue_logs: the queue in which logs are going to placed.
            - last_message_from_root: TODO:adddoc for last_message_from_root.
        '''
        super(SmallChunkCollector, self).__init__()
        self.patient = patient
        self.queue_smallchunks = queue_smallchunks
        self.const_global_info = const_global_info
        self.type_bigchunkloader = type_bigchunkloader
        self._queue_logs = queue_logs
        self.old_checkpoint = old_checkpoint
        self.queue_checkpoint = queue_checkpoint
        self.last_message_from_root = last_message_from_root
        #make internals =====
        self._cached_checkpoint = "TODO:packagename reserverd: empty cache"
        self._queue_status = mp.Queue()
        self._cached_status = "TODO:packagename reserverd: empty cache"
        self._queue_bigchunkloader_terminated = mp.Queue()
        
    def log(self, str_input):
        '''
        log a message to the queue _queue_logs
        '''
        '''
        Logs to the log file, i.e. the file with `fname_logfile`.
        '''
        self._queue_logs.put_nowait(str_input)
    
    def set_status(self, status):
        '''
        set the status of the SmallChunkCollector - any pickleable object that can be set by the caller.
        '''
        '''
        This function sets the status of smallchunkloader.
        The dataloader can read the status of the SmallChunkCollector by calling `SmallChunkCollector.get_status`.
        Input:
            - status: can be any pickleable object, e.g., a string, a dictionary, etc.
        '''
        self._queue_status.put_nowait(status)
    
    def get_status(self):
        '''
        If there is no new status available, it returns the last cached status.
        '''
        '''
        This function returns the last status of `SmallChunkCollector`, which is previously set
        by calling the function `SmallChunkCollector.set_status`.
        '''
        # ~ print("get status called")
        qsize_status = self._queue_status.qsize()
        # ~ print("qsize_status = {}".format(qsize_status))
        if(qsize_status > 0):
            last_status = None
            for count in range(qsize_status):
                try:
                    last_status = self._queue_status.get_nowait()
                except Exception as e:
                    pass
                    # ~ print("an excection occured in get_status function")
                    # ~ print(str(e))
                    # ~ print("   get status returned")
            # ~ print("   get status returned")
            self._cached_status = last_status
            return last_status
        else:
            # ~ print("return cached status")
            return self._cached_status
    
    def get_checkpoint(self):
        '''
        used to get the checkpoint that is cached in the instance. 
        If the cache is empty, it returns the old_checkpoint.
        '''
        if(isinstance(self._cached_checkpoint, str)):
            if(self._cached_checkpoint == "TODO:packagename reserverd: empty cache"):
                return self.old_checkpoint
            else:
                return self._cached_checkpoint
        else:
            return self._cached_checkpoint
    
    def set_checkpoint(self, checkpoint):
        '''
        used to set the checkpoint of the SmallChunkCollector.
        The checkpoint is placed in the queue queue_checkpoint and is also cached in the instance.
        '''
        self.queue_checkpoint.put_nowait(checkpoint)
        self._cached_checkpoint = checkpoint
    
    def run(self):
        '''
        called when the SmallChunkCollector instance is started as a separate process using the start() method.
        first sets the random seed using the current time and assigns the process to a specific core if specified in the configuration. 
        It then creates a subprocess to load a bigchunk and waits for it to finish loading before starting to extract small chunks from it. The bigchunk is obtained from a multiprocessing queue. Once obtained, the method enters a loop where it repeatedly extracts small chunks from the bigchunk and places them in the queue_smallchunks queue until the queue reaches its maximum size - details = extract_smallchunk method is called to extract each small chunk from the bigchunk. The method logs its progress and status updates to the log file and updates the status of the SmallChunkCollector. Once a small chunk is extracted, it is placed in the queue_smallchunks queue for the data loader process to consume. If the extract_smallchunk method returns None, the small chunk is not added to the queue. The loop continues until the process is terminated or interrupted.
        '''
        '''
        Loads a bigchunk, waits for the bigchunk to be loaded, and then makes calls to
        self.extract_smallchunk.
        '''
        #set random seed using time ====
        np.random.seed(int(time.time()))
        
        # ~ os.nice(1000) #TODO:make tunable
        # ~ print(" pid of smallchunkcollector: {}".format(os.getpid()))
        #print("in smallchunkloader, pid is: {}".format(os.getpid()))
        if(self.const_global_info["core-assignment"]["smallchunkloaders"] != None):
            # ~ p = psutil.Process()
            # ~ idx_cores = [int(u) for u in self.const_global_info["core-assignment"]["smallchunkloaders"].split(",")]
            # ~ p.cpu_affinity(idx_cores)
            # ~ print("in smallchunkloaders, idx_cores={}".format(idx_cores))
            os.system("taskset -cp {} {}".format(self.const_global_info["core-assignment"]["smallchunkloaders"], os.getpid()))
            print(" taskset called for smallchunkloader")
        #print("    subprocess pinded to cores")
        
        # ~ print("reached here 1")
        #Load a bigchunk in a subprocess
        queue_bc = mp.Queue()
        # ~ print("reached here 2")
        proc_bcloader = self.type_bigchunkloader(self.patient, queue_bc,\
                                                 self.const_global_info, self._queue_logs, self.old_checkpoint, self.last_message_from_root)
        # ~ print("reached here 3")
        proc_bcloader.start()
        # ~ print("reached here 4")
        while(queue_bc.empty()):
            pass
            #print("Smallchunk collector is waiting to collect the bigchunk.")
            #wait for the sploader to load the superpatch
            #TODO:make the waiting more efficient
        #collect the bigchunk and start extracting patches from it
        # ~ print("reached here 5")
        self._queue_bigchunkloader_terminated.put_nowait("Finished loading a bigchunk")
        
        bigchunk = queue_bc.get()
        # ~ print("reached here 6")
        call_count = 0
        while(True):
            if(self.queue_smallchunks.qsize() < self.const_global_info["maxlength_queue_smallchunk"]):
                # ~ print(" ----------------- reached here 7")
                #print("  smallchunkcollector saw emtpy place in queue.")
                smallchunk = self.extract_smallchunk(call_count, bigchunk, self.last_message_from_root)
                call_count += 1
                #print("... and extracted a smallchunk.")
                if(isinstance(smallchunk, np.ndarray) == False):
                    if(smallchunk == None):
                        pass
                    else:
                        self.queue_smallchunks.put_nowait(smallchunk)
                else:
                    self.queue_smallchunks.put_nowait(smallchunk)
                #print("     placed a smallchunk in queue.")
        
    def get_flag_bigchunkloader_terminated(self):
        '''
        checks if the _queue_bigchunkloader_terminated attribute has any items in it. If it does, it returns True, indicating that the big chunk loader has finished loading a big chunk. Otherwise, it returns False.
        '''
        try:
            if(self._queue_bigchunkloader_terminated.qsize() > 0):
                return True
            else:
                return False
        except:
            print("Exception occured in get_flag_bigchunkloader_terminated")
            return False
    
    @abstractmethod     
    def extract_smallchunk(self, call_count, bigchunk, last_message_fromroot):
        '''
        abstract method - needs to be implemented by a subclass of SmallChunkCollector
        '''
        '''
        Extract and return a smallchunk. Please note that in this function you have access to 
        self.bigchunk, self.patient, self.const_global_info.
        Inputs:
            - `call_count`: an integer. While the `SmallChunkCollector` is collecting `SmallChunk`s, 
                             the function `extract_smallchunk` is called several times.
                             The argument `count_calls` is the number of times the `extract_smallchunk` is called
                             since the `SmallChunkCollector` (and its child `BigChunkLoader`) has started working. 
            - bigchunk: the extracted bigchunk.
            - `last_message_fromroot`: The last messsage sent to this patient. Indeed, this is the message sent by calling the function
                                       `lightdl.send_message`.
        Output:
            - smallchunk: has to be either an instance of `SmallChunk` or None.
                          Returning None means the `SmallChunkCollector` is no longer willing to extract `SmallChunk`s for, e.g.,
                          it has sufficiently explored the patient's records. 
        '''
        pass


class LightDL(mp.Process):
    def __init__(self, dataset, type_bigchunkloader, type_smallchunkcollector,\
                 const_global_info, batch_size, tfms, flag_grabqueue_onunsched=True, collate_func=None, fname_logfile=None,
                 flag_enable_sendgetmessage = True, flag_enable_setgetcheckpoint = True):
        '''
        inherits from mp.Process
        initializes the instance variables inputs, the transformation function (tfms). The class also has flags to enable/disable sending and getting messages (flag_enable_sendgetmessage) and setting/getting checkpoints (flag_enable_setgetcheckpoint).
        active_subprocesses - keeps track of currently active processes
        dict_patient_to_schedcount - dictionary that keeps track of the number of times SmallChunkCollector has been scheduled for each patient 
        list_smallchunksforvis - a list of small chunks for visualization purposes
        _queue_logs - a queue that stores log messages
        If the flag flag_enable_setgetcheckpoint is True, dict_patient_to_checkpoint is a dictionary that keeps track of the last checkpoint for each patient, and _dict_patient_to_queueckpoint is a dictionary that maps each patient to a queue for storing checkpoints.
        If the flag flag_enable_sendgetmessage is True, _queue_messages_to_subprocs is a dictionary that maps each patient to a queue for storing messages to be sent to subprocesses.
        '''
        '''
        Inputs:
            - dataset: an instance of pydmed.utils.Dataset.
            - type_bigchunkloader: the type (Class) of bigchunkloader, 
                a subplcass of BigChunkLoader.
            - type_smallchunkcollector: the type (Class) of smallchunkloader, 
                a subplcass of SmallChunkLoader.
            - const_global_info: global information visible by all subprocesses, 
                        a dictionary. It can contain, e.g., the lenght of the queues, 
                        waiting times. etc.
            - batch_size: the size of each batch, an integer.
            - fname_logfile: the name of the file to which `.log(str)` function will write.
        '''
        #grab privates ====
        super(LightDL, self).__init__()
        self.dataset = dataset
        self.type_bigchunkloader = type_bigchunkloader
        self.type_smallchunkcollector = type_smallchunkcollector
        self.const_global_info = const_global_info
        self.queue_lightdl = mp.Queue()
        self.batch_size = batch_size
        self.flag_grabqueue_onunsched = flag_grabqueue_onunsched
        self.fname_logfile = fname_logfile
        if(collate_func == None):
            self.collate_func = LightDL.default_collate
        else:
            self.collate_func = collate_func
        self.tfms = tfms
        self.flag_enable_setgetcheckpoint = flag_enable_setgetcheckpoint
        self.flag_enable_sendgetmessage = flag_enable_sendgetmessage
        #make internals ====
        self.active_subprocesses = set() #set of currently active processes
        self._queue_pid_of_lightdl = mp.Queue()
        self._queue_message_lightdlfinished = mp.Queue()
        self.dict_patient_to_schedcount = {patient:0 for patient in self.dataset.list_patients}
        #self.list_poped_entities = []
        self.list_smallchunksforvis = [] #smallchunks without data and only for visualization.
        if(flag_enable_setgetcheckpoint == True):
            self.dict_patient_to_checkpoint = {patient:None for patient in self.dataset.list_patients}
            self._dict_patient_to_queueckpoint = {patient:mp.Queue() for patient in self.dataset.list_patients}
        else:
            self.dict_patient_to_checkpoint = None
            self._dict_patient_to_queueckpoint = None
        if(flag_enable_sendgetmessage == True):
            self._queue_messages_to_subprocs = {patient:mp.Queue() for patient in self.dataset.list_patients}
        else:
            self._queue_messages_to_subprocs = None
        self._queue_logs = mp.Queue()
        if(self.fname_logfile != None):
            self.logfile = open(self.fname_logfile, "a")
    
    def flush_log(self):
        '''
        used to flush the contents of the log queue to the log file
        If a fname_logfile was provided during initialization, then the log file will be opened and the log messages in the queue will be written to the file. 
        If no fname_logfile was provided, then the method does nothing.
        '''
        if(self.fname_logfile == None):
            return #DO nothing
        size_queue_log = self._queue_logs.qsize()
        for count in range(size_queue_log):
            try:
                elem = self._queue_logs.get_nowait()
                self.logfile.write(elem)
            except:
                pass 
        self.logfile.flush()
        self.logfile.close()
            
    def log(self, str_input):
        '''
        used to add a log message to the log queue
        '''
        '''
        Logs to the log file, i.e. the file with `fname_logfile`.
        '''
        self._queue_logs.put_nowait(str_input)
        
    def send_message(self, patient, message):
        '''
        used to send a message to a subprocess corresponding to a patient. 
        The message will be added to a queue for the patient, and the message will be received by the subprocess when it is scheduled to run. 
        The message can be accessed in the SmallChunkCollector.extract_smallchunk() and BigChunkLoader.extract_bigchunk() methods.
        '''
        '''
        Sends message to a subprocess corresponding to the patient. 
        Once the subproc is schedulled to run, it will recieve the last sent message. 
        You can access the last recieved message in `SmallChunkCollector.extract_smallchunk` and `BigChunkLoader.extract_bigchunk`. 
        '''
        self._queue_messages_to_subprocs[patient].put_nowait(message)
        
    
    @staticmethod
    def _terminaterecursively(pid):
        '''
        static method - recursively terminates all child processes of a given process ID (pid) and the parent process itself
        terminates the subprocesses that were created by the LightDataLoader instance
        '''
        parent = psutil.Process(pid)#TODO:copyright, https://www.reddit.com/r/learnpython/comments/7vwyez/how_to_kill_child_processes_when_using/
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except:
                pass
            #print(" killed subprocess {}".format(child))
        #if including_parent:
        try:
            parent.kill()
        except:
            pass
    
    def pause_loading(self):
        '''
        used to pause the data loading process by killing the LightDataLoader subprocess
        retrieves the PID of the LightDataLoader process from the queue and then terminates it using the _terminaterecursively method
        also flushes the log file using the flush_log method
        '''
        lightdl_pid = self._queue_pid_of_lightdl.get()
        self.flush_log()
        parent = psutil.Process(lightdl_pid)#TODO:copyright, https://www.reddit.com/r/learnpython/comments/7vwyez/how_to_kill_child_processes_when_using/
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except:
                pass
            #print(" killed subprocess {}".format(child))
        #if including_parent:
        try:
            parent.kill()
        except:
            pass
    
    def is_dl_running(self):
        '''
        used to check if the data loading process is still running
        checks the size of the self._queue_message_lightdlfinished queue. 
        If there are any messages in the queue, it means that the data loading process has finished and the method returns False
        If the queue is empty, it means that the data loading process is still running and the method returns True
        '''
        '''
        If the DL is still working, returns True.
        Otherwise returns False.
        Warning: executing this function may take alot of time.
                Avoid making frequent calls to this function.
        '''
        #try to get the qsize of finish message queue.
        try:
            qsize_finishmessage = self._queue_message_lightdlfinished.qsize()
        except:
            return True
        
        if(qsize_finishmessage > 0):
            return False
        else:
            return True
        # ~ p = subprocess.Popen(['date']) #date command exists in both linux and windows.
        # ~ poll = p.poll()
        # ~ if poll is None:
            # ~ return True
        # ~ else:
            # ~ return False
        
    def visualize(self, func_visualize_one_patient):
        '''
        visualizes the collected instances by LightDL by separating the collected small chunks based on patients
        calls the func_visualize_one_patient function for each patient
        func_visualize_one_patient - passed in as an argument to the visualize method, and it should visualize a specific patient given all small chunks collected for that patient
        '''
        '''
            When visualizing the collected instances by lightdl, you should call `visualize` function.
            You should pass in the function `func_visualize_one_patient` that works as follows:        
                Given all smallchunks collected for a specific patient, this function
                should visualize the patient. 
                Inputs:
                    - patient: the patient under considerations, an instance of `utils.data.Patient`.
                    - list_smallchunks: the list of all collected small chunks for the patient,
                        a list whose elements are an instance of `lightdl.SmallChunk`.
        '''
        #separate collected smallchunks based on patients ======
        dict_patient_to_listsmallchunnks = {patient:[] for patient in self.dataset.list_patients}
        for smallchunk in self.list_smallchunksforvis:
            dict_patient_to_listsmallchunnks[smallchunk.patient].append(smallchunk)
        #call the visualization function ======
        for patient in dict_patient_to_listsmallchunnks.keys():
            func_visualize_one_patient(patient, dict_patient_to_listsmallchunnks[patient])
    
    @staticmethod
    def default_collate(list_smallchunks, tfms):
        '''
        static method - takes a list of small chunks and applies any transformations specified by tfms to the data in the small chunks. 
        It then stacks the data from all small chunks into a tensor x and returns x, along with a list of patients and a list of small chunks. 
        The data attribute of each small chunk is set to "None to avoid memory leak" to prevent memory leaks.
        '''
        list_data = [smallchunks.data for smallchunks in list_smallchunks]
        if(tfms != None):
            for n in range(len(list_data)): #apply transforms
                list_data[n] = tfms(list_data[n])
        x = torch.stack(list_data, axis=0)
        list_patients = [smallchunks.patient for smallchunks in list_smallchunks]
        list_smallchunks = [smallchunk for smallchunk in list_smallchunks]
        for smallchunk in list_smallchunks:
            smallchunk.data = "None to avoid memory leak"
        return x, list_patients, list_smallchunks
    
    def get(self):
        '''
        responsible for retrieving a batch of data instances from the internal queue
        checks whether the data loader is still running or not. If the data loader is still running, the method tries to retrieve instances from the queue until it has collected enough instances to make up a batch of the specified size. If the queue is empty and the data loader has finished running, the method returns a flag indicating that it is the last batch to be returned. 
        If the data loader is not running, the method retrieves a single instance from the queue regardless of the batch size, until it has retrieved at least one instance. If the queue is empty and the data loader is not running, the method returns the flag indicating that it is the last batch to be returned.
        Once the instances are retrieved, the method applies the collate function specified in the constructor to convert the list of instances to a batch tensor and returns it. Finally, the method creates a new list of small chunks, similar to the input list but with the actual data replaced with the string "None to avoid memory leak", and adds these data-free small chunks to the internal list used for visualization. This is done to prevent memory leaks from accumulating during the lifetime of the LightDL object.
        '''
        #make toret values =================
        list_poped_smallchunks = []
        flag_dl_running = self.is_dl_running()
        if(flag_dl_running == True):
            while(len(list_poped_smallchunks) < self.batch_size):
                #try to get a new instance ====
                if(self.queue_lightdl.qsize()>0):
                    try:
                        smallchunk = self.queue_lightdl.get_nowait()
                        list_poped_smallchunks.append(smallchunk)
                    except:
                        pass
                #if dl_is_finished and Q is empty, exit the while loop
                if((self.is_dl_running()==False) and (self.queue_lightdl.qsize()==0)):
                    #in this case,  `get` function has no chance to collect more instances
                    
                    if(len(list_poped_smallchunks) == 0):
                        #if the list is empty, it means no more instances are to be collected.
                        return PYDMEDRESERVED_DLRETURNEDLASTINSTANCE
                    else:
                        #some instances are collected, then break the while loop and continue as before
                        break
        elif(flag_dl_running == False):
            #in this case, `get` will return the Q instances one-by-one regardless of the the `batch_size`.
            if(self.queue_lightdl.qsize() > 0):
                while(len(list_poped_smallchunks) < 1):
                    #try to get a new instance ====
                    if(self.queue_lightdl.qsize()>0):
                        try:
                            smallchunk = self.queue_lightdl.get_nowait()
                            list_poped_smallchunks.append(smallchunk)
                        except:
                            pass
            else:
                #in this case, dl is finished and Q is empty.
                return PYDMEDRESERVED_DLRETURNEDLASTINSTANCE
                        
                
        # ~ print("get: reached here 2")
        returnvalue_of_collatefunc = self.collate_func(list_poped_smallchunks, self.tfms)
        # ~ print("get: reached here 3")
        #grab visualization info ============
        for smallchunk in list_poped_smallchunks:
            smallchunk_datafree = SmallChunk(data = "None to avoid memory leak",\
                                             dict_info_of_smallchunk = smallchunk.dict_info_of_smallchunk,\
                                             dict_info_of_bigchunk = smallchunk.dict_info_of_bigchunk,\
                                             patient = smallchunk.patient)
            self.list_smallchunksforvis.append(smallchunk_datafree)
        # ~ toret_list_smallchunks = [smallchunk for smallchunk in list_poped_smallchunks]
        # ~ for smallchunk in toret_list_smallchunks:
            # ~ smallchunk.data = "None to avoid memory leak"
        # ~ print("get: reached here 4")
        return returnvalue_of_collatefunc #batch_smallchunks, batch_patients, toret_list_smallchunks
    
    def get_list_loadedpatients(self):
        '''
        returns a list of Patient instances that are currently being loaded (one SmallChunkCollector is collecting SmallChunks from them)
        '''
        '''
        Returns the list of `Patient`s that are loaded, 
        i.e., one `SmallChunkCollector` is collecting `SmallChunk`s from them. 
        '''
        list_loadedpatients = [subproc.patient\
                               for subproc in list(self.active_subprocesses)]
        return list_loadedpatients
    
    
    def get_list_waitingpatients(self):
        '''
        returns a list of Patient instances that are not currently being loaded (no SmallChunkCollector is collecting SmallChunks from them)
        '''
        '''
        Returns the list of `Patient`s that are not loaded, 
        i.e., no `SmallChunkCollector` is collecting `SmallChunk`s from them. 
        '''
        set_running_patients = set(self.get_list_loadedpatients())
        set_waiting_patients = set(self.dataset.list_patients).difference(
                                        set_running_patients)
        return list(set_waiting_patients)
    
    
    def get_schedcount_of(self, patient):
        ''' 
        given a patient, returns the number of times the patient has been scheduled for loading in the past
        '''
        '''
        Reuturns the number of times that a specific `Patient` has
        been schedulled by scheduller.
        '''
        return self.dict_patient_to_schedcount[patient]
    
     
    def initial_schedule(self):
        '''
        used for selecting the initial BigChunks to load
        returns a list of Patient instances with a length equal to self.const_global_info["num_bigchunkloaders"]
        '''
        '''
        Used for selecting the initiail BigChunks.
        This funciton has to return,
            - `list_initial_patients`: a list containing `Patients` who are initially loaded.
                        The length of the list must be equal to `self.const_global_info["num_bigchunkloaders"]`
        '''
        #Default is to choose randomly from dataset.
        return random.choices(self.dataset.list_patients,\
                              k=self.const_global_info["num_bigchunkloaders"])
        
    def schedule(self):
        '''
        called when scheduling a new patient (loading a new BigChunk)
        returns the patient to remove (an instance of utils.data.Patient) and the patient to load (an instance of utils.data.Patient)
        main scheduling algorithm - selects a patient to remove (unload from memory) and a patient to load (load into memory)
        The algorithm selects a patient to remove randomly from the list of currently loaded patients and selects a patient to load based on the number of times it has been scheduled in the past. 
        Patients that have not been scheduled before are given a high weight to increase the chance of selecting them, while patients that have been scheduled multiple times are given a lower weight. The specific weights are calculated based on the number of times each patient has been scheduled using the get_schedcount_of() method.
        '''
        '''
        This function is called when schedulling a new patient, i.e., loading a new BigChunk.
        This function has to return:
            - patient_toremove: the patient to remove, an instance of `utils.data.Patient`.
            - patient_toload: the patient to load, an instance of `utils.data.Patient`.
        In this function, you have access to the following fields:
            - self.dict_patient_to_schedcount: given a patient, returns the number of times the patients has been schedulled in dl, a dictionary.
            - self.list_loadedpatients:
            - self.list_waitingpatients:
            - TODO: add more fields here to provide more flexibility. For instance, total time that the patient have been loaded on DL.
        '''
        #get initial fields ==============================
        list_loadedpatients = self.get_list_loadedpatients()
        list_waitingpatients = self.get_list_waitingpatients()
        waitingpatients_schedcount = [self.get_schedcount_of(patient)\
                                      for patient in list_waitingpatients]
        
        #patient_toremove is selected randomly =======================
        patient_toremove = random.choice(list_loadedpatients)
        
        #when choosing a patient to load, give huge weight to the instances which are not schedulled so far.
        weights = 1.0/(1.0+np.array(waitingpatients_schedcount))
        weights[weights==1.0] =10000000.0 #the places where schedcount is zero
        patient_toload = random.choices(list_waitingpatients,\
                                        weights = weights, k=1)[0]
        
        
        return patient_toremove, patient_toload
        
        
    def run(self):
        
        try:
        
            #assign to core
            if(self.const_global_info["core-assignment"]["lightdl"] != None):
                # ~ p = psutil.Process()
                # ~ idx_cores = [int(u) for u in self.const_global_info["core-assignment"]["lightdl"].split(",")]
                # ~ p.cpu_affinity(idx_cores)
                # ~ print("in lightdl, idx_cores={}".format(idx_cores))
                os.system("taskset -a -cp {} {}".format(self.const_global_info["core-assignment"]["lightdl"], os.getpid()))
                print(" taskset called for lightdl")
            #save pid of lightdl (to do recursive kill on finish)
            self._queue_pid_of_lightdl.put_nowait(os.getpid())
            #initially fill the pool of subprocesses ========
            patients_forinitialload = self.initial_schedule()
                    # ~ random.choices(self.dataset.list_patients,\
                                   # ~ k=self.const_global_info["num_bigchunkloaders"])
            print(" loading initial bigchunks, please wait ....")
            t1 = time.time()
            for i in range(len(patients_forinitialload)):
                # ~ print(" reached here 1")
                print("     bigchunk {} from {}, please wait ...\n".format(i, len(patients_forinitialload)))
                if(self._queue_messages_to_subprocs != None):
                    last_message_from_root = pydmed.utils.multiproc.poplast_from_queue(
                                    self._queue_messages_to_subprocs[patients_forinitialload[i]]
                                )
                else:
                    last_message_from_root = None
                if(self._queue_messages_to_subprocs != None):
                    old_checkpoint = self.dict_patient_to_checkpoint[patients_forinitialload[i]]
                    queue_checkpoint = self._dict_patient_to_queueckpoint[patients_forinitialload[i]]
                else:
                    old_checkpoint = None
                    queue_checkpoint = None
                subproc = self.type_smallchunkcollector(
                                        patient=patients_forinitialload[i],\
                                        queue_smallchunks=mp.Queue(),\
                                        const_global_info=self.const_global_info,\
                                        type_bigchunkloader=self.type_bigchunkloader,\
                                        queue_logs=self._queue_logs,\
                                        old_checkpoint = old_checkpoint,\
                                        queue_checkpoint = queue_checkpoint,\
                                        last_message_from_root = last_message_from_root
                                )
                # ~ print(" reached here 2")
                self.active_subprocesses.add(subproc)
                # ~ print(" reached here 3")
                subproc.start()
                # ~ print(" reached here 4")
                self.dict_patient_to_schedcount[subproc.patient] = self.dict_patient_to_schedcount[subproc.patient] + 1
                # ~ print(" reached here 5")
                while(subproc.queue_smallchunks.qsize() == 0):
                    pass
                # ~ print(" reached here 6")
                    #wait untill the bigchunk is loaded and something the smallchunkcollector collects the first smallchunk.
            t2 = time.time()
            print("The initial loading of bigchunks took {} seconds.".format(t2-t1))
            #patrol the subprocesses ======================
            time_lastresched = time.time() + 1*self.const_global_info["interval_resched"]
            while(True):
                # ~ print("============= lightdl-queue.qsize() = {} ===========".format(self.queue_lightdl.qsize()))
                #collect patches from the subporcesses ============
                while(self.queue_lightdl.qsize() >=\
                      self.const_global_info["maxlength_queue_lightdl"]):
                    pass
                    #wait until queue_lightdl becomes less heavy.
                for subproc in list(self.active_subprocesses):
                    if(subproc.queue_smallchunks.empty() == False):
                        try:
                            smallchunk = subproc.queue_smallchunks.get_nowait()
                            self.queue_lightdl.put_nowait(smallchunk)
                            #print("lightdl placed smallchunk in queue")
                            #print("LightPatcher collected patches from WSI {}"\
                            #      .format(subproc.fname_wsi))
                        except:
                            pass
                #replace a subprocesses if needed =======================
                tnow = time.time()
                time_from_lastresched = tnow - time_lastresched
                if(time_from_lastresched > self.const_global_info["interval_resched"]):
                    # ~ print("rescheduling -------- in time = {}".format(time.time()))
                    time_lastresched = time.time()
                    #setlect a ptient to add, and a subrpocess to remove
                    # ~ set_running_patients = set([subproc.patient\
                                                # ~ for subproc in list(self.active_subprocesses)])
                    # ~ list_loadedpatients = [subproc.patient\
                                           # ~ for subproc in list(self.active_subprocesses)]
                    # ~ set_waiting_patiens = set(self.dataset.list_patients).difference(
                                           # ~ set_running_patients)
                    # ~ list_waitingpatients = list(set_waiting_patiens)
                    # ~ print("reached before schedule")
                    patient_toremove, patient_toadd = self.schedule()
                    if(isinstance(patient_toremove, str)):
                        if(patient_toremove == PYDMEDRESERVED_HALTDL):
                            self._queue_message_lightdlfinished.put_nowait("DL-Finished")
                            while(True): pass #the DL process has to be terminated by the parent process. LightDL._terminaterecursively(self.pid)
                            
                    
                    # ~ print("reached after schedule")
                    flag_sched_returnednan = True
                    if(isinstance(patient_toremove, pydmed.utils.data.Patient)):
                        flag_sched_returnednan = False
                        assert(isinstance(patient_toadd, pydmed.utils.data.Patient))
                    if(flag_sched_returnednan == True):
                        # ~ print("  reached here 1")
                        pass #do not reschedule
                    else:
                        # ~ print("  reached here 2")
                        subproc_toremove = None
                        for subproc in list(self.active_subprocesses):
                            if(subproc.patient == patient_toremove):
                                # ~ print("   found subproc_toremove")
                                subproc_toremove = subproc
                                break
                        # ~ print(subproc_toremove)
                        # ~ print("  list_loadedpatients: ")
                        # ~ print(list_loadedpatients)
                        # ~ print("  patient_toremove: ")
                        # ~ print(patient_toremove)
                        
                        #add the smallchunks of subproc_toremove to lightdl.queue ===============
                        if(self.flag_grabqueue_onunsched == True):
                            size_queueof_subproctoremove = subproc_toremove.queue_smallchunks.qsize()
                            for count in range(size_queueof_subproctoremove):
                                try:
                                    smallchunk = subproc_toremove.queue_smallchunks.get_nowait()
                                    self.queue_lightdl.put_nowait(smallchunk)
                                except Exception as e:
                                    print("Warning: Some smallchunks may have lost. If not, you can safely ignore this warning.")
                                    #print(str(e))
                                    
                        
                        #grab the last checkpoint of the subproc =======================
                        if(self.flag_enable_setgetcheckpoint == True):
                            numcheckpoints_subproctoremove = subproc_toremove.queue_checkpoint.qsize()
                            last_checkpoint = None
                            for count in range(numcheckpoints_subproctoremove):
                                try:
                                    last_checkpoint = subproc_toremove.queue_checkpoint.get_nowait()
                                except Exception as e:
                                    print("an exception occured at line 652")
                                    print(str(e))
                                    
                            self.dict_patient_to_checkpoint[patient_toremove] = last_checkpoint
                        # ~ print("  reached here 3")
                        if(subproc_toremove == None):
                            print("patient_toremove not found in the list of waiting patients.")
                            _terminaterecursively(self.pid)
                        # ~ print("  reached here 4")
                        # ~ print(subproc_toremove)
                        # ~ print("  reached here 5")
                        #print("  patient toremove: {}".format(subproc_toremove.patient.name))
                        #print("  patient toadd: {}".format(patient_toadd.name))
                        #remove the subprocess
                        self.active_subprocesses.remove(subproc_toremove)
                        # ~ print("  reached here 6")
                        #print("reached here 4")
                        # ~ dlmed.utils.multiproc.terminaterecursively(subproc_toremove.pid)
                        LightDL._terminaterecursively(subproc_toremove.pid)
                        # ~ print("  reached here 7")
                        #subproc_toremove.kill()
                        #print("reached here 5")
                        #add a new process for patient_toadd
                        if(self._queue_messages_to_subprocs != None):
                            last_message_from_root = pydmed.utils.multiproc.poplast_from_queue(
                                    self._queue_messages_to_subprocs[patients_forinitialload[i]]
                                )
                        else:
                            last_message_from_root = None
                        if(self._queue_messages_to_subprocs != None):
                            old_checkpoint = self.dict_patient_to_checkpoint[patient_toadd]
                            queue_checkpoint = self._dict_patient_to_queueckpoint[patient_toadd]
                        else:
                            old_checkpoint = None
                            queue_checkpoint = None
                        new_subproc = self.type_smallchunkcollector(\
                                            patient=patient_toadd,\
                                            queue_smallchunks=mp.Queue(),\
                                            const_global_info=self.const_global_info,\
                                            type_bigchunkloader=self.type_bigchunkloader,\
                                            queue_logs=self._queue_logs,\
                                            old_checkpoint = old_checkpoint,\
                                            queue_checkpoint = queue_checkpoint,\
                                            last_message_from_root = last_message_from_root
                                        )
                        # ~ print("  reached here 8")
                        #print("reached here 6")
                        self.active_subprocesses.add(new_subproc)
                        # ~ print("  reached here 9")
                        #print("reached here 7")
                        self.dict_patient_to_schedcount[new_subproc.patient] = self.dict_patient_to_schedcount[new_subproc.patient] + 1 
                        # ~ print("  reached here 10")
                        new_subproc.start()
                        # ~ print("  reached here 11")
        except Exception as e:
            print("An exception occured in LightDL.")
            print(str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print("\n\n\n")
            _terminaterecursively(self.pid)
            
                        
                
                
            
