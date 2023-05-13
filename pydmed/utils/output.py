
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import multiprocessing as mp
import csv
import time



class StreamWriter(mp.Process):
    def __init__(self, list_patients=None, rootpath=None, fname_tosave=None, 
                 waiting_time_before_flush = 3):
        '''
        StreamWriter works in two modes:
            1) one file is created for the whole dataset. In this case, 
                only `fname_tosave` is used and the argument `rootpath` must be None.
            2) one file is created for each `Patient` in the directory `rootpath`.
               In this case, `fname_tosave` must be None.
        Inputs:
            - waiting_time_before_flush: before flushing the contents, it should 
                sleep a few seconds. Default is 3 seconds.
        '''
        super(StreamWriter, self).__init__()
        if(isinstance(rootpath, str) and isinstance(fname_tosave, str)):
            if((rootpath!=None) and (fname_tosave!=None)):
                exception_msg = "One of the arguments `rootpath` and `fname_tosave`"+\
                                " must be set to None. For details, please refer to"+\
                                " `StreamWriter` documentation"
                raise Exception(exception_msg)
        if(isinstance(fname_tosave, str)):
            if(fname_tosave != None):
                self.op_mode = 1
        if(isinstance(rootpath, str)):
            if(rootpath != None):
                self.op_mode = 2
        if(hasattr(self, "op_mode") == False):
            exception_msg = "Exactly one of the arguments `rootpath` or `fname_tosave`"+\
                    " must be set to a string."+\
                    " For details, please refer to"+\
                     " `StreamWriter` documentation"
            raise Exception(exception_msg)
        if(self.op_mode == 1):
            if(fname_tosave.endswith(".csv") == False):
                raise Exception("The argument `fname_tosave` must end with .csv."+\
                                "Because only .csv format is supported.")
        if(self.op_mode == 2):
            if(len(list(os.listdir(rootpath))) > 0):
                print(list(os.listdir(rootpath)))
                raise Exception("The folder {} \n is not empty.".format(rootpath)+\
                        " Delete its files before continuing.")
        #grab privates ================
        self.list_patients = list_patients
        self.rootpath = rootpath
        self.fname_tosave = fname_tosave
        self.waiting_time_before_flush = waiting_time_before_flush
        #make/open csv file(s) =======================
        if(self.op_mode == 1):
            self.list_files = [open(fname_tosave, mode='a+')]
            self.list_writers = [csv.writer(f, delimiter=',',\
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL
                                    ) for f in self.list_files]
        elif(self.op_mode == 2):
            self.list_files = [open(os.path.join(rootpath,\
                                    "patient_{}.csv".format(patient.int_uniqueid))
                                   , mode='a+') for patient in list_patients]
            self.list_writers = [csv.writer(f, delimiter=',',\
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL
                                    ) for f in self.list_files]
        #make mp stuff ========
        '''
        two mp.Queue objects:
        1. queue_towrite is used to pass data to the writing process.
        2. queue_signal_end is used to signal the end of writing process.
        3. flag_closecalled is used to indicate whether the close method has been called or not. Once close is called, writing would be disabled.
        '''
        self.queue_towrite = mp.Queue() #there is one queue in both operating modes.
        self.queue_signal_end = mp.Queue() #this queue not empty means "close"
        self.flag_closecalled = False #once close is called, writing would be disabled.
    
    def flush_and_close(self):
        '''
        1. flushes any remaining data in the queue_towrite
        2. sleeps for the waiting_time_before_flush period
        3. adds a "stop" signal to the queue_signal_end to signal that writing is complete and the process can be terminated
        When the StreamWriter process receives the "stop" signal from queue_signal_end, it will stop writing to the file and terminate.
        '''
        self.flag_closecalled = True
        time.sleep(self.waiting_time_before_flush)
        self.queue_signal_end.put_nowait("stop")
    
    
    def run(self):
        '''
        executed when an instance of the class is started as a separate process. The method runs an infinite loop where it checks if the queue_signal_end has any item in it. 
        If the queue has any item in it, it means that the flush_and_close method has been called, so the loop should be terminated, and all open files should be flushed and closed. 
        If the queue does not have any item in it, the method executes the _wrt_patrol method, which is responsible for writing any data added to the queue_towrite by the main program to the appropriate CSV file.
        '''
        while True:
            if(self.queue_signal_end.qsize()>0):
                #execute flush_and_close ==========
                self.flag_closecalled = True
                self._wrt_onclose()
                for f in self.list_files:
                    f.flush()
                    f.close()
                break
            else:
                #patrol the queue ==========
                self._wrt_patrol()
    
    
    def write(self, patient, str_towrite):
        '''
        writes a string to file(s) in a separate process. Depending on the op_mode attribute, it either writes to a single CSV file or writes to individual files for each patient.
        If the close method has been called previously, it raises a warning message and does not write anything. 
        Else, it puts a dictionary containing the patient and string to be written into a multiprocessing queue, which is checked in the _wrt_patrol method and used to write the string to file(s) in the separate process.
        '''
        '''
        Writes to file (s).
        Inputs.
            - patient: an instance of `Patient`. This argument is ignored
                    when operating in mode 1.
            - str_towrite: the string to be written to file.
        '''
        if(self.flag_closecalled == False):
            self.queue_towrite.put_nowait({"patient": patient, "str_towrite":str_towrite})
        else:
            print("`StreamWriter` cannot `write` after calling the `close` function.")
    
    def _wrt_patrol(self):
        '''
        private method 
        checks if there are elements in the queue_towrite and writes the first one to the corresponding file(s) if there are.
        If the op_mode is 1, it writes the str_towrite to the first file in list_files.
        If the op_mode is 2, it extracts the patient and str_towrite from the first element in the queue, checks that patient is in list_patients, finds the index of the patient in the list, and writes str_towrite to the file corresponding to that patient index in list_files.
        If there are no elements in the queue, the method does nothing.
        '''
        '''
        Pops/writes one element from the queue
        '''
        if(self.queue_towrite.qsize() > 0):
            try:
                poped_elem = self.queue_towrite.get_nowait()
                
                if(self.op_mode == 1):
                    self.list_files[0].write(poped_elem["str_towrite"])
                elif(self.op_mode == 2):
                    patient, str_towrite = poped_elem["patient"], poped_elem["str_towrite"]
                    assert(patient in self.list_patients)
                    idx_patient = self.list_patients.index(patient)
                    self.list_files[idx_patient].write(str_towrite)
            except Exception as e:
                pass
                #print("\n\n\n\n*************")
                #print(str(e))
            
        
    def _wrt_onclose(self):
        '''
        responsible for popping and writing all remaining elements from the queue_towrite queue when the close function is called. 
        It first retrieves the size of the queue and iterates over all the elements using a for loop. It then pops each element and writes it to the corresponding file based on the operation mode. 
            1. If op_mode is 1, it writes to the first file in list_files
            2. If op_mode is 2, it looks up the index of the corresponding patient in list_patients and writes to the corresponding file in list_files. 
        Any exception that occurs during the popping and writing of elements is caught and ignored.
        '''
        '''
        Pops/writes all elements of the queue.
        '''
        qsize = self.queue_towrite.qsize()
        if(qsize > 0):
            for idx_elem in range(qsize):
                try:
                    poped_elem = self.queue_towrite.get_nowait()
                    if(self.op_mode == 1):
                        self.list_files[0].write(poped_elem["str_towrite"])
                    elif(self.op_mode == 2):
                        patient, str_towrite = poped_elem["patient"], poped_elem["str_towrite"]
                        assert(patient in self.list_patients)
                        idx_patient = self.list_patients.index(patient)
                        self.list_files[idx_patient].write(str_towrite)
                except:
                    pass
        
        
        
