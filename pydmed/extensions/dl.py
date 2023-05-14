

'''
Extensions related to PyDmed's dataloader.

'''

import math
import numpy as np
from abc import ABC, abstractmethod
import random
import time
import openslide
import copy
import torchvision
import pydmed
import pydmed.lightdl
from pydmed import *
from pydmed.lightdl import *



class LabelBalancedDL(pydmed.lightdl.LightDL):
    '''
    inherits from the LightDL
    purpose is to ensure that the returned small chunks are balanced in terms of label frequency
    '''
    '''
    This dataloader makes sure that the returned smallchunks are have a balanced label
            frequency.
    Inputs.
        - func_getlabel_of_patient: a function that takes in a `Patient` and returns
            the corresponding label. The returned smallchunks are balanced in terms of
            this label.
        - ... other arguments, same as LightDL, 
        https://github.com/amirakbarnejad/PyDmed/blob/8575ea991fe464b6e451d1a3381f9026581153da/pydmed/lightdl.py#L292
    '''
    def __init__(self, func_getlabel_of_patient, *args, **kwargs):
        '''
        attributes:
        possible_labels - a list of all possible classes in the dataset
        dict_label_to_listpatients- a dictionary that maps each class to a list of patients in the dataset with that label. To create dict_label_to_listpatients, the code loops through all the patients in the dataset and applies func_getlabel_of_patient to get their label. It then adds the patient to the corresponding list in dict_label_to_listpatients.
        '''
        '''
        Inputs.
        - func_getlabel_of_patient: a function that takes in a `Patient` and returns
            the corresponding label. The returned smallchunks are balanced in terms of
            this label.
        - ... other arguments, same as LightDL, 
        https://github.com/amirakbarnejad/PyDmed/blob/8575ea991fe464b6e451d1a3381f9026581153da/pydmed/lightdl.py#L29
        '''
        super(LabelBalancedDL, self).__init__(*args, **kwargs)
        #grab privates
        self.func_getlabel_of_patient = func_getlabel_of_patient
        #make separate lists for different classes ====
        possible_labels = list(
                  set(
                   [self.func_getlabel_of_patient(patient)\
                    for patient in self.dataset.list_patients]
                  )     
                )
        dict_label_to_listpatients = {label:[] for label in possible_labels}
        for patient in self.dataset.list_patients:
            label_of_patient = self.func_getlabel_of_patient(patient)
            dict_label_to_listpatients[label_of_patient].append(patient)
        self.possible_labels = possible_labels
        self.dict_label_to_listpatients = dict_label_to_listpatients
    
    def initial_schedule(self):
        '''
        initialize the schedule for loading the big chunks of data
        splits the big chunks of data into smaller chunks, which will be processed in parallel by different threads
        first calculates the average number of big chunks that each label should have, based on the total number of big chunks and the number of possible labels. Then, it creates a list list_binsize with the number of big chunks that each label should have. If the number of big chunks is not evenly divisible by the number of possible labels, then the remaining chunks are added one by one to the first num_toadd labels in the list.
        creates a list of patients toret_list_patients by randomly selecting patients from the different classes in proportion to the number of big chunks that each class should have. This list is returned as the initial schedule for loading big chunks.
        '''
        # ~ print("override initsched called.")
        #split numbigchunks to lists of almost equal length ======
        avg_inbin = self.const_global_info["num_bigchunkloaders"]/len(self.possible_labels)
        avg_inbin = math.floor(avg_inbin)
        list_binsize = [avg_inbin for label in self.possible_labels]
        num_toadd = self.const_global_info["num_bigchunkloaders"]-\
                    avg_inbin*len(self.possible_labels)
        for n in range(num_toadd):
            list_binsize[n] += 1
        #randomly sample patients from different classes =====
        toret_list_patients = []
        for idx_bin, size_bin in enumerate(list_binsize):
            label = self.possible_labels[idx_bin]
            toret_list_patients = toret_list_patients +\
                            random.choices(self.dict_label_to_listpatients[label], k=size_bin)
        return toret_list_patients
    
    def schedule(self):
        '''
        override of the schedule in LightDL
        selects a patient to remove based on a random choice from the list of currently loaded patients. Then it chooses a patient to load based on which class (i.e., label) has the smallest number of currently loaded patients. The minority label is determined using the multiminority function from pydmed.utils.minimath, which returns the label(s) that have the smallest number of loaded patients.
        selects candidate patients to load from the minority class using toadd_candidates. It calculates a weight for each candidate patient based on the number of times it has been scheduled to load. It gives a higher priority to patients that have not been loaded yet. Finally, it chooses a patient to load using these weights, and returns the selected patient to load and the selected patient to remove.
        '''
        # ~ print("override sched called.")
        #get initial fields ==============================
        list_loadedpatients = self.get_list_loadedpatients()
        list_waitingpatients = self.get_list_waitingpatients()
        schedcount_of_waitingpatients = [self.get_schedcount_of(patient)\
                                         for patient in list_waitingpatients]
        #patient_toremove is selected randomly =======================
        patient_toremove = random.choice(list_loadedpatients)
        #choose the patient to load ================
        minority_label = pydmed.utils.minimath.multiminority(
                [self.func_getlabel_of_patient(patient) for patient in list_loadedpatients]
                        )
        toadd_candidates = self.dict_label_to_listpatients[minority_label]
        weights = 1.0/(1.0+np.array(
                            [self.get_schedcount_of(patient) for patient in toadd_candidates]
                        ))
        weights[weights==1.0] =10000000.0 #if the case is not loaded sofar, give it a high prio
        patient_toload = random.choices(toadd_candidates,\
                                        weights = weights, k=1)[0]
        return patient_toremove, patient_toload







