

import numpy as np
import os, sys
import math
from pathlib import Path
import re
import time
import random
import copy
import multiprocessing as mp
from multiprocessing import Process, Queue
import pydmed.utils.minimath


class Patient:
    def __init__(self, int_uniqueid, dict_records):
        '''
        - int_id: the unique id of the patient, an integer. 
        - dict_records: a dict of objects, where each object can be, e.g., "H&E":WSI().
        '''
        self.int_uniqueid = int_uniqueid
        self.dict_records = dict_records
    
    def __hash__(self):
        return self.int_uniqueid #TODO:adding datasetinfo to patient's uniqueid is safer. Because two datasets may have patitents with same ids. 
    
    def __repr__(self):
        return "utils.data.Patient with unique id: {}".format(self.int_uniqueid)
        
    def __eq__(self, other):
        return (self.int_uniqueid == other.int_uniqueid)
        
    def __lt__(self, other):
         return self.int_uniqueid < other.int_uniqueid
         
    def __le__(self, other):
         return self.int_uniqueid <= other.int_uniqueid
         
    def __gt__(self, other):
         return self.int_uniqueid > other.int_uniqueid
         
    def __ge__(self, other):
         return self.int_uniqueid >= other.int_uniqueid
    
class Record:
    def __init__(self, rootdir, relativedir, dict_infos):
        '''
        - rootdir: the rootdirectory of the dataset. a string, e.g., "/usr/Dataset1/"
        - relativedir: the relative dir with respect to the rootdir, a string like "1010.svs"
        - dict_infos: a dictionary containing information about the WSI, e.g., zooming "40x",
                      "20x", "10x", the date that the WSI is scanned, etc.
        '''
        # ~ rootdir = "/media/user1/9894F11594F0F69A/Ak/Data/CCI_RecurrenceScore/"
        # ~ relativedir = "Gilbert2020-03-24/10101010.svs"
        
        self.rootdir = rootdir
        self.relativedir = relativedir
        self.dict_infos = dict_infos

class Dataset:
    def __init__(self, str_dsname, list_patients):
        '''
        initializes the object with a dataset name and a list of Patient objects. it first assigns the str_dsname and list_patients arguments to the corresponding object attributes self.str_dsname and self.list_patients. Then, it loops through all elements in list_patients to check if they are instances of Patient using the isinstance function. If any element is not an instance of Patient, it raises an exception with an error message.
        '''
        '''
        - str_dsname: the name of the dataset, a string.
        - list_patients: a list whose elements are an instance of `Patient`.
        '''
        self.str_dsname = str_dsname
        self.list_patients = list_patients
        for pat in self.list_patients:
            if(isinstance(pat, Patient) == False):
                raise Exception("The second argument of Dataset.__init__, i.e., list_patients "+\
                                " contains an object which is not an instance of Patient.")
    
    @staticmethod
    def balance_by_repeat(ds, func_getlabel_of_patient, newlen_each_class=None):
        '''
        static method 
        inputs - dataset ds and a function func_getlabel_of_patient 
        returns - new dataset with balanced class labels by repeating Patients in the dataset 
        It first creates a dictionary mapping each patient in the dataset to its label using the provided func_getlabel_of_patient function. It then calculates the frequency of each label and computes the least common multiple (lcm) of the frequencies. If the input parameter newlen_each_class is not provided, the lcm is used as the number of patients for each label in the new dataset.
        Then, for each patient in the original dataset, it calculates the number of times it needs to be repeated to reach the desired number of patients for its label in the new dataset. It creates a copy of the patient with a new unique ID for each repetition and adds it to the list of patients for the new dataset.
        '''
        '''
        Repeats `Patients` in the dataset to make the labels balances.
        Inputs:
            - ds: TODO:adddoc
            - dict_patient_to_label: TODO:adddoc.
            - newlen_each_class: TODO:adddoc, if None is passed the lcm of frequencies would be used.
        '''
        #make dict_patient_to_label ====
        dict_patient_to_label = {patient:func_getlabel_of_patient(patient) for patient in ds.list_patients}
        #make dict_label_to_freq ====
        numdigits_old_idx = len(str(max([patient.int_uniqueid for patient in ds.list_patients])))
        list_labels = set([dict_patient_to_label[patient]  for patient in dict_patient_to_label.keys()])
        dict_label_to_freq = {label:0 for label in list_labels}
        for patient in dict_patient_to_label.keys():
            label = dict_patient_to_label[patient]
            dict_label_to_freq[label] = dict_label_to_freq[label] + 1 
        #if needed, set newlen_each_class to lcm of frequencies =========
        if(newlen_each_class == None):
           list_freqs = list(set([dict_label_to_freq[label]  for label in dict_label_to_freq.keys()]))
           newlen_each_class = pydmed.utils.minimath.lcm(list_freqs)
        #repeat patients to makde newds ===================
        list_patients_of_newds = []
        for patient in dict_patient_to_label.keys():
            label = dict_patient_to_label[patient]
            freq_of_label = dict_label_to_freq[label]
            repeatcount = int(newlen_each_class/freq_of_label)
            for idx_patient_copy in range(repeatcount):
                new_dict_records = copy.deepcopy(patient.dict_records)
                new_dict_records["TODO:packagename reserved, original patient"] = patient
                copy_of_patient = Patient(int_uniqueid = idx_patient_copy*(10**numdigits_old_idx)+patient.int_uniqueid,\
                                          dict_records = new_dict_records)
                list_patients_of_newds.append(copy_of_patient)
        newds = Dataset(ds.str_dsname, list_patients_of_newds)
        return newds
        
    @staticmethod
    def splits_from(dataset, percentage_partitions):
        '''
        static method 
        inputs - a Dataset instance and a list of percentage
        splits the dataset into multiple partitions based on the provided percentages  
        returns a list of Dataset instances, each corresponding to one of the partitions. The percentages indicate the size of each partition relative to the total number of patients in the original dataset. For example, if the input dataset contains 100 patients and the percentages are [60, 20, 20], then the method will return a list containing three Dataset instances: the first with 60 patients, the second with 20 patients, and the third with 20 patients. The method shuffles the patients before splitting them, to make sure that the splits are randomized.
        '''
        '''
        Splits a dataset to different datasets, e.g., [training-validation-test].
        Inputs:
            - dataset: the dataset, an instance of Dataset.
            - percentage_partitions: the percentage of the partitions, a list.
        '''
        #get constants/values
        if(np.sum(percentage_partitions) != 100):
            raise Exception("The elements of `percentage_partitions` must sum up to 100.")
        num_chunks = len(percentage_partitions)
        list_patients = dataset.list_patients
        N = len(list_patients)
        #make random splits
        random.shuffle(list_patients)
        toret_list_patients = []
        for percentage in percentage_partitions:
            picked_so_far = sum([len(u) for u in toret_list_patients])
            size_partition = math.floor(percentage*N/100.0)
            idx_begin = picked_so_far
            idx_end = min(picked_so_far+size_partition, N)
            toret_list_patients.append(list_patients[idx_begin:idx_end])
        #make datasets from list_patients
        toret = [Dataset(dataset.str_dsname, u) for u in toret_list_patients]
        return toret
    
    
    @staticmethod
    def _split_list(list_input, percentage_partitions):
        '''
        private static method 
        splits a given list into several partitions based on provided percentages. 
        inputs- list_input (the list that is to be partitioned) and percentage_partitions (list of integers that represent the percentage of elements in each partition)
        returns a list of partitions, where each partition is a list of elements from list_input
        works by iterating through each percentage in percentage_partitions. For each percentage, it calculates the number of elements that should be included in that partition based on the total length of list_input.
        It then uses this number to slice list_input and append the resulting sublist to list_toret. The method ensures that the final partition contains all the remaining elements of list_input. Finally, the method asserts that the length of all partitions added together is equal to the length of list_input and that the set of elements in all partitions is equal to the set of elements in list_input.
        '''
        list_toret = []
        for idx_percentage, percentage in enumerate(percentage_partitions):
            picked_so_far = sum([len(u) for u in list_toret])
            size_partition = math.floor(percentage* len(list_input)/100.0)
            idx_begin = picked_so_far
            idx_end = min(picked_so_far+size_partition, len(list_input))
            if(idx_percentage == (len(percentage_partitions)-1)):
                idx_end = len(list_input)
            list_toret.append(list_input[idx_begin:idx_end])
        assert(len(list_input) == sum([len(u) for u in list_toret]))
        assert(set(list_input) == set.union(*[set(u) for u in list_toret]))
        return list_toret
            
    
    @staticmethod
    def labelbalanced_splits_from(dataset, percentage_partitions,\
                                  func_getlabel_of_patient, verbose=False):
        '''
        static method 
        splits a given dataset into different datasets with a balanced distribution of labels across all partitions. 
        inputs - dataset to be split, percentage_partitions (list of percentages for each partition), and a func_getlabel_of_patient (maps a patient to its label)
        First, it creates a dictionary dict_label_to_listpatients that maps each label to a list of patients with that label in the input dataset. It then splits the list of patients corresponding to each label based on the percentage_partitions. For this, it shuffles the list of patients, uses the _split_list function to partition it into different sub-lists, and stores them in a dictionary dict_label_to_listpartitions.
        Finally, the method aggregates the splits from each class into num_chunks splits that have a balanced distribution of labels. It then creates a new dataset object for each partition and returns a list of them. The method also performs some assertions to ensure the correctness of the output. If verbose is set to True, it reports the frequency of labels in each partition.
        '''
        '''
        Splits a dataset to different datasets, e.g., [training-validation-test] such that all partitions
        have equal share from different classes.
        Inputs:
            - dataset: the dataset, an instance of Dataset.
            - percentage_partitions: the percentage of the partitions, a list.
            - func_get_function_name: the labeling function for which the split is balanced.
        '''
        #get some constants/values ====
        if(np.sum(percentage_partitions) != 100):
            raise Exception("The elements of `percentage_partitions` must sum up to 100.")
        num_chunks = len(percentage_partitions)
        list_patients = dataset.list_patients
        N = len(list_patients)
        #split patients based on label ======
        possible_labels = list(
                  set(
                    [func_getlabel_of_patient(patient)\
                     for patient in dataset.list_patients]
                  )     
                )
        # ~ print("possible_labels = {}".format(possible_labels))
        dict_label_to_listpatients = {label:[] for label in possible_labels}
        for patient in dataset.list_patients:
            label_of_patient = func_getlabel_of_patient(patient)
            dict_label_to_listpatients[label_of_patient].append(patient)
        #make splits from each class =======
        dict_label_to_listpartitions = {label:None for label in possible_labels}
        for label in possible_labels:
            patients_of_class = dict_label_to_listpatients[label]
            random.shuffle(patients_of_class)
            dict_label_to_listpartitions[label] = Dataset._split_list(patients_of_class, percentage_partitions)
        #aggregate the splits of each class ====
        list_toret = [[] for n in range(len(percentage_partitions))]
        for label in possible_labels:
            partitions_of_label = dict_label_to_listpartitions[label]
            for idx_partition in range(len(percentage_partitions)):
                list_toret[idx_partition] = list_toret[idx_partition] + partitions_of_label[idx_partition]
        toret = [Dataset(dataset.str_dsname, u) for u in list_toret]
        #make some assertions =====
        set_union_of_splits = set.union(*[set(dataset.list_patients) for dataset in toret])
        assert(set_union_of_splits == set(dataset.list_patients))
        for i in range(len(toret)):
            for j in range(len(toret)):
                if(i != j):
                    set_i = set(toret[i].list_patients)
                    set_j = set(toret[j].list_patients)
                    assert(set_i.isdisjoint(set_j))
                    assert(set_j.isdisjoint(set_i))
        #report the label frequencies, in verbose mode ====
        if(verbose == True):
            #TODO:HERE
            assert False
        return toret
        
        
            
        
    
    @staticmethod
    def create_onetoone(str_dsname, rootdir, imgsprefix,\
                        func_get_patientrecords, func_get_wsiinfos):
        '''
        static method 
        creates a dataset from a directory containing one WSI per patient, where each WSI file has a specific prefix. 
        input - the name of the dataset, the root directory, the prefix of the images, and two functions: func_get_patientrecords (extract the patient information from the file name) and func_get_wsiinfos (extract information from the WSI file). 
        creates a Record object for each WSI file, and then creates a Patient object for each record, where the record is added to the patient's dictionary of records. Finally, it creates a Dataset object from the list of patients and returns it.
        '''
        '''
        If there is a one to one mapping between patients and images (i.e. one image per patient)
        this function can create the dataset.
        Inputs.
            - str_dsname: name of the str_dsname, a string.
            - rootdir: rootdir of the dataset, a string.
            - imgsprefix: prefix of the images, e.g., "svs", "ndpi", ... .
            - func_get_patientrecords: a function that takes in the file name, and has to return
                                       dict_patientrecords (excluding the WSI).
            - func_get_wsiinfos: a function that takes in the file name, and has to return
                                       dict_wsiinfos.
        '''
        #initial checks ==========================
        if(rootdir[-1]!="/"):
            raise Exception("Arguement: \n {} \n does not end with `/`")
        #get all file-names=========================
        #get the absolute fnames
        list_fnames = []
        for fname in Path(rootdir).rglob("*.{}".format(imgsprefix)):
            list_fnames.append(os.path.abspath(fname))
        #remove the rootdir from the beginning
        for idx_fname in range(len(list_fnames)):
            list_fnames[idx_fname] = list_fnames[idx_fname][len(rootdir)::]
        #sort fnames (to get consistent patient_names in different machines)
        list_fnames.sort()
        #make list_patients =================================
        list_patients = []
        count_createdpatients = 0
        for fname in list_fnames:
            new_record = Record(rootdir=rootdir,\
                       relativedir=fname,\
                       dict_infos=func_get_wsiinfos(fname))
            dict_patientrecord = func_get_patientrecords(fname)
            dict_patientrecord["wsi"] = new_record
            new_patient = Patient(int_uniqueid = count_createdpatients,\
                                  dict_records = dict_patientrecord)
            count_createdpatients += 1
            list_patients.append(new_patient)
        #make the Dataset ========
        dataset = Dataset(str_dsname, list_patients)
        return dataset
            
        
        
        
