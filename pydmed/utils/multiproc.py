import numpy as np
import matplotlib.pyplot as plt
import os, sys
import psutil
from pathlib import Path
import re
import time
import random
import multiprocessing as mp #alias - mp: The multiprocessing module provides a way to create and manage parallel processes, allowing programs to utilize multiple CPU cores and perform computations faster. It provides a way to spawn processes using an API similar to the threading module, and also includes additional features such as shared memory, synchronization primitives, and inter-process communication.
from abc import ABC, abstractmethod #abc helps in creating abstract base classes that defines methods and properties (using a decorator '@') that must be implemented by its concrete subclasses 
import openslide
import torch
import torchvision
import torchvision.models as models
from multiprocessing import Process, Queue 
'''
These classes are used to create and manage parallel processes and enable inter-process communication. 
1. The Process class is used to create a new process. An instance of the Process class represents a separate process that runs independently of the main program. To create a new process, create an instance of the Process class and call the start() method on it. 
2. The Queue class is used for inter-process communication. It provides a way for multiple processes to share data, such as messages, without accessing the same memory space. A Queue instance can be used to add items from one process and retrieve them from another process.
'''
def poplast_from_queue(queue):
    '''
    pops the last element from a multiprocessing.Queue instance and returns it.
    First, it gets the size of the queue using the qsize() method. If the size of the queue is zero, the function returns None. Else, it starts a loop that iterates over the number of elements in the queue. For each iteration, the function tries to get an element from the queue using the get_nowait() method, which retrieves an element from the queue without blocking if the queue is empty.
    Since the function iterates over the entire queue, it effectively removes all elements except for the last one. The last element is the one that is finally returned by the function. If an exception is encountered while getting an element from the queue, the function ignores it and continues with the loop. This only works correctly for multiprocessing.Queue instances where elements are retrieved in the order they were added. If elements are removed from the queue out of order, the function may return an incorrect result.
    '''
    '''
    Pops the last element of a `multiprocessing.Queue`.
    '''
    size_queue = queue.qsize()
    if(size_queue == 0):
        return None
    elem = None
    for count in range(size_queue):
        try:
            elem = queue.get_nowait()
        except:
            pass
    return elem


def set_nicemax():
    '''
    sets the priority of the current process to the highest possible value. The priority of a process determines how much CPU time it is allocated and how quickly it is scheduled to run compared to other processes. A higher priority means that the process is given more CPU time and is scheduled to run more frequently. The function first sets the value of maxcount to 1000, which is used to limit the maximum number of iterations the function will run. The function then gets the current priority level of the process using the os.nice() function and sets it to N_old. The function then enters a loop that increments a counter count and sets the priority level of the process to N_old + 1000 using os.nice(N_old+1000). The loop continues until the new priority level is the same as the old priority level, indicating that the highest priority level has been reached. If the loop iterates more than maxcount times without reaching the highest priority level, the function returns without setting the priority level to the maximum value.
    Overall, the function is used to ensure that the current process is given the highest priority by the operating system, which can be useful in situations where the process needs to complete a task quickly or perform real-time processing. 
    '''
    '''
    Sets the priority of the process to the highest value. 
    '''
    maxcount = 1000
    N_old = os.nice(0)
    count = 0
    while(True):
        count += 1
        N_new = os.nice(N_old+1000)
        if(N_new == N_old):
            return
        if(count > maxcount):
            return



def terminaterecursively(pid):
    '''
    recursively terminates all child processes and the parent process with the given process id (pid). The function uses the psutil library to get a Process object representing the parent process with the given pid. It then uses the children(recursive=True) method to get a list of all child processes of the parent process, including their descendants. It then iterates over each child process and attempts to kill it using the kill() method of the Process object. If an exception occurs while attempting to kill a child process, the function ignores it and continues with the loop. After all child processes have been killed, the function attempts to kill the parent process using the same method. 
    Overall, the function is used to terminate a process and all of its child processes, which can be useful in situations where a process has become unresponsive or needs to be stopped quickly. 
    '''
    print("=================================================================================")
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
