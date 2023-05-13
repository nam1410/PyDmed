import numpy as np
import math
def lcm(list_numbers):
    '''
    Computes the lcm of numbers in a list.
    '''
    '''
    initializes the variable lcm to the first element in the input list_numbers - using a for loop, it multiplies each subsequent element of the list with lcm and divides the result by the greatest common divisor (GCD) of lcm and the current element, using the math.gcd() function. Finally, the function returns the resulting lcm value after the loop is completed. 
    '''
    lcm = list_numbers[0]
    for idx_number in range(1, len(list_numbers)):
        lcm = (lcm*list_numbers[idx_number])/math.gcd(lcm, list_numbers[idx_number])
        lcm = int(lcm)
    return lcm
        
def multimode(list_input):
    '''
    calculates the mode(s) of a given list of values. 
    The function first creates a set of input list list_input. Then, it creates a dictionary dict_freqs with the keys as values in list_input and their values as their frequencies (number of occurrences) in the list, initially set to zero - for loop, the function counts the frequencies of each element in list_input by updating the corresponding value in dict_freqs. After the loop is completed, the function finds the mode(s) by finding the key(s) with the highest frequency in dict_freqs. 
    '''
    '''
    `statistics.multimode` does not exist in all python versions.
    Therefore minimath.multimode is implemented.
    '''
    set_data = set(list_input)
    dict_freqs = {val:0 for val in set_data}
    for elem in list_input:
        dict_freqs[elem] = dict_freqs[elem] + 1
    mode = max((v, k) for k, v in dict_freqs.items())[1]
    return mode

def multiminority(list_input):
    '''
    returns the minority value(s) in a given list of values. 
    The function first creates a set of input list list_input. Then, it creates a dictionary dict_freqs with the keys as  values in list_input and their values as their frequencies (number of occurrences) in the list, initially set to zero - for loop, the function counts the frequencies of each element in list_input by updating the corresponding value in dict_freqs. After the loop is completed, the function finds the minority value(s) by finding the key(s) with the lowest frequency in dict_freqs.
    '''
    '''
    Returns the minority in a list. This function works if there are many minorities available in the list.
    '''
    set_data = set(list_input)
    dict_freqs = {val:0 for val in set_data}
    for elem in list_input:
        dict_freqs[elem] = dict_freqs[elem] + 1
    minority = min((v, k) for k, v in dict_freqs.items())[1]
    return minority
