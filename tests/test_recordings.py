import unittest
import os
import sys
import platform
import numpy as np
from subprocess import call
from brian2 import *
from brian2lems.supporting import *
import matplotlib.pyplot as plt

from numpy.testing import assert_raises, assert_equal, assert_array_equal

plain_numbers_from_list = lambda x: str(x)[1:-1].replace(',','')

plot = False

if platform.system()=='Windows':
    JNML_PATH = "C:/jNeuroMLJar"
else:
    JNML_PATH = ""
xml_filename = "ifcgmtest.xml"
idx_to_record = [2, 55, 98]
output_jnml_file = "recording_ifcgmtest"

def test_simulation1():
    """
    Test example 1: simulation1_lif.py
    """
    SCRIPT = "simulation1_lif.py"
    RECORDING_BRIAN_FILENAME = "recording"

    current_path = os.getcwd()
    outcommand = os.popen("python {file} -d --recidx {recidx}".format(file=SCRIPT,
                                                                      recidx=plain_numbers_from_list(idx_to_record)))
    outcommand = call("python {file} --recidx {recidx}".format(file=SCRIPT,
                                                               recidx=plain_numbers_from_list(idx_to_record)),
                      shell=True)

    outbrian = np.load(os.path.join(os.path.dirname(__file__), RECORDING_BRIAN_FILENAME+".npy"))
    #print outbrian
    if JNML_PATH:
        os.chdir(JNML_PATH)
    outcommand = call("jnml {path} -nogui".format(path=os.path.join(current_path, xml_filename)),
                      shell=True)
    #print outcommand

    timevec = []
    valuesvec = []
    with open(output_jnml_file+'.dat','r') as f:
        for line in f:
            timevec.append(float(line.split("\t")[0]))
            valuesvec.append([float(x) for x in line.split("\t")[1:-1]])

    timevec = np.asarray(timevec)
    valuesvec = np.asarray(valuesvec).T
    for i in range(len(idx_to_record)):
        assert np.allclose(outbrian[i, :], valuesvec[i, 1:], atol=1e-02)==True

    if plot:
        plt.subplot(3,2,1)
        plt.plot(outbrian[0,:])
        plt.subplot(3,2,2)
        plt.plot(valuesvec[0,:])
        plt.subplot(3,2,3)
        plt.plot(outbrian[1,:])
        plt.subplot(3,2,4)
        plt.plot(valuesvec[1,:])
        plt.subplot(3,2,5)
        plt.plot(outbrian[2,:])
        plt.subplot(3,2,6)
        plt.plot(valuesvec[2,:])
        plt.show()
    os.chdir(current_path)
