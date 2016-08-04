import unittest
import os
import sys
import numpy as np
from subprocess import call
from brian2 import *
from brian2lems.supporting import *
import matplotlib.pyplot as plt

plain_numbers_from_list = lambda x: str(x)[1:-1].replace(',','')

class RecordingsTest(unittest.TestCase):
    "test of LEMSExporter"

    def test_one(self):
        pass

JNML_PATH = "C:/jNeuroMLJar"
xml_filename = "ifcgmtest.xml"
script = "simulation1_lif.py"
idx_to_record = [2, 55, 98]
output_jnml_file = "recording_ifcgmtest"

current_path = os.getcwd()

outcommand = os.popen("python {file} -d --recidx {recidx}".format(file=script,
                                                                  recidx=plain_numbers_from_list(idx_to_record)))

outcommand = call("python {file} --recidx {recidx}".format(file=script,
                                                           recidx=plain_numbers_from_list(idx_to_record)),
                  shell=True)

outbrian = np.load("recording.npy")
#print outbrian
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
print outbrian[1, 0:10]
print valuesvec[1, 1:10]
print np.allclose(outbrian[1, 0:10], valuesvec[1, 1:11])
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
#call(["python", "{file}".format(file=script)], shell=True)
