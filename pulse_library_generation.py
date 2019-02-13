import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal as signal
import random
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import utils
import h5py
'''
Based on work by Federico De Guio: https://github.com/deguio/ML4DQM/blob/master/notebooks/Semi_Supervised/AE_ssl_random_hotdead_size.ipynb
See: https://indico.cern.ch/event/783825/contributions/3261553/attachments/1789464/2914642/hcaldpg_fpga_ml_reco_01feb2019.pdf
for possible implementation into FPGA
'''


def plot_loss(data, title,yscale="linear"):     
    """ Plots the training and validation loss yscale can be: linear,log,symlog """
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.plot(data.history["loss"])#, linestyle=line_styles[0], color=color_palette["Indigo"][900], linewidth=3)
    plt.plot(data.history["val_loss"])#, linestyle=line_styles[2], color=color_palette["Teal"][300], linewidth=3)
    plt.legend(["Train", "Validation"])#, loc="upper right", frameon=False)
    plt.yscale(yscale)
    plt.show();
    
class Pulse:
    '''
    A class that generates a Pulse that includes both Cerenkov and Scintillation Pulse. The pulse is randomly generated 
    through the use of a probability density function that is generated by the given parameters.
    The pulse has 512 samples and is identifiable by 3 parameters:
        1) ratio: If ratio is positive, it is the ratio of the Area of Cernekov:Area of Scintillation. 
                  If ratio is negative, it is the ratio of the Area of Scintillation:Area of Cerenkov
                
        This is altered by changing the height of the Cerenkov pulse (Cerenkov Radiation).
        
        
        2) separation: Time between the two pulses
       
    '''
    
    
    
    def __init__(self, photoelectrons, ratio, scint_decay):
        self.start=10
        self.rise=0.05
        self.cerenkov_decay=2.5
        self.scint_decay=scint_decay
        
        self.end_time=500
        self.timestep=0.1
        self.t=t=np.arange(0,self.end_time,self.timestep)
        self.bin_number=100
        

        if ratio>0:
            self.ratio=ratio
        else:
            self.ratio=-(1/ratio)
            
        self.photoelectrons=photoelectrons
        self.cerenkov_strength=self.ratio
        #self.cerenkov_percent=self.ratio/(self.ratio+1)
        #self.scint_percent=1/(self.ratio+1)
 
    def cerenkov(self,t):
        return (1-np.exp(-((t-self.start)/self.rise)))*(np.exp(-((t-self.start)/self.cerenkov_decay)))

    def scintillation(self, t):
        return (1-np.exp(-((t-self.start)/self.rise)))*(np.exp(-((t-self.start)/self.scint_decay)))
    
    def cerenkov_scaled(self, t):
        cerenkov=self.cerenkov(t)
        norm=sp.integrate.quad(self.cerenkov, self.start, self.end_time)
        return (1/norm[0])*self.cerenkov_strength*cerenkov
    
    def scintillation_normalized(self, t):
        scint=self.scintillation(t)
        norm=sp.integrate.quad(self.scintillation, self.start, self.end_time)
        return (1/norm[0])*scint


    def pulse_func_helper(self, time):
        if time<self.start:
            return 0
        else:
            return self.cerenkov_scaled(time) + self.scintillation_normalized(time) 
    
    
    def pulse_func(self, t):
        pulse=[]
                        
        for time in t:
            pulse.append(self.pulse_func_helper(time))
        return np.asarray(pulse)


    def randomValues(self,number):
        val=[]
        for x in range(number):
            val.append(random.random())
        return val

    
    def final_pdf(self, t):
        radiation=self.pulse_func(t)
        norm=sp.integrate.quad(self.pulse_func_helper, 0, self.end_time)
        return (1/norm[0])*radiation

    def final_cdf(self, t):
        prob=self.final_pdf(t)
        return np.cumsum(prob)*self.timestep

    def final_distribution(self):
        cumul=self.final_cdf(self.t)
        hits=self.photoelectrons
        #print(hits)
        rand=self.randomValues(hits)
        #print(rand)
        interp=sp.interpolate.interp1d(cumul,self.t, fill_value="extrapolate")
        vals=interp(rand)
        #print(vals)
        return vals
    
    def histogram_generator(self):
        data=self.final_distribution()
        hist, bin_edges=np.histogram(data, bins=self.bin_number, range=(0,self.end_time))
        return hist
    
    def output(self):
        '''
        Returns Histogram data and percent of area under 
        '''
        self.cerenkov_area=sp.integrate.quad(self.cerenkov_scaled, self.start, self.end_time)
        self.scint_area=sp.integrate.quad(self.scintillation_normalized, self.start, self.end_time)
        
        self.cerenkov_percent=(self.cerenkov_area[0])/(self.cerenkov_area[0]+self.scint_area[0])
        self.scint_percent=(self.scint_area[0])/(self.cerenkov_area[0]+self.scint_area[0])
        return self.histogram_generator(), self.cerenkov_percent, self.scint_percent



class PulseLibrary:
    '''
    Creates a list of pulses with the attached ratio.
    labels has form of ratioXXXscint_decayXXeventXX
    '''
    
    def __init__(self, seed, name, eventsperexperiment, photoelectrons,ratios=[1],scint_decays=[50]):
        np.random.seed(seed)
        self.name=name +'.hdf5'
        self.scint_decays=scint_decays
        self.ratios=ratios
        self.photoelectrons=photoelectrons
        self.eventsperexperiment=eventsperexperiment
        self.labels=[]
        self.pulses=[]
        self.inputs=[]
        self.outputs=[]
        
        
        
        file=h5py.File(self.name, 'w')
        
        for ratio in self.ratios:
            for scint_decay in self.scint_decays:
                pulse=Pulse(self.photoelectrons, ratio, scint_decay)
                
                temp1=str(ratio)
                group_name="ratio"+temp1+"scintdecay"+str(scint_decay)
                
                group = file.create_group(group_name)
                
                for i in range(eventsperexperiment):
                    
                    self.pulses.append(pulse)
                    ins, out1, out2=pulse.output()
                    outs=[out1, out2]
                    
                    
                    label="ratio"+temp1[:3]+"scintdecay"+str(scint_decay)+"event"+str(i)
                    self.labels.append(label)
                    self.inputs.append(ins)
                    self.outputs.append(outs)
                    
                    subgroup=group.create_group(label)
                    
                    
                    d1 = subgroup.create_dataset(label+'input', data=ins)
                    d2 = subgroup.create_dataset(label+'output', data=outs)
                    
        
        file.close()
    
    def get_inputs(self):
        return np.asarray(self.inputs)
    
    def get_outputs(self):
        return np.asarray(self.outputs)
    
    def get_labels(self):
        return self.labels
    
    def get_pulses(self):
        return np.asarray(self.pulses)
    

training_seed=7
testing_seed=8

scint_decays=np.linspace(2.5,200,6)
ratios=np.linspace(-100,100,6)
train_library=PulseLibrary(training_seed,"VaryRatioAndScintDecayTrainingLibrary", 50,1000, ratios, scint_decays)
test_library=PulseLibrary(testing_seed,"VaryRatioAndScintDecay", 50,1000, ratios, scint_decays)