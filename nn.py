class NeuralNet:
    '''
    Provides functionality for creating and accessing attributes of 
    a deep fully connected neural network
    
    
    Instance Attributes:
        _network_info: contains all relevant information about network 
                      (see init function for more detail)
        _number_of_layers: number of layers [int>0]
        _input_shape: input vector size [int>0]
        _architecture: a dict containing architecture of network
        _loss: loss function used in network
        _optimizer: optimizer used to train network
        _model: the model itself [keras.models.Sequential]
    
    
    '''
    
    
    def __init__(self, network_info):
        '''
        Creates Instance of NeuralNet
        
        network_info is a dict that contains 3 elements
            1. input_shape[int]
            2. loss [str]
            3. optimizer [str]
            2. architecture [list of dicts]
                each element in the list is a layer represented by
                a dict that contains 3 elements
                    i)   number_of_nodes
                    ii)  kernel_initializer
                    iii) activation
                    
        Example: the dict  below creates a 1-layer autoencoder
        {'input_shape':20, 'loss':'mean_squared_error', 'optimizer':'adam', 
          'architecture':[{'number_of_nodes':  5, 'kernel_initializer': 'ones', 'activation': 'relu'},
                          {'number_of_nodes': 20, 'kernel_initializer': 'ones', 'activation': 'relu'}]
        }
        
        '''
        self._network_info=network_info
        self._input_shape=self._network_info['input_shape']
        self._number_of_layers=len(self._network_info['architecture'])
        self._architecture=self._network_info['architecture']
        self._loss=self._network_info['loss']
        self._optimizer=self._network_info['optimizer']
        
        self._model=Sequential()
        
        for i in range(len(self._architecture)):
            number_of_nodes=self._architecture[i]['number_of_nodes']
            kernel_initializer=self._architecture[i]['kernel_initializer']
            activation=self._architecture[i]['activation']
            if i==0:
                self._model.add(Dense(number_of_nodes,  
                                      input_shape=(self._input_shape,), 
                                      kernel_initializer=kernel_initializer, 
                                      activation=activation))
            else:
                self._model.add(Dense(number_of_nodes,  
                                      kernel_initializer=kernel_initializer, 
                                      activation=activation))
        self._model.compile(loss=self._loss, optimizer=self._optimizer)
    
    
    
    
    
    def model_trainer(self, pulse_library, model_name):
    file=h5py.File(pulse_library, 'r')
    X=[]
    Y=[]
    for groups in file.values():
        for subgroup in groups.values():
            for event in subgroup.values():
                #if not "ratio0.0scint"in event.name:
                #if 'event0' in event.name:  
                if event.name[-5:]=="input":
                    data=event[:]
                    data=np.asarray(data)                      
                    X.append(data)
                    #X.append(data)
                elif event.name[-6:]=="output":
                    Y.append(event[:])
    file.close()    
    X=np.asarray(X)

    Y=np.asarray(Y)
    norm_factor=np.amax(X)
    X_scaled=X/norm_factor
    Xtrain, Xval, Ytrain, Yval = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    checkpointer = ModelCheckpoint(filepath=model_name+"weights.hdf5",verbose=3, save_best_only=True)
    earlystop= EarlyStopping(monitor='val_loss', min_delta=0, patience=50 ,verbose=1, mode='auto')
    history=self._model.fit(Xtrain,Ytrain,epochs=200,verbose=1
                             ,validation_data=(Xval,Yval)
                             ,shuffle=True, batch_size=16
                             ,callbacks=[earlystop,checkpointer])
    self._history=history
    self._model.save(model_name)
    return self._model, self.history

    @staticmethod
    def model_verifier(self, pulse_library, model_name, ratios, scint_decays, eventsperexperiment):
        model=load_model(model_name)
        file=h5py.File(pulse_library, 'r')
        X=[]
        Y=[]
        labels=[]
        
        for ratio in ratios:
            for scint_decay in scint_decays:
                tempX=[]
                tempY=[]
                templabels=[]
                for i in range(eventsperexperiment):
                    label="/ratio"+str(ratio)+"scintdecay"+str(scint_decay)+"/ratio"+str(ratio)+"scintdecay"+str(scint_decay)+"event"+str(i)+"/ratio"+str(ratio)+"scintdecay"+str(scint_decay)+"event"+str(i)
                    #print(label)
                    Xdata=file[label+"input"][:]
                    Ydata=file[label+"output"][:]
                    Xdata=np.asarray(Xdata)
                    
                    tempX.append(Xdata)
                    tempY.append(Ydata)
                    templabels.append(label)
                X.append(tempX)
                Y.append(tempY)
                labels.append(templabels)
        
        file.close()
        X=np.asarray(X)
        Y=np.asarray(Y)
        results=[]
        norm_factor=np.amax(X)
        X_scaled=X/norm_factor
        
        for x in X_scaled:
            prediction=model.predict(x)
            results.append(prediction)
    
        results=np.array(results)
        #results=model.predict(X)
        return results, Y, labels
    
    
    
    def get_architecture(self):
        return self._architecture
    
    def get_number_of_layers(self):
        return self._number_of_layers
    
    def get_network_info(self):
        return self._network_info
    
    def get_loss(self):
        return self._loss
    
    def get_optimizer(self):
        return self._optimizer
    
    def get_input_shape(self):
        return self._input_shape
    
    def get_model(self):
        return self._model
    
    
    
    