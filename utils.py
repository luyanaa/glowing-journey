import wormfunconn as wfc
from torch.utils.data import Dataset
import torch, numpy
import os
import joblib
from c302 import timeStep

# Get the responses
# stim_neu_id = "AWBL"
# resp_neu_ids = ["AVAR","AVAL","ASEL","AWAL","VD1"] #include wrong_id to test handling of error

# Feed Single Response
class responseGenerator:
    def __init__(self, folder, strain="wild-type", stim_type="rectangular" ):
        # Create FunctionalAtlas instance from file
        # strains: unc-31, wild-type
        self._funatlas = wfc.FunctionalAtlas.from_file(folder, strain)
        self.respList = self._funatlas.get_neuron_ids()
        self.stimList = self._funatlas.get_neuron_ids(stim=True)
        self.dt = timeStep
        self.stim_type= stim_type

    def step(self, stim_neu_id, resp_neu_ids, points):
        # Generate the stimulus array
        stim = self._funatlas.get_standard_stimulus(points,dt=self.dt,stim_type=self.stim_type,duration=points* self.dt)
        resp, labels, confidences, _ = self._funatlas.get_responses(stim, self.dt, stim_neu_id, resp_neu_ids=resp_neu_ids)
        return stim, resp, labels
    
    def Dataset(self):
        x = []
        y = []
        NeuronName=numpy.loadtxt("./pumpprobe/pumpprobe/aconnectome_ids.txt", dtype=object)[:, 1]

        def func(i):
            _stim, _resp, _labels= self.step(self.stimList[i], self.respList, 5000)
            stim = numpy.zeros((300, 5000))
            resp = numpy.zeros((300, 5000))
            stim[numpy.where(NeuronName == i)] = _stim
            for index in range(len(_labels)):
                resp[numpy.where(NeuronName == _labels[index])] = _resp[index]
            resp[numpy.where(NeuronName == i)] = _stim
            return (stim, resp)
        result = joblib.Parallel(n_jobs=-1)(joblib.delayed(func)(i) for i in range(len(self.stimList)))
        result = numpy.array(result)

        x = torch.Tensor(result[:,0,:,:]).swapaxes(1,2)
        y = torch.Tensor(result[:,1,:,:]).swapaxes(1,2)

        return x,y
class PsuedoDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

# Feed Raw Response
# Problem: We dont known half of labels
class AtlasLoader(Dataset):
    def __init__(self, path, strain="wild-type", device="cpu"):
        self.x = []
        self.y = []
        if strain == "wild-type":
            path = os.path.join(path, "exported_data")
        elif strain == "unc-31":
            path = os.path.join(path, "exported_data_unc31")
        else:
            raise NotImplementedError        
        # From 0 to 112. 
        for i in range(0, 113):
            _resp_data = []
            _resp_lbl = []
            data = numpy.loadtxt(os.path.join(path, str(i)+"_gcamp.txt"))
            t = numpy.loadtxt(os.path.join(path, str(i)+"_t.txt"))
            # stim_neu: The neuron stimulated. 
            stim_neu = numpy.loadtxt(os.path.join(path, str(i)+"_stim_neurons.txt"), converters=numpy.int64)
            stim_vol_i = numpy.loadtxt(os.path.join(path, str(i)+"_stim_volume_i.txt"), converters=numpy.int64)
            
            # _labels.txt, respond label
            # Randi et al. didn't recognize every neuron's label, so there's unlabeled ones. 
            with open(os.path.join(path, str(i)+"_labels.txt")) as f:
                resp_lbl = f.readlines()
            for label in range(len(resp_lbl)):
                if resp_lbl[label] == "" or resp_lbl[label] == "\n":
                    continue
                else:
                    _resp_data.append(data[label])
                    _resp_lbl.append(resp_lbl[label])
            
            _resp_data = numpy.array(_resp_data)
            _resp_lbl = numpy.array(_resp_lbl, dtype=object)

            _resp_data = torch.Tensor(_resp_data)
            self.x.append((stim_neu, stim_vol_i))
            self.y.append((t, _resp_data, _resp_lbl))
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class WormTensor(Dataset):
    def __init__(self, path, device="cpu"):
        self.x = []
        self.y = []        
        # From 1 to 24. 
        for i in range(1, 25):
            _resp_data = []
            _resp_lbl = []
            data = numpy.loadtxt(os.path.join(path, str(i)+"_gcamp.txt"))
            t = numpy.loadtxt(os.path.join(path, str(i)+"_t.txt"))
            # stim_neu: The neuron stimulated. 
            stim_neu = numpy.loadtxt(os.path.join(path, str(i)+"_stim_neurons.txt"), converters=numpy.int64)
            stim_vol_i = numpy.loadtxt(os.path.join(path, str(i)+"_stim_volume_i.txt"), converters=numpy.int64)
            
            # _labels.txt, respond label
            # Randi et al. didn't recognize every neuron's label, so there's unlabeled ones. 
            with open(os.path.join(path, str(i)+"_labels.txt")) as f:
                resp_lbl = f.readlines()
            for label in range(len(resp_lbl)):
                if resp_lbl[label] == "" or resp_lbl[label] == "\n":
                    continue
                else:
                    _resp_data.append(data[label])
                    _resp_lbl.append(resp_lbl[label])
            
            _resp_data = numpy.array(_resp_data)
            _resp_lbl = numpy.array(_resp_lbl, dtype=object)

            _resp_data = torch.Tensor(_resp_data)
            self.x.append((stim_neu, stim_vol_i))
            self.y.append((t, _resp_data, _resp_lbl))
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
