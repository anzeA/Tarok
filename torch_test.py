import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import  TensorDataset


class Net_vrednotenje_roke(pl.LightningModule):
    def __init__(self,dropout=0.2):
        super(Net_vrednotenje_roke, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(54,32),
        nn.ELU(),
        nn.BatchNorm1d(32),
        nn.Dropout(dropout),
        nn.Linear( 32,32 ),
        nn.ELU(),
        nn.BatchNorm1d( 32 ),
        nn.Dropout( dropout ),
        nn.Linear( 32,
        1+ #Naprej
        4*3+ # tri dva ena v vseh sterih kraljih
        3+ # 3 solo
        1+# zaprti
        #1+  odprti berac
        1#solo brez
         )
        )

    def forward(self, x):
        return self.model(x)

class Net_Navadna_igra(pl.LightningModule):
#class Net_Navadna_igra(nn.Module):
    def __init__(self,dropout=0.2):
        super(Net_Navadna_igra, self).__init__()
        self.data =None

        self.nasprotniki = nn.GRU(54*3,32,batch_first=True) #poprav batch first
        self.roka = nn.GRU(54,32,batch_first=True)
        self.zdruzi_roke = nn.Linear(64,32)
        self.talon_lin = nn.Linear(55*6,16)
        self.zalozil_lin = nn.Linear(54,16)
        self.dodaj_talon_igralca = nn.Linear(40,32)
        self.model = nn.Sequential(
            nn.Linear(64,54),
            nn.ELU(),
            nn.BatchNorm1d( 54 ),
            nn.Dropout( dropout ),

            nn.Linear( 54, 54 ),
            nn.ELU(),
            nn.BatchNorm1d( 54 ),
            nn.Dropout( dropout ),
            nn.Linear(54,54)
        )
        self.optimizer = optim.Adam( self.parameters(), lr=0.01 )

    #def forward(self, nasprotniki,roka,kralj,index_tistega_ki_igra,talon_input,zalozil_input): # ostalo = kralj,index_tistega_ki_igra,zalozil_input
    def forward(self, inp): # ostalo = kralj,index_tistega_ki_igra,zalozil_input
        nasprotniki, roka, kralj, index_tistega_ki_igra, talon_input, zalozil_input = inp
        nasprotniki = self.nasprotniki(nasprotniki)
        nasprotniki = nasprotniki[0].squeeze()[:,-1]
        roka = self.roka(roka)[0].squeeze()[:,-1]
        x = torch.cat((roka,nasprotniki),1)
        x = F.elu( self.zdruzi_roke(x) ) #32
        x = torch.cat((x,index_tistega_ki_igra,kralj,),1) #40
        x = F.elu( self.dodaj_talon_igralca(x) )  #32
        talon_x = F.elu(self.talon_lin(talon_input) )
        zalozil_x = F.elu(self.zalozil_lin(zalozil_input) )
        x = torch.cat((x,zalozil_x,talon_x),1) # 64
        x  =self.model(x)
        return x

    def train_dataloader(self) :
        device=None
        batch_size = 50000
        # data = nasprotniki,                        roka,                           kralj,              index_tistega_ki_igra,  talon_input,            zalozil_input
        #Y = TensorDataset(torch.randn( batch_size, 54, device=device ))
        Y = torch.randn( batch_size, 54, device=device )
        t = [torch.randn( batch_size, 7, 54 * 3, device=device ),
         torch.randn( batch_size, 7, 54, device=device ), torch.randn( batch_size, 4, device=device ),
         torch.randn( batch_size, 4, device=device ), torch.randn( batch_size, 55 * 6, device=device ),
         torch.randn( batch_size, 54, device=device )]
        #t = [TensorDataset(tens) for tens in t]
        t.append( Y )
        t = TensorDataset(*t)
        return torch.utils.data.DataLoader( t,batch_size=128*8,pin_memory=True, num_workers=7) #,pin_memory=True,, num_workers=5



        #return DataLoader([( ( torch.randn(7,54*3,device=device),torch.randn(7,54,device=device),torch.randn(4,device=device),torch.randn(4,device=device),torch.randn(55*6,device=device),torch.randn(54,device=device) ),torch.randn(54,device=device) )  for i in range(1000)],batch_size=32,shuffle=True)
        #return [( ( torch.randn(7,54*3,device=device),torch.randn(7,54,device=device),torch.randn(4,device=device),torch.randn(4,device=device),torch.randn(55*6,device=device),torch.randn(54,device=device) ),torch.randn(54,device=device) )  for i in range(1000)]

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch,batch_idx):

        output = self.forward(batch[:-1])
        loss = F.mse_loss(output,batch[-1])
        return {'loss':loss}

class Net_Klop(pl.LightningModule):
    def __init__(self,dropout=0.2):
        super(Net_Klop, self).__init__()
        self.nasprotniki = nn.GRU(54*3,32,batch_first=True)
        self.roka = nn.GRU(54,32,batch_first=True)
        self.zdruzi_roke = nn.Linear(64,32)
        self.talon_lin = nn.Linear(54,16)
        self.model = nn.Sequential(
            nn.Linear(48,54),
            nn.ELU(),
            nn.BatchNorm1d( 54 ),
            nn.Dropout( dropout ),

            nn.Linear( 54, 54 ),
            nn.ELU(),
            nn.BatchNorm1d( 54 ),
            nn.Dropout( dropout ),
            nn.Linear(54,54)
        )
        self.optimizer = optim.Adam( self.parameters(), lr=0.01 )

    def forward(self, inp):
        input_layer_nasprotiki,roka_input,talon_input = inp
        nasprotniki = self.nasprotniki(input_layer_nasprotiki)[0].squeeze()[:,-1] #32
        roka = self.roka(roka_input)[0].squeeze()[:,-1]#32
        x = torch.cat((roka,nasprotniki),1)#32
        x = F.elu( self.zdruzi_roke(x) ) #32
        talon_x = F.elu(self.talon_lin(talon_input) )#16
        x = torch.cat((x,talon_x),1) # 48
        x  =self.model(x)
        return x

    def train_dataloader(self) :
        device=None
        batch_size = 50000
        time_Stamp=7
        # data = nasprotniki,                        roka,                           kralj,              index_tistega_ki_igra,  talon_input,            zalozil_input
        #Y = TensorDataset(torch.randn( batch_size, 54, device=device ))
        Y = torch.randn( batch_size, 54, device=device )
        t = [torch.randn( batch_size, time_Stamp, 54 * 3, device=device ),
         torch.randn( batch_size, time_Stamp, 54, device=device ),
         torch.randn( batch_size, 54, device=device )]
        #t = [TensorDataset(tens) for tens in t]
        t.append( Y )
        t = TensorDataset(*t)
        return torch.utils.data.DataLoader( t,batch_size=128*8,pin_memory=True, num_workers=7) #,pin_memory=True,, num_workers=5



        #return DataLoader([( ( torch.randn(7,54*3,device=device),torch.randn(7,54,device=device),torch.randn(4,device=device),torch.randn(4,device=device),torch.randn(55*6,device=device),torch.randn(54,device=device) ),torch.randn(54,device=device) )  for i in range(1000)],batch_size=32,shuffle=True)
        #return [( ( torch.randn(7,54*3,device=device),torch.randn(7,54,device=device),torch.randn(4,device=device),torch.randn(4,device=device),torch.randn(55*6,device=device),torch.randn(54,device=device) ),torch.randn(54,device=device) )  for i in range(1000)]

    def configure_optimizers(self):
        return self.optimizer#self.optimizer

    def training_step(self, batch,batch_idx):

        output = self.forward(batch[:-1])
        loss = F.mse_loss(output,batch[-1])
        return {'loss':loss}

def test_vrednotenje_roke(device):
    roka = torch.randn( 2, 54 )
    #print( roka )
    net = Net_vrednotenje_roke()
    y = net( roka )
    print(y)
    trainloader = [(torch.randn(32,54,device=device),torch.randn(32,18,device=device)) for i in range(1000)]
    #print( y )
    criterion = nn.MSELoss()
    optimizer = optim.Adam( net.parameters(), lr=0.01 )
    t = time.time()
    for epoch in range( 5 ):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate( trainloader, 0 ):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net( inputs )
            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print( '[%d, %5d] loss: %.3f' %
                       (epoch + 1, i + 1, running_loss / 100) )
                running_loss = 0.0
    t = time.time()-t
    print( 'Finished Training in:',t )
    print(y)
    print(net( roka ))

def test_navadna_igra(device):
    model = Net_Navadna_igra()
    if device != 'cpu':
        model.cuda(device=device)
    batch_size = 32
    #data = nasprotniki,                        roka,                           kralj,              index_tistega_ki_igra,  talon_input,            zalozil_input
    trainloader = [( ( torch.randn(7,batch_size,54*3,device=device),torch.randn(7,batch_size,54,device=device),torch.randn(batch_size,4,device=device),torch.randn(batch_size,4,device=device),torch.randn(batch_size,55*6,device=device),torch.randn(batch_size,54,device=device) ),torch.randn(batch_size,54,device=device) )  for i in range(1000)]
    criterion = nn.MSELoss()
    optimizer = optim.Adam( model.parameters(), lr=0.01 )
    t = time.time()
    for epoch in range( 2 ):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate( trainloader, 0 ):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model( *inputs )
            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 2000 mini-batches
                print( '[%d, %5d] loss: %.3f' %
                       (epoch + 1, i + 1, running_loss / 100) )
                running_loss = 0.0
    t = time.time() - t
    print( 'Finished Training on',device,'in:', t )

from sklearn.datasets import make_regression



def test_multi_input():
    net = Net_Navadna_igra()
    trainer = Trainer(  profile=True )
    # trainer = Trainer()
    trainer.fit( net )
    # test_navadna_igra(device)
    # test_navadna_igra('cpu')

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
    #device = "cpu"
    #test_vrednotenje_roke(device)
    #device = 'cpu'
    #test_gpu()
    Trainer().fit(Net_Klop())
    #test_2()
    print(device)
