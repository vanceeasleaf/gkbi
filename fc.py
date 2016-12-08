from __future__ import division

import numpy as np
from numpy.linalg  import norm
from ase.calculators.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from aces.f import readfc2,readfc3,rotationMatrix
from ase import io
from aces.f import fn_timer
from numpy.fft import fftn,ifftn
from aces.tools import *
class FC(Calculator):
    implemented_properties = ['energy', 'forces']
    nolabel = True
    @fn_timer
    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)       
        self.fc2=self.parameters.fc2
        self.fc3=self.parameters.fc3
    def calculate(self, atoms=None,
                  properties=['forces','energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        zeropos=self.parameters.zeropos  
        positions = self.atoms.positions      
        u=positions-zeropos.positions
        us=np.zeros([27,len(u)])
        uss=np.zeros([27,len(u),3])
        import itertools
        for j,(ja,jb,jc) in enumerate(itertools.product(xrange(-1,2),
                                                xrange(-1,2),
                                                xrange(-1,2))):
            uj=u+np.dot([ja,jb,jc],zeropos.cell)
            us[j]=np.linalg.norm(uj,axis=1)
            uss[j]=uj
        map0=np.argmin(us,axis=0)
        for i,x in enumerate(map0):
            u[i]=uss[x,i]
        f2=-np.einsum('ijkl,jl',self.fc2,u)
        f3=-np.einsum('ijklmn,jm,kn',self.fc3 ,u,u)
        self.results['forces']=(f2+f3)
        self.results['energy'] = -np.einsum('ik,ik',f2*.5+1.0/6*f3,u)
