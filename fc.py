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
    default_parameters = {'zeropos':None,'refatoms2':None,'refatoms3':None}
    nolabel = True
    @fn_timer
    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)       
        refatoms2=self.parameters.refatoms2
        refatoms3=self.parameters.refatoms3
        zeropos=self.parameters.zeropos  
        unit=io.read('POSCAR')
        self.fc2=readfc2('fc2new')
        self.fc3=readfc3(refatoms3,unit,'fc3new')       
        self.fc2=self.expand2(refatoms2,zeropos,unit,self.fc2)
        self.fc3=self.expand3(refatoms3,zeropos,unit,self.fc3)  

    @fn_timer
    def expand2(self,refatoms,zeropos,unit,fc):
        n=len(zeropos)
        d=fftn(fc,axes=[0,1])
        fc2=ifftn(d,s=[n,n],axes=[0,1]).real
        return fc2
    @fn_timer
    def expand3(self,refatoms,zeropos,unit,fc):
        n=len(zeropos)
        d=fftn(fc,axes=[0,1,2])
        fc3=ifftn(d,s=[n,n,n],axes=[0,1,2]).real
        return fc3
    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        zeropos=self.parameters.zeropos  
        positions = self.atoms.positions      
        u=positions-zeropos.positions
        f2=np.einsum('ijkl,jl',self.fc2,u)
        f3=np.einsum('ijklmn,jm,kn',self.fc3 ,u,u)
        self.results['forces']=f2+f3
        self.results['energy'] = np.einsum('ik,ik',f2*.5+1.0/6*f3,u)
