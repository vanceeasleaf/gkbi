from __future__ import division

import numpy as np
from numpy.linalg  import norm
from ase.calculators.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from aces.f import readfc2,readfc3,rotationMatrix
from ase import io
from aces.f import fn_timer
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
    def getmapping(self,ref,abs):
        maps=np.zeros(len(abs))
        for ii,x in enumerate(abs):
            for jj,y in enumerate(ref):
                if(np.allclose(x,y)):
                    maps[ii]=jj
                    break
        return maps
    @fn_timer
    def getdup2(self,n,maps,fc):
        fc2=np.zeros([n,n,3,3])
        for i in range(n):
            for j in range(n):
                ii=maps[i]
                jj=maps[j]
                fc2[i,j]=fc[ii,jj]
        return fc2
    @fn_timer
    def removepbc2(self,refatoms,fc):
        dis=refatoms.get_all_distances(mic=True)
        dis1=refatoms.get_all_distances(mic=False)
        filter=dis-dis1<0.0
        fc[filter]=0.0
        return fc
    @fn_timer
    def removepbc3(self,refatoms,fc):
        dis=refatoms.get_all_distances(mic=True)
        dis1=refatoms.get_all_distances(mic=False)
        filter=dis-dis1<0.0
        fc[filter]=0.0
        fc[:,filter]=0.0
        fc=np.einsum('ikjlmn',fc)
        fc.flags.writeable = True
        fc[filter]=0.0
        fc=np.einsum('ikjlmn',fc)
        return fc
    @fn_timer
    def expand2(self,refatoms,zeropos,unit,fc):
        fc=self.removepbc2(refatoms,fc)
        n=len(zeropos)
        fc2=np.zeros([n,n,3,3])
        dim=(norm(zeropos.cell,axis=1)/norm(unit.cell,axis=1)).astype(np.int) #3x3
        for i in range(dim[0]):
            for j in range(dim[1]):
                for k in range(dim[2]):
                    pos=refatoms.positions+np.array([i,j,k]).dot(unit.cell)
                    nn=np.floor(pos/norm(zeropos.cell,axis=1))
                    pos-=nn.dot(zeropos.cell)
                    maps=self.getmapping(pos,zeropos.positions)
                    fc2+=self.getdup2(n,maps,fc)
        fc2/=(float)(dim[0]*dim[1]*dim[2])
        return fc2
    @fn_timer
    def getdup3(self,n,maps,fc):
        fc3=np.zeros([n,n,n,3,3,3])
        for i in xrange(n):
            ii=maps[i]
            for j in xrange(n):
                jj=maps[j]
                for k in xrange(n):                          
                    kk=maps[k]
                    fc3[i,j,k]=fc[ii,jj,kk]
        return fc3
    @fn_timer
    def expand3(self,refatoms,zeropos,unit,fc):
        fc=self.removepbc3(refatoms,fc)
        n=len(zeropos)
        fc3=np.zeros([n,n,n,3,3,3])
        dim=(norm(zeropos.cell,axis=1)/norm(unit.cell,axis=1)).astype(np.int) #3x3
        for i in range(dim[0]):
            for j in range(dim[1]):
                for k in range(dim[2]):
                    pos=refatoms.positions+np.array([i,j,k]).dot(unit.cell)
                    nn=np.floor(pos/norm(zeropos.cell,axis=1))
                    pos-=nn.dot(zeropos.cell)
                    maps=self.getmapping(pos,zeropos.positions)
                    fc3+=self.getdup3(n,maps,fc)
        fc3/=(float)(dim[0]*dim[1]*dim[2])
        return fc3
    @fn_timer
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
