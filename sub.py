# -*- coding: utf-8 -*-
# @Author: YangZhou
# @Date:   2016-11-09 22:18:07
# @Last Modified by:   YangZhou
# @Last Modified time: 2016-12-05 17:44:45
from ase import io 
from ase.calculators.lj import LennardJones
from fc import FC
from ase import units
import numpy as np
from ase.md.npt import NPT
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.io.trajectory import PickleTrajectory
from aces.f import fn_timer,rotationMatrix,premitiveSuperMapper,readfc2
from aces.tools import passthru
from ase import Atoms
def test_rotationMatrix():
	M=rotationMatrix([0,0,1],0)
	assert np.allclose(M,np.eye(3))
	M=rotationMatrix([0,0,1],np.pi/2)
	assert np.allclose(M,np.array([0,-1,0,1,0,0,0,0,1]).reshape([3,3]))
def test_getfc3():
	test_rotationMatrix()
	fc=np.zeros([2,2,2,3,3,3])
	fc[1,1,0]=np.eye(3)
	fc1=getfc3([0,0,1],90,fc)
	u=np.array([1,0,1])
	v=np.array([1,1,0])
	M=np.array([0,-1,0,1,0,0,0,0,1]).reshape([3,3])
	u1=M.dot(u)
	assert np.allclose(u1,[0,1,1])
	v1=M.dot(v)
	assert np.allclose(v1,[-1,1,0])
	f=np.einsum('ijk,j,k',fc[1,1,0],u,v)
	f1=np.einsum('ijk,j,k',fc1[1,1,0],u1,v1)
	assert np.allclose(f1,M.dot(f))
def getfc3(direct,t,fc):
	M=rotationMatrix(direct,t*np.pi/180.0)
	assert np.allclose(M.dot(M.T),np.eye(3))
	fc=np.einsum(M,[6,3],M.T,[7,4],M.T,[8,5],fc,[0,1,2,3,4,5],[0,1,2,6,7,8])
	return fc
def getfc2(direct,t,fc):
	M=rotationMatrix(direct,t*np.pi/180.0)	
	fc=np.einsum(M,[4,2],fc,[0,1,2,3],[0,1,4,3])
	fc=np.einsum(M.T,[4,3],fc,[0,1,2,3],[0,1,2,4])
	return fc
def test_fc2():
	atoms=io.read('POSCAR-5')
	n=len(atoms)
	forces=np.loadtxt('force-5.txt')
	assert np.allclose(forces.shape,[n,3])
	ref=io.read('SPOSCAR')
	u=atoms.get_scaled_positions()-ref.get_scaled_positions()
	fc2=readfc2('FORCE_CONSTANTS_2ND')
	f=-np.einsum('ijkl,jl',fc2,u)
	print f-forces
	assert np.allclose(f,forces,atol=1e-3)
@fn_timer
def rotate_fc(rot):

	direct,phi,direct1,phi1=rot
	fc2=readfc2('FORCE_CONSTANTS_2ND')
	#test_fc2()
	fc2=getfc2(direct,phi,fc2)
	fc2=getfc2(direct1,phi1,fc2)
	fc3=np.load('fc3.npy')
	fc3=np.einsum('ijknml->nmlijk',fc3)
	fc3=getfc3(direct,phi,fc3)
	fc3=getfc3(direct1,phi1,fc3)
	return fc2,fc3
@fn_timer
def test_expand_fc():
	
	unit=Atoms('CN',[[0,0,0],[1,0,0]],cell=[2,1,1])
	atoms=Atoms('C3N3',[[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0]],cell=[6,1,1])
	fc=np.zeros([2,6,6,3,3,3])
	"""offset=[2,0,0],[2.0,3.0,4,5,0,1],0=2,1=3,2=4,3=5,4=0,5=1=>s2p=[2,3,4,5,0,1],p2s=[4,5,0,1,2,3]"""
	I=np.eye(3)
	fc[1,3,5]=I
	fc[0,2,3]=I*2
	fc3=expand_fc(fc,atoms,unit)
	assert np.allclose(fc3[1,3,5],I)
	"""fc3[3,5,1]=fc[1,3,5]=fc[p2s[3],p2s[5],p2s[1]]"""
	assert np.allclose(fc3[3,5,1],I)
	assert np.allclose(fc3[2,4,5],2*I)
	assert np.allclose(fc3[4,0,1],2*I)
def getp2s(atoms,offset,s2p=None):
	if s2p is None:
		s2p=gets2p(atoms,offset)
	n=len(s2p)
	x=np.ones(n,dtype=np.int32)
	for i in range(n):
		x[s2p[i]]=i
	return x
def gets2p(atoms,offset):
	a=atoms.copy()
	a.translate(offset)
	psm=premitiveSuperMapper(atoms,a)
	s2p=psm.getS2p()[0]	
	return s2p
def test_gets2p():
	atoms=Atoms('C3N3',[[0,0,0],[1,0,0],[4,0,0],[5,0,0],[2,0,0],[3,0,0]],cell=[6,1,1])
	offset=[2,0,0]
	s2p=gets2p(atoms,offset)
	"""[2.0,3,0,1,4,5],0=4,1=5,2=0,3=1,4=2,5=3"""
	assert np.allclose(s2p,[4,5,0,1,2,3])
	p2s=getp2s(atoms,offset)
	assert np.allclose(p2s,[2,3,4,5,0,1])
	offset=[3,0,0]
	s2p=gets2p(atoms,offset)
	"""[3,4,1,2,5,0],0=5,1=2,2=1,3=4,4=3,5=0"""
	assert np.allclose(s2p,[5,2,1,4,3,0])
	p2s=getp2s(atoms,offset)
	assert np.allclose(p2s,[5,2,1,4,3,0])
@fn_timer
def expand_fc(fc,atoms,unit):

	cellp=unit.cell
	n=len(unit)
	cells=atoms.cell
	m=len(atoms)
	c=(np.linalg.norm(cells,axis=1)/np.linalg.norm(cellp,axis=1)+np.array([0.5,.5,.5])).astype(np.int32)
	fc3=np.zeros([m,m,m,3,3,3])
	import itertools
	for i,j,k in itertools.product(xrange(c[0]),xrange(c[1]),xrange(c[2])):
		print i,j,k
		offset=cellp.dot([i,j,k])
		s2p=gets2p(atoms,offset)
		p2s=getp2s(atoms,offset,s2p)
		x=np.arange(n)
		a=s2p[x]
		y=p2s[np.arange(m)]
		fc3[a]=fc[x][:,y][:,:,y]
	return fc3
@fn_timer
def get_rotated_atoms(atoms):
	unit=atoms.copy()
	#debug(unit.cell)
	direct,phi=mergeVec(unit.cell[2],[0,0,1])
	unit.rotate(direct,phi,rotate_cell=True)
	#debug(unit.cell)
	yn=[unit.cell[1,0],unit.cell[1,1],0]
	direct1,phi1=mergeVec(yn,[0,1,0])
	unit.rotate(direct1,phi1,rotate_cell=True)
	filter=np.abs(unit.cell)<1e-5
	unit.cell[filter]=0.0
	#debug(unit.cell)
	return unit,(direct,phi,direct1,phi1)
@fn_timer
def mergeVec(x,y):
	direct=np.cross(x,y)
	if np.allclose(np.linalg.norm(direct),0):
		direct=[0,0,1]
	else:
		direct=direct/np.linalg.norm(direct)
	phi=np.arccos(np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y))
	return (direct,phi)
		
atoms=io.read('SPOSCAR')
atoms,rot=get_rotated_atoms(atoms)
test_getfc3()
fc2,fc3=rotate_fc(rot)
unit=io.read('POSCAR')
test_gets2p()
test_expand_fc()
fc3=expand_fc(fc3,atoms,unit)
zeropos=atoms.copy()
atoms.set_calculator(FC(zeropos=zeropos,fc2=fc2,fc3=fc3))
T=300.0*units.kB
timestep=.25* units.fs
MaxwellBoltzmannDistribution(atoms, T)
method=4
if(method==1):
	dyn = NPT(atoms,
		timestep= timestep,
		temperature=T,
		externalstress=1,
		pfactor=None,
		ttime=timestep*.01,
		mask=(1,1,1))
if method==2:
	dyn = VelocityVerlet(atoms,	dt= timestep)
if method==3:
	dyn=NVTBerendsen(atoms,timestep=timestep,temperature=T,taut=timestep*.01)
if method==4:
	dyn = Langevin(atoms, timestep=timestep, temperature=T, friction=0.002)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=False,
peratom=True, mode="w"), interval=10)
traj = PickleTrajectory('a.traj', 'w',atoms)
def xx():
	#print atoms.get_velocities()
	print atoms.get_forces()
def printenergy(a=atoms):
	epot = a.get_potential_energy() / len(a)
	ekin = a.get_kinetic_energy() / len(a)
	power=(atoms.get_forces()*atoms.get_velocities()).sum()
	print ("%d\t%.6f\t%.6f\t%3.0f\t%.6f\t%.6f" %(dyn.nsteps,epot, ekin, ekin/(1.5*units.kB), epot+ekin,power))
# Now run the dynamics
print "nstep\tEpot(eV)/N\tEkin(eV)/N\tT(K)\tEtot(eV)/N\tPower"
dyn.attach(printenergy, interval=1)
dyn.attach(traj, interval=1)
dyn.run(2000)
passthru("ase-gui a.traj -o a.xyz ")
