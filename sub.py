# -*- coding: utf-8 -*-
# @Author: YangZhou
# @Date:   2016-11-09 22:18:07
# @Last Modified by:   YangZhou
# @Last Modified time: 2016-11-21 22:25:37
from ase import io 
from ase.calculators.lj import LennardJones
from fc import FC
from ase import units
import numpy as np
from ase.md.npt import NPT
from ase.md.verlet import VelocityVerlet
#from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.io.trajectory import PickleTrajectory
from aces.f import fn_timer
from aces.tools import passthru
@fn_timer
def rotate_fc(rot):
	from aces.f import rotatefc2,rotatefc3
	direct,phi,direct1,phi1=rot
	rotatefc2(file2='tmp',direct=direct,t=phi)
	rotatefc2(file1='tmp',direct=direct1,t=phi1)
	rotatefc3(file2='tmp',direct=direct,t=phi)
	rotatefc3(file1='tmp',direct=direct1,t=phi1)
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
		
atoms=io.read('POSCAR')
atoms,rot=get_rotated_atoms(atoms)
rotate_fc(rot)
atoms=atoms.repeat([2,2,2])
zeropos=atoms.copy()
refatoms2,rot=get_rotated_atoms(io.read("SPOSCAR_2ND"))
refatoms3,rot=get_rotated_atoms(io.read("SPOSCAR_3RD"))
atoms.set_calculator(FC(zeropos=zeropos,refatoms2=refatoms2,refatoms3=refatoms3))
T=300.0
timestep=25
MaxwellBoltzmannDistribution(atoms, T*units.kB)
"""
dyn = NPT(atoms,
	timestep= timestep* units.fs,
	temperature=T*units.kB,
	externalstress=0,
	pfactor=None,
	ttime=timestep*.01*units.fs,
	mask=(0,0,0))
	"""
dyn = VelocityVerlet(atoms,
	dt= timestep* units.fs)
#dyn=NVTBerendsen(atoms,25 * units.fs,100.0*units.kB,0.25*units.fs)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=False,
peratom=True, mode="w"), interval=10)
traj = PickleTrajectory('a.traj', 'w',atoms)
def xx():
	#print atoms.get_velocities()
	print atoms.get_forces()
def printenergy(a=atoms):
	epot = a.get_potential_energy() / len(a)
	ekin = a.get_kinetic_energy() / len(a)
	print ("%d\t%.6f\t%.6f\t%3.0f\t%.6f" %(dyn.nsteps,epot, ekin, ekin/(1.5*units.kB), epot+ekin))
# Now run the dynamics
print "nstep\tEpot(eV)/N\tEkin(eV)/N\tT(K)\tEtot(eV)/N"
dyn.attach(printenergy, interval=1)
dyn.attach(traj, interval=1)
dyn.run(2000)
passthru("ase-gui a.traj -o a.xyz ")
