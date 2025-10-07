import os
import json
import numpy as np

# creates slurm script mom.sub
def create_slurm(p, filename, call_function):
    # p - dictionary with parameters
    if p['mem'] < 1:
        mem = str(int(p['mem']*1000))+'MB'
    else:
        mem = str(p['mem'])+'GB'
    
    lines = [
    '#!/bin/bash',
    '#SBATCH --nodes='+str(p['nodes']),
    '#SBATCH --ntasks-per-node='+str(p['ntasks']),
    '#SBATCH --cpus-per-task=1',
    '#SBATCH --mem='+mem,
    '#SBATCH --time='+str(p['time'])+':00:00',
    '#SBATCH --begin=now+'+str(p['begin']),
    '#SBATCH --job-name='+str(p['name']),
    '#SBATCH --export=NONE',

    call_function,
    
    'module purge',
    'source ~/MOM6-examples/build/intel/env',
    'module list',
    'for e in $(env | egrep ^SLURM_ | cut -d= -f1); do unset ${e}; done',
    'mpiexec --bind-to none -np ' + str(p['ntasks']) + ' env LD_LIBRARY_PATH=${LD_LIBRARY_PATH} ' + p['executable'],

    'mkdir -p output',
    'mv *.nc output'
    ]
    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def create_MOM_override(p, filename):
    # p - dictionary of parameters
    lines = []
    for key in p.keys():
        lines.append('#override '+key+' = '+str(p[key]))
    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def run_experiment(folder, hpc, parameters, call_function=''):
    if os.path.exists(folder):
        print('Folder '+folder+' already exists. We skip it')
        return
    os.system('mkdir -p '+folder)
    
    create_slurm(hpc, os.path.join(folder,'mom.sub'), call_function)
    create_MOM_override(parameters, os.path.join(folder,'MOM_override'))
    
    os.system('cp -r /home/pp2681/MOM6-examples/build/configurations/double_gyre/* '+folder)

    with open(os.path.join(folder,'args.json'), 'w') as f:
        json.dump(parameters, f, indent=2)
    
    
    os.system('cd '+folder+'; sbatch mom.sub')
    # print('Run experiment yourself in folder', folder)

#########################################################################################
class dictionary(dict):  
    def __init__(self, **kw):  
        super().__init__(**kw)
    def add(self, **kw): 
        d = self.copy()
        d.update(kw)
        return dictionary(**d)
    def __add__(self, d):
        return self.add(**d)
    
def configuration(exp='R4'):
    if exp=='R2':
        return dictionary(
            NIGLOBAL=44,
            NJGLOBAL=40,
            DT=2160.,
            DT_FORCING=2160.
        )

    if exp=='R4':
        return dictionary(
            NIGLOBAL=88,
            NJGLOBAL=80,
            DT=1080.,
            DT_FORCING=1080.
        )

HPC = dictionary(
    nodes=1,
    ntasks=1,
    mem=0.5,
    time=24,
    name='mom6',
    begin='0hour'
)  

PARAMETERS = dictionary(
    DAYMAX=7300.0,
    RESTINT=1825.0,
    LAPLACIAN='False',
    BIHARMONIC='True',
    SMAGORINSKY_AH='True',
    SMAG_BI_CONST=0.06, 
    USE_ZB2020='False',
    ZB_SCALING=1.,
    ZB_TRACE_MODE=0, 
    ZB_SCHEME=1, 
    VG_SHARP_PASS=0,
    VG_SHARP_SEL=1,
    STRESS_SMOOTH_PASS=0,
    STRESS_SMOOTH_SEL=1,
    U_TRUNC_FILE = 'U_velocity_truncations',
    V_TRUNC_FILE = 'V_velocity_truncations'
) + configuration('R4')