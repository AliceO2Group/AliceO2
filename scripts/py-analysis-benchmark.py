"""
Script for the local benchmarking of the o2 analysis tasks,
running them with multiple processing jobs (NCORES)
and multiple readers (NREADERS) over input files (INPUT_FILE).
Tasks to be benchmarked are in the BENCHMARK_TASKS dict.

Usage: python3 py-analysis-benchmark.py

Ouput: CSV file (OUTPUT_CSV) with benchmarking results:
'tname', 'ncores', 'nreaders', 'time_mean' (s), 'time_std' (s), 
'input_size' (MB), 'input_length', 'timestamp', 'cpu_load', 'ncpu', 'machine'
"""

import csv
from datetime import datetime
import itertools
import os
import statistics as stat
from string import Template
import subprocess
import timeit


def get_cl_output(cmd) -> str:
    try:
        output_ = str(subprocess.check_output(cmd, shell=True), 'utf-8')
    except subprocess.CalledProcessError:
        output_ = ''
    return output_.strip('\n')


def get_cpu_load():
    uptime_ = get_cl_output('uptime')
    load_last_15 = uptime_.split(' ')[-1]
    return load_last_15


def get_timestamp():
    return datetime.now().strftime("%m/%d/%Y %H:%M")
    
    
def get_time_std(t_res):
    try:
        std_ = stat.stdev(t_res)
    except stat.StatisticsError:
        std_ = -1
    return std_


#benchmarking setup    
INPUT_FILE = '@filelist.txt'
OUTPUT_CSV = 'benchmark_data.csv' 
NCORES = [1, 2, 4]
NREADERS = [1, 2, 4]
NTRIALS = 2
LARGE_SHM_SEGMENT_SIZE = False
CPU_SELECTION = False

#tasks to be benchmarked
BENCHMARK_TASKS = {
      'o2-analysistutorial-void': '-b --pipeline void:${n}',
      'o2-analysistutorial-histograms': '-b --pipeline eta-and-phi-histograms:${n},pt-histogram:${n},etaphi-histogram:${n}',
      'o2-analysis-trackselection': '-b --pipeline track-selection:${n},track_extension:${n}',
      'o2-analysis-correlations': '-b --pipeline correlation-task:${n}',
      #'o2-analysis-vertexing-hf': '-b --pipeline vertexerhf-candidatebuildingDzero:${n},vertexerhf-decayvertexbuilder2prong:${n}'
    }


O2_ROOT = os.environ.get('O2_ROOT')
if not O2_ROOT:
    print('O2_ROOT not found')
    raise ValueError 
    
MACHINE = get_cl_output('hostname')
NCPU = get_cl_output('grep processor /proc/cpuinfo | wc -l')
with open(INPUT_FILE[1:],'r') as f:
    fnames = f.readlines()
    INPUT_SIZE = round(sum([os.stat(l.strip('\n')).st_size for l in fnames])/1024/1024)
    INPUT_LENGTH = len(fnames)
    

SHA256SUM_TASK = Template('cat ${file_list} | xargs -P ${n} -n1 -I{} sha256sum {}')
#COMPOSITE_TASK = Template('o2-analysis-trackselection -b --pipeline track-selection:${n},track-extension:${n} --aod-file ${file_list} --readers ${n} | o2-analysistutorial-histogram-track-selection -b --pipeline histogram-track-selection:${n} --select=0')

for k in BENCHMARK_TASKS:
    BENCHMARK_TASKS[k] = Template(BENCHMARK_TASKS[k])
    
with open(OUTPUT_CSV, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(('tname', 'ncores', 'nreaders', 'time_mean', 'time_std', 
                     'input_size', 'input_length', 'timestamp', 'cpu_load', 'ncpu', 'machine'))
    
    for ncores in NCORES:
        cmd_sha256sum = SHA256SUM_TASK.substitute(file_list=INPUT_FILE[1:], n=str(ncores))
        t = timeit.Timer('os.system(cmd_sha256sum)', globals=globals())
        t_res = t.repeat(NTRIALS, 1)
        writer.writerow( ('sha256sum', ncores, -1, stat.mean(t_res), get_time_std(t_res), 
                          INPUT_SIZE, INPUT_LENGTH, get_timestamp(), get_cpu_load(), NCPU, MACHINE) )
    
    for ncores, nreaders in itertools.product(NCORES, NREADERS):

        for tname, targ in BENCHMARK_TASKS.items():
            targ = targ.substitute(n=str(ncores))
            cmd_list = [tname] + targ.split(' ')
            
            if CPU_SELECTION:
                if ncores == 2:
                    cmd_list = ['taskset','-c','5,15'] + cmd_list
                elif ncores == 4:
                    cmd_list = ['taskset','-c','1,3,11,13'] + cmd_list
            
            if LARGE_SHM_SEGMENT_SIZE:
                cmd_list += ['--shm-segment-size', str(16000000000)]

            cmd_list += ['--aod-file', INPUT_FILE]
            cmd_list += ['--readers', str(nreaders)]
            
            t = timeit.Timer('subprocess.run(cmd_list)', globals=globals())
            t_res = t.repeat(NTRIALS, 1)
            writer.writerow( (tname[3:], ncores, nreaders, stat.mean(t_res), get_time_std(t_res), 
                              INPUT_SIZE, INPUT_LENGTH, get_timestamp(), get_cpu_load(), NCPU, MACHINE) )

#alinsure
#numa0 0-11,24-35
#numa1 12-23,36-47
