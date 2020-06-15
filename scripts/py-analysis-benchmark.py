import csv
from datetime import datetime
import itertools
import os
import statistics as stat
from string import Template
import subprocess
import timeit

O2_ROOT = os.environ.get('O2_ROOT')
if not O2_ROOT:
    print('O2_ROOT not found')
    raise ValueError
    
INPUT_FILE = '@filelist.txt'
OUTPUT_CSV = 'benchmark_data.csv' 

with open(INPUT_FILE[1:],'r') as f:
    fnames = f.readlines()
    input_size = round(sum([os.stat(l.strip('\n')).st_size for l in fnames])/1024/1024)
    input_length = len(fnames)

NCORES = [1, 2, 4]
NREADERS = [1, 2, 4]
NTRIALS = 3

CPU_SELECTION = False 

SHA256SUM_TASK = Template('cat ${file_list} | xargs -P ${n} -n1 -I{} sha256sum {}')

#COMPOSITE_TASK = Template('o2-analysis-trackselection -b --pipeline track-selection:${n},track-extension:${n} --aod-file ${file_list} --readers ${n} | o2-analysistutorial-histogram-track-selection -b --pipeline histogram-track-selection:${n} --select=0')

BENCHMARK_TASKS = {
      'o2-analysistutorial-histograms': '-b --pipeline eta-and-phi-histograms:${n},pt-histogram:${n},etaphi-histogram:${n}',
      'o2-analysis-trackselection': '-b --pipeline track-selection:${n},track_extension:${n}',
      #'o2-analysis-vertexing-hf': '-b --pipeline vertexerhf-candidatebuildingDzero:${n},vertexerhf-decayvertexbuilder2prong:${n}',
    }
        
for k in BENCHMARK_TASKS:
    BENCHMARK_TASKS[k] = Template(BENCHMARK_TASKS[k])
    
with open(OUTPUT_CSV, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['tname', 'ncores', 'nreaders', 'time_mean', 'time_std', 'input_size', 'input_length'])
    
    for ncores in NCORES:
        cmd_sha256sum = SHA256SUM_TASK.substitute(file_list=INPUT_FILE[1:], n=str(ncores))
        t = timeit.Timer('os.system(cmd_sha256sum)', globals=globals())
        t_res = t.repeat(NTRIALS, 1)
        writer.writerow( ('sha256sum', ncores, -1, stat.mean(t_res), stat.stdev(t_res), input_size, input_length) )
    
    for ncores, nreaders in itertools.product(NCORES, NREADERS):
        
        #cmd_composite = COMPOSITE_TASK.substitute(file_list=INPUT_FILE,n=str(ncores))
        #t = timeit.Timer('os.system(cmd_composite)', globals=globals())
        #t_res = t.repeat(NTRIALS, 1)
        #writer.writerow( ('analysistutorial-histogram-track-selection', ncores, nreaders, stat.mean(t_res), stat.stdev(t_res), input_size, input_length) )
    
        for tname, targ in BENCHMARK_TASKS.items():
            targ = targ.substitute(n=str(ncores))
            cmd_list = [tname] + targ.split(' ')
            
            if CPU_SELECTION:
                if ncores == 2:
                    cmd_list = ['taskset','-c','5,15'] + cmd_list
                elif ncores == 4:
                    cmd_list = ['taskset','-c','1,3,11,13'] + cmd_list

            cmd_list += ['--aod-file', INPUT_FILE]
            cmd_list += ['--readers', str(nreaders)]
            
            t = timeit.Timer('subprocess.run(cmd_list)', globals=globals())
            t_res = t.repeat(NTRIALS, 1)
            writer.writerow( (tname[3:], ncores, nreaders, stat.mean(t_res), stat.stdev(t_res), input_size, input_length) )

#alinsure
#numa0 0-11,24-35
#numa1 12-23,36-47
