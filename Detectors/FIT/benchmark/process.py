# load modules
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# use classic plot style
plt.style.use('classic')

# read and save user input filenames
mem_filename = sys.argv[1]
cpu_filename = sys.argv[2]

# save the process id names
process_id_mem = re.findall('mem_evolution_(\\d+)', mem_filename)[0]
process_id_cpu = re.findall('cpu_evolution_(\\d+)', cpu_filename)[0]

# check that the process id names are the same
if not process_id_mem==process_id_cpu:
    # throw error if true and exit program
    sys.stderr.write("The memory and cpu process filenames do not match...\n")
    print("input memory filename: ",mem_filename)
    print("inpu cpu filename: ",cpu_filename)
    exit(1)

# save the main process id (driver application)
process_id = process_id_mem + '.txt' # as string '<PID>.txt'

# save the same process id (driver application), but as a float
driver = float(process_id_mem)

# load the o2 command given
with open(mem_filename) as f:
    title = f.readline()

# extract the command given
title = re.findall('#command line: (\\w.+)', title)[0]

# declare string variables for different runs
simulation = 'o2-sim '
serial = 'o2-sim-serial'
digitization = 'o2-sim-digitizer-workflow'

# print the command for the user
print("\nYour command was: ", title)

# check what type of command and parse it to a logfile variable
if title.find(simulation) == 0:
    print("You have monitored o2 simulation in parallel.\n")
    command=simulation
    logfilename = 'o2sim.log'

elif title.find(serial) == 0:
    print("You have monitored o2 simulation in serial.\n")
    command=serial
    logfilename = 'o2sim.log'

elif title.find(digitization) == 0:
    command=digitization
    print("You have monitored o2 digitization.\n")
    logfilename = 'o2digi.log'

else :
    print("I do not know this type of simulation.\n")
    exit(1)

#################################################
#                                               #
#       Extract the PIDs from logfile           #
#                                               #
#################################################

if command==simulation: # True if you typed o2-sim
    
    try:
        # open o2sim.log file name
        with open(logfilename) as logfile:
            # read and save the first 6 lines in o2sim.log
            loglines = [next(logfile) for line in range(6)]

    #    print("*******************************\n")
    #    print("Driver application PID is: ", driver)

        # find the PID for the event generator (o2-sim-primary-..)
        eventgenerator_line = re.search('Spawning particle server on PID (.*); Redirect output to serverlog\n',loglines[3])
        event_gen = float(eventgenerator_line.group(1))
    #    print("Eventgenerator PID is: ", event_gen)

        # find the PID for sim worker 0 (o2-sim-device-runner)
        sim_worker_line = re.search('Spawning sim worker 0 on PID (.*); Redirect output to workerlog0\n',loglines[4])
        sim_worker = float(sim_worker_line.group(1))
    #    print("SimWorker 0 PID is: ", sim_worker)

        # find the PID for the hitmerger (o2-sim-hitmerger)
        hitmerger_line = re.search('Spawning hit merger on PID (.*); Redirect output to mergerlog\n',loglines[5])
        hit_merger = float(hitmerger_line.group(1))
    #    print("Hitmerger PID is: ", hit_merger, "\n")
    #    print("*******************************\n")

        # find the number of simulation workers
        n_workers = int(re.findall('Running with (\\d+)', loglines[1])[0])

        # save into a list
        pid_names = ['driver','event gen','sim worker 0','hit merger']
        pid_vals = [driver,event_gen,sim_worker,hit_merger]

        # append pid names for remaining workers
        for i in range(n_workers-1):
            pid_names.append(f"sim worker {i+1}")
        
        no_log = False
    
    except IOError:
        print("There exists no o2sim.log..")
        print("No details of devices will be provided.")
        no_log = True

elif command==digitization: # True if you typed o2-sim-digitizer-workflow

    try:
        # open o2digi.log file name
        with open(logfilename) as logfile:

            # save the first 100 lines in o2digi.log
            loglines = [next(logfile) for line in range(100)]

        # declare list for PID numbers and names
        pid_vals = []
        pid_names = []

        # loop through lines to find PIDs
        for line_num,line in enumerate(loglines):
            pid_line = re.findall('Starting (\\w.+) on pid (\\d+)',line)
            if pid_line: # True if the line contains 'Start <PID name> on pid <PID number>'

                # assign the name and value to variables
                pid_name = pid_line[0][0]
                pid_val = float(pid_line[0][1])

                # save to list
                pid_names.append(pid_name)
                pid_vals.append(pid_val)

        # insert driver application name and value
        pid_names.insert(0,'driver')
        pid_vals.insert(0,driver)

    #    for id in range(len(pid_names)):
    #        print(pid_names[id],"PID is: ",pid_vals[id])
    #        print(pid_vals[pid])
    #    print("*******************************\n")
        no_log = False
        
    except IOError:
        print("There exists no o2digi.log..")
        print("No details of devices will be provided.")
        no_log = True
    

elif command==serial:
    print("*******************************\n")
    print("Driver application PID is: ", driver)
    print("There are no other PIDs")
    no_log = False

else :
    print("Something went wrong.. exiting")
    exit(1)

############### End of PID extraction #################

# get time and PID filenames
time_filename = 'time_evolution_' + process_id
pid_filename = 'pid_evolution_' + process_id

# load data as pandas DataFrame (DataFrame due to uneven number of coloumns in file)
mem = pd.read_csv(mem_filename, skiprows=2, sep=" +", engine="python",header=None)
cpu = pd.read_csv(cpu_filename, skiprows=2, sep=" +", engine="python",header=None)
pid = pd.read_csv(pid_filename, skiprows=2, sep=" +", engine="python",header=None)
t = np.loadtxt(time_filename) # time in ms (mili-seconds)

# extract values from the DataFrame
mem = mem[1:].values
cpu = cpu[1:].values
pid = pid[1:].values

# process time series
t = t-t[0] # rescale time such that t_start=0
t = t*10**(-3) # convert mili-seconds to seconds

# replace 'Nones' (empty) elements w/ zeros and convert string values to floats
mem = np.nan_to_num(mem.astype(np.float))
cpu = np.nan_to_num(cpu.astype(np.float))
pid = np.nan_to_num(pid.astype(np.float))

# find all process identifaction numbers involved (PIDs), the index of their first
# occurence (index) for an unraveled array and the total number of apperances (counts) in the process
PIDs, index, counts = np.unique(pid,return_index=True,return_counts=True)

# NOTE: we don't want to count 'fake' PIDs. These are PIDs that spawns only once not taking
# any memory or cpu. Due to their appearence they shift the colomns in all monitored files.
# This needs to be taken care of and they are therefore deleted from the removed.

# return the index of the fake pids
fake = np.where(counts==1)

# delete the fake pids from PIDs list
PIDs = np.delete(PIDs,fake)
index = np.delete(index,fake)
counts = np.delete(counts,fake)

# we also dele PID=0, as this is not a real PID
PIDs = np.delete(PIDs,0)
index = np.delete(index,0)
counts = np.delete(counts,0)

# get number of real PIDs
nPIDs = len(PIDs)

# dimension of data
dim = pid.shape # could also use from time series
# NOTE: dimensiton is always (n_steps, 40)
# because of '#' characters in ./monitor.sh

# number of steps in simulation for o2-sim
steps = len(pid[:,0]) # could also use from time series

# declare final lists
m = [] # memory
c = [] # cpu
p = [] # process

for i in range(nPIDs): # loop through all valid PIDs

    # find the number of zeros to pad with
    init_zeros, _ = np.unravel_index(index[i],dim)

    # pad the 'initial' zeros (begining)
    mem_dummy = np.hstack((np.zeros(init_zeros),mem[pid==PIDs[i]]))
    cpu_dummy = np.hstack((np.zeros(init_zeros),cpu[pid==PIDs[i]]))
    pid_dummy = np.hstack((np.zeros(init_zeros),pid[pid==PIDs[i]]))

    # find the difference in final steps
    n_diff = steps - len(mem_dummy)

    # pad the ending w/ zeros
    mem_dummy = np.hstack((mem_dummy,np.zeros(n_diff)))
    cpu_dummy = np.hstack((cpu_dummy,np.zeros(n_diff)))
    pid_dummy = np.hstack((pid_dummy,np.zeros(n_diff)))

    # save to list
    m.append(mem_dummy)
    c.append(cpu_dummy)
    p.append(pid_dummy)

    #print("PID is: ",PIDs[i])
    #print("initial number of zeros to pad: ", init_zeros)
    #print("final number of zeros to pad: ", n_diff)
    #print("**************\n")

# convert to array and assure correct shape of arrays
m = np.asarray(m).T
c = np.asarray(c).T
p = np.asarray(p).T

###################################
#                                 #
#       COMPUTATIONS              #
#                                 #
###################################

print("********************************")

# compute average memory and maximum memory
M = np.sum(m,axis=1) # sum all processes memory
max_mem = np.max(M) # find maximum
mean_mem = np.mean(M) # find mean
print(f"max mem: {max_mem:.2f} MB")
print(f"mean mem: {mean_mem:.2f} MB")

C = np.sum(c,axis=1) # compute total cpu
max_cpu = np.max(C)
print(f"max cpu: {max_cpu:.2f}s")

# print total wall clock time
wall_clock = t[-1]
print(f"Total wall clock time: {wall_clock:.2f} s")

# print ratio
ratio = np.max(C)/t[-1]
print(f"Ratio (cpu time) / (wall clock time) :  {ratio:.2f}")

print("********************************")

###################################
#                                 #
#           PLOTTING              #
#                                 #
###################################

if no_log: # True if user hasn't provided logfiles
    
    # plot of total, max and mean memory
    fig,ax = plt.subplots(dpi=125,facecolor="white")
    ax.plot(t,M,'-k',label='total memory');
    ax.hlines(np.mean(M),np.min(t),np.max(t),color='blue',linestyles='--',label='mean memory');
    ax.hlines(np.max(M),np.min(t),np.max(t),color='red',linestyles='--',label='max memory');
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Memory [MB]")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(prop={'size': 10},loc='best')
    ax.grid();

    # plot of total, max and mean CPU
    fig1,ax1 = plt.subplots(dpi=125,facecolor="white")
    ax1.plot(t,C,'-k',label='total cpu');
    ax1.hlines(np.mean(C),np.min(t),np.max(t),color='blue',linestyles='--',label='mean cpu');
    ax1.hlines(np.max(C),np.min(t),np.max(t),color='red',linestyles='--',label='max cpu');
    ax1.set_title(title)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("CPU [s]")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.legend(prop={'size': 10},loc='best');
    ax1.grid()

    plt.show();

else : # details about the PID exists (from logfiles)
    
#    # convert to pid info lists to arrays
#    pid_vals = np.asarray(pid_vals)
#    pid_names = np.asarray(pid_names)
#
#    # be sure of the correct ordering of pids
#    pid_placement = np.where(pid_vals==PIDs)

    # plot memory
    fig,ax = plt.subplots(dpi=125,facecolor="white")
    ax.plot(t,m);

    # some features for the plot
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Memory [MB]")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(pid_names,prop={'size': 10},loc='best')
    ax.grid();

    # plot cpu
    fig1,ax1 = plt.subplots(dpi=125,facecolor="white")
    ax1.plot(t,c);

    # some features for the plot
    ax1.set_title(title)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("CPU [s]")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.legend(pid_names,prop={'size': 10},loc='best');
    ax1.grid()

    plt.show();
