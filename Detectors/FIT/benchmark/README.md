<!-- doxy
\page refFITbenchmark Performace testing
/doxy -->

# Documentation for Performance testing
This document will summarize the tools that can be used to get information about the memory and CPU time evolution of simulations in ALICE O2.

In this folder you will find two scripts:
1. `monitor.sh`
2. `process.py`

Both of these scripts define the two step procedure (**monitoring** and **processing**) of obtainging performance metrics of interest: _maximum memory, average memory, maximum CPU time, wall clock time, (CPU)/(wall clock) time ratio_ and lastly _plots of the evolving memory and cpu as a function of wall clock time._

## 1) Monitoring 
You can monitor whatever you like as:

`$> ./monitor.sh <your o2 command>`

To obtain plots of (FairMQ) devices operating in the simulation you will have to generate a **logfile**  as:

`$> ./monitor.sh <your o2 command> | tee o2xxx.log`

e.g. if you wish to monitor 50 pp (pythia) events with Geant3 as the VMC backend using the FIT detector and utilizing parallel mode with 2 simulation workers AND keep track of FairMQ devices, you can do:

`$> ./monitor.sh o2-sim -g pythia8pp -e TGeant3 -m FV0 FT0 FDD -j 2 -n 50 | tee o2sim.log`

---

Similarly you can monitor the digitization routine as: 

`$> ./monitor.sh o2-sim-digitizer-workflow -b --run | tee  o2digi.log`

NB! notice the `--run` that is needed (only digitization) in order to overload the PIPE `|` command in DPL (Data Processing Layer).

The `./monitor.sh` script will generate 4 .txt files in total: _mem_evolution_xxxx.txt, cpu_evolution_xxxx.txt, time_evolution_xxxx.txt, pid_evolution_xxxx.txt_. Here _xxxx_ is the PID (process identifcation) number of the main process responsible for the command (driver application). You will have to parse two of these files (mem and cpu) in the next step.

## 2) Processing
The monitored data has to be processed as: 
`$> python3 process.py mem_evolution_xxxx.txt cpu_evolution_xxxx.txt`

This will generate an output: 

```Your command was:  o2-sim -g pythia8pp -e TGeant3 -m FV0 FT0 FDD -j 2 -n 50
You have monitored o2 simulation in parallel.

********************************
max mem: 723.30 MB
mean mem: 544.63 MB
max cpu: 120.69s
Total wall clock time: 82.54 s
Ratio (cpu time) / (wall clock time) :  1.46
********************************
```
and generate two plots each:

![alt text](https://github.com/AliceO2Group/AliceO2/tree/dev/Detectors/FIT/benchmark/images/Figure_1.png)
![alt text](https://github.com/AliceO2Group/AliceO2/tree/dev/Detectors/FIT/benchmark/images/Figure_2.png)

if no logfiles where provided the plots would look like: 

![alt text](https://github.com/AliceO2Group/AliceO2/tree/dev/Detectors/FIT/benchmark/images/Figure_1_nolog.png)
![alt text](https://github.com/AliceO2Group/AliceO2/tree/dev/Detectors/FIT/benchmark/images/Figure_2_nolog.png)
