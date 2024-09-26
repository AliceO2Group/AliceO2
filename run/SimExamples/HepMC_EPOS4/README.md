<!-- doxy
\page refrunSimExamplesHepMC_EPOS4 Example HepMC_EPOS4
/doxy -->

The usage of EPOS4 with the O2 machinery is presented in this short manual. 
An in-depth explanation of the mechanisms behind the HepMC(3) data handling can be found in the 
HepMC_fifo folder of the MC examples.  
The scripts use the `cmd` parameter of `GeneratorHepMC` to spawn the EPOS4 generation
via the `epos.sh` script.  
EPOS4 uses the outdated HepMC2 libraries, so this had to be specified in the steering scripts
of the generators configuration. If `HepMC.version=2` is removed then the scripts will not work 
anymore. This is to say that the balance achieved with the configurations provided is easily 
destroyed if the user base edits parts that are not understood completely. 

# Scripts description

Four scripts are available to run the simulations
- **epos.sh** &rarr; starts the actual EPOS4 generation 
- **runo2sim.sh** &rarr; allows the generation of events using o2-sim
- **rundpg.sh** &rarr; starts the DPG machinery for event generation
- **rundpl.sh** &rarr; starts the event generation with DPL

In addition an example.optns file is provided to start EPOS4, but more can be found in the generator example folder 
or in the website [epos4learn.web.cern.ch](https://epos4learn.docs.cern.ch/) where an extensive tutorial on the generator is provided. 

## epos.sh

It can be run without the help of the other scripts to simply generate an .hepmc file or print 
to the stdout the HepMC results. It it worth nothing though that EPOS4 must be loaded (via cvmfs through AliGenerators or O2sim for example) or installed. In this case the user should simply redirect the stdout to a file:
```
./epos.sh -i test -s 234345 > test.hepmc
```
This example shows all the functionalities of the script (which are implemented in a similar way inside
the generation steering scripts). In particular the `-i` flag allows to provide .optns parameters to EPOS4, 
`-s` feeds the generator with a user seed, and the HepMC output is given by test.hepmc by redirecting the 
stdout which will contain only the HepMC data thanks to the `-hepstd` flag set automatically in epos.sh and 
the `set ihepmc 2` option which **MUST** be set in the option file (otherwise either an hepmc file will be created - ihepmc 1 - or nothing will be generated - missing ihepmc or != 1|2 ). 
It is important to note that setting an empty/null seed in the generator out of the box makes EPOS4 crash, so a protection was added in our steering epos.sh script which now generates a random number if 0 is provided. 

## runo2sim.sh, rundpg.sh and rundpl.sh

The three scripts have little differences (especially in the first part), so they will be described together.  
They work after loading any O2sim version after the 20/09/2024 (included), since multiple modifications had to be performed on both EPOS4 and the introduction of AliGenO2 in order to be able to load both O2sim and EPOS4 simultaneously. 
If no parameters are provided to the scripts, they will run with default values (energy and nevents provided in the example.optns file), but few flags are available to change the settings of the generation: 
- **-m , --more** &rarr; feeds the simulation with advanced parameters provided to the configuration key flags
- **-n , --nevents** &rarr; changes the number of events in the .optns file or gets the one in the file if no events are provided
- **-i , --input** &rarr; .optns filename to feed EPOS4, no extension must be set in the filename
- **-j , --jobs** &rarr; sets the number of workers (jobs)
- **-h , --help** &rarr; prints usage instructions
- **-e , --ecm** &rarr; sets the center-of-mass energy in the options file

In the `rundpg.sh` script an additional flag is available
- **-t , --tf** &rarr; number of timeframes to be generated  

In this case the options file will be copied in each tf$n folder, otherwise the epos script won't be able to run with multiple timeframes.  
In o2sim and DPG scripts the randomly generated seed is set directly, instead this is not feasible with the DPL one, given that the --seed option is not able 
to redirect this number to GeneratorHepMC. So a seed 0 is automatically given to epos.sh which generates a random number in return.  
Now the three scripts start to differ:

- **runo2sim.sh** &rarr; o2-sim is launched
- **rundpg.sh** &rarr; first the o2dpg_sim_workflow.py script will be launched generating the json configuration, then the o2_dpg_workflow_runner.py script will start the workflow
- **rundpl.sh** &rarr; o2-sim-dpl-eventgen is executed piping its results to o2-sim-mctracks-to-aod and afterwards to o2-analysis-mctracks-to-aod-simple-task

The last few lines of the scripts contain the execution of o2-sim, DPG worflow creator/runner and DPL software respectively, so this part can be modified by the users following their requirements. It's important not to delete from the configuration keys `GeneratorFileOrCmd.cmd=$cmd -i $optns;GeneratorFileOrCmd.bMaxSwitch=none;HepMC.version=2;` 
and it would be better to provide additional configurations via the -m flag. EPOS4 cannot set a maximum impact parameter value, so it's better to leave the bMaxSwitch to none, while the others serve the sole purpose of running successfully the generator using auto generated FIFOs. 


