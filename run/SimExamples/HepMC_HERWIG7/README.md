<!-- doxy
\page refrunSimExamplesHepMC_HERWIG7 Example HepMC_HERWIG7
/doxy -->

The usage of HERWIG7 with the O2 machinery is presented in this short manual.
The example generates HEPMC3 data using the Herwig executable and then
reads the data via the hepmc generator defined in o2-sim.

# Files description

Two files are provided in the folder:
- **runo2sim.sh** &rarr; allows the generation of events using o2-sim
- **LHC.in** &rarr; example input file for the configuration of the HERWIG generator

## runo2sim.sh

The script works after loading any O2sim version containing HERWIG7 as a package (dependence of AliGenO2).

If no parameters are provided, the script will run with default values (energy and nEvents provided in the LHC.in file), but few flags are available to change the settings on-the-fly:
- **-m , --more** &rarr; feeds the simulation with advanced parameters provided to the configuration key flags
- **-n , --nevents** &rarr; changes the number of events in the .in file or gets the one in the file if no events are provided
- **-i , --input** &rarr; .in filename for HERWIG7 configuration
- **-j , --jobs** &rarr; sets the number of workers (2 jobs by default)
- **-e , --ecm** &rarr; sets the center-of-mass energy in the input file
- **-h , --help** &rarr; prints usage instructions