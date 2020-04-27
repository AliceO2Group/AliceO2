This is a simple simulation example showing how to run event simulation using the `STARlight` event generator.
The `STARlight` event generation process is launched in a clean environment by the script `run-starlight.sh`.

The `run-starlight.sh` script performs initialisation operations of the enviroment to run `STARlight`.
It copies in the currect directory the relevant files need for the operation from the `STARlight` installation directory.
The relevant files are
* `slight.in` to configure the event generator
* `starlight2hepmc.awk` and `pdgMass.awk to convert the output in `HepMC2` format.

At the end of the `STARlight` process, the `HepMC2` output file is `startlight.hepmc` and can be used for the `o2` simulation in the second part of this example.

# IMPORTANT
To run this example you need to have the STARlight package installed on your system.
```
$ aliBuild build STARlight --defaults o2
```
Notice that it is not mandatory to have `--defaults o2` given that in this example `STARlight` runs in a completely separate environment than the one used by the `o2` simulation. 

# WARNING
This example takes the `STARlight` configuration file `slight.in` directly from the `STARlight` installation directory. In that file, the number of events to be generated is hardcoded to `N_EVENTS = 1000`. Therefore, the number of events that can be injected in the `o2` is at maximum `NEV = 1000`. Of course this can be taken care in a dynamic way, such that `STARlight` is instructed to generated the number of events that the `o2` expects.

# IMPROVEMENT
This example can be improved by running `STARlight` event generator in background and sending the `HepMC` data into a pipe that is then read by the `o2` simulation. For the sake of simplicity of the example, this is not done.


