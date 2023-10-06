<!-- doxy
\page refrunSimExamplesHepMC Example reading HepMC events
/doxy -->

Here are pointers on how to use `GeneratorHepMC` selected by the
option `-g hepmc` for `o2-sim`.

## Reading HepMC files

The generator `GeneratorHepMC` can read events from a
[HepMC(3)](http://hepmc.web.cern.ch/hepmc/) formatted file.  These
files can be produced by a standalone event generator program (EG).
Examples of such programs are

- [Pythia8](https://pythia.org)
- The [CRMC](https://gitlab.iap.kit.edu/AirShowerPhysics/crmc) suite
- [Herwig](https://herwig.hepforge.org/)
- [SMASH](https://smash-transport.github.io/)
- ... and many others

Please refer to the documentation of these for more on how to make
event files in the HepMC format.

To make a simulation reading from the file `events.hepmc`, do
    o2-sim -g hepmc --configKeyValues "HepMC.fileName=events.hepmc"	...
	
See also [`read.sh`](read.sh).

## Reading HepMC events from child process
`GeneratorHepMC` can not only read HepMC events from a file, but can
also spawn an child EG to produce events.  Suppose we have a program
named `eg` which is some EG that writes HepMC event records to the
standard output.  Then we can execute a simulation using this external
EG by

    o2-sim -g hepmc --configKeyValues "HepMC.progCmd=eg"

See also [`child.sh`](child.sh). 

There are some requirements on the program `eg`:

- It _must_ write the HepMC event structures to standard output
  (`/dev/stdout`).
- It may _not_ write other information to standard output.
- It _must_ accept the option `-n n-events` to set the number of
  events to produce to `n-events`.

If a program does not adhere to these requirements, it will often be
simple enough to make a small wrapper script that enforce this.  For
example, `crmc` will write a lot of information to standard output.
We can filter that out via a shell script ([`crmc.sh`](crmc.sh)) like

    #!/bin/sh
	crmc $@ -o hepmc3 -f /dev/stdout | sed -n 's/^\(HepMC::\|[EAUWVP] \)/\1/p'
	
The `sed` command selects lines that begin with `HepMC::`, or one
of single characters `E` (event), `A` (attribute), `U` (units), `W`
(weight), `V` (vertex), or `P` (particle) followed by a space.  This
should in most cases be enough to filter out extra stuff written on
standard output.

The script above also passes any additional command line options on to
`crmc` via `$@`.  We can utilise this with `o2-sim` to set options to
the CRMC suite.  For example, if we want to simulate p-Pb collisions
using DpmJET, we can do

    o2-sim -g hepmc --configKeyValues "HepMC.progCmd=crmc.sh -m 12 -i2212 -I 1002080820"
	
	
### Implementation details

Internally `GeneratorHepMC`

1. creates a unique temporary file name in the working directory,
2. then creates a FIFO (or named pipe, see
   [Wikipedia](https://en.wikipedia.org/wiki/Named_pipe)),
3. builds a command line, e.g.,

        eg options > fifo-name &

4. and executes that command line

## The future

The `GeneratorHepMC` (and sister generator `GeneratorTParticle`) will
in the not so distant future be upgraded with new functionality to
more easily customise reading files and executing a child process.  In
particular

- HepMC event structures can be read from any file format supported by
  HepMC it self (see
  [here](http://hepmc.web.cern.ch/hepmc/group__IO.html) and
  [here](http://hepmc.web.cern.ch/hepmc/group__factory.html).

- New options that can be specified in `--configKeyValues`

  - `HepMC.eventsToSkip=number` a number events to skip at the
    beginning of each file read.

  - `FileOrCmd.fileNames=list` a comma separated list of HepMC files
    to read.

  - `FileOrCmd.cmd=command line` a command line to execute as a
    background child process.  If this is set (not the empty string),
    then `FileOrCmd.fileNames` is ignored.

  - A number of keys that specifies the command line option switch
    that the child program accepts for certain things.  If any of
    these are set to the empty string, then that switch and
    corresponding option value is not passed to the child program.

    - `FileOrCmd.outputSwitch=switch` (default `>`) to specify output
      file.  The default of `>` assumes that the program write HepMC
      events, and _only_ those, to standard output.

    - `FileOrCmd.seedSwitch=switch` (default `-s`) to specify the
      random number generator seed. The value passed is selected by
      the `o2-sim` option `--seed`

    - `FileOrCmd.bMaxSwitch=switch` (default `-b`) to specify the
       upper limit on the impact parameters sampled.  The value passed
       is selected by the `o2-sim` option `--bMax`

    - `FileOrCmd.nEventsSwitch=switch` (default `-n`) to specify the
       number of events to generate.  The value passed is selected by
       the `o2-sim` option `--nEvents` or (`-n`)

    - `FileOrCmd.backgroundSwitch=switch` (default `&`) to specify how
      the program is put in the background.  Typically this should be
      `&`, but a program may itself fork to the background.

- Some options are no longer available
  - `HepMC.fileName` - use `FileOrCmd.fileNames`
  - `HepMC.progCmd` - use `FileOrCmd.cmd`

The command line build will now be

> _commandLine_ _nEventsSwitch_ _nEvents_ _seedSwitch_ _seed_
> _bMaxSwitch_ _bMax_ _outputSwitch_ _output_ _backgroundSwitch_

If any of the `Switch` keys are empty, then the corresponding option
is not propagated to the command line.  For example, if _bMaxSwitch_
is empty, then the build command line will be

> _commandLine_ _nEventsSwitch_ _nEvents_ _seedSwitch_ _seed_
> _outputSwitch_ _output_ _backgroundSwitch_

