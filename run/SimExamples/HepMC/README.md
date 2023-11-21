<!-- doxy
\page refrunSimExamplesHepMC Example reading HepMC events
/doxy -->

Here are pointers on how to use `GeneratorHepMC` selected by the
option `-g hepmc` for `o2-sim`.

HepMC event structures can be read from any file format supported by
HepMC it self (see
[here](http://hepmc.web.cern.ch/hepmc/group__IO.html) and
[here](http://hepmc.web.cern.ch/hepmc/group__factory.html).

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

    o2-sim -g hepmc --configKeyValues "GeneratorFileOrCmd.fileNames=events.hepmc" ...

See also [`read.sh`](read.sh).

## Reading HepMC events from child process

`GeneratorHepMC` can not only read HepMC events from a file, but can
also spawn an child EG to produce events.  Suppose we have a program
named `eg` which is some EG that writes HepMC event records to the
standard output.  Then we can execute a simulation using this external
EG by

    o2-sim -g hepmc --configKeyValues "GeneratorFileOrCmd.cmd=eg"

See also [`child.sh`](child.sh).

There are some requirements on the program `eg`:

- The EG program _must_ be able to write the HepMC event structures to
  a specified file.  The option passed to the program is specified via
  the key `GeneratorFileOrCmd.outputSwitch`.  This defaults to `>`
  which means the EG program is assumed to write the HepMC event
  structures to standard output, _and_ that nothing else is printed on
  standard output.
- It _must_ accept an option to set the number of events to generate.
  This is controlled by the configuration key
  `GeneratorFileOrCmd.nEventsSwitch` and defaults to `-n`.  Thus, the
  EG application should accept `-n 10` to mean that it should generate
  `10` events, for example.
- The EG application should accept a command line switch to set the
  random number generator seed.  This option is specified via the
  configuration key `GeneratorFileOrCmd.seedSwitch` and defaults to
  `-s`.  Thus, the EG application must accept `-s 123456` to mean to
  set the random number seed to `123456` for example.
- The EG application should accept a command line switch to set the
  maximum impact parameter (in Fermi-metre) sampled.  This is set via
  the configuration key `GeneratorFileOrCmd.bMaxSwithc` and defaults
  to `-b`.  Thus, the EG application should take the command line
  argument `-b 10` to mean that it should only generate events with an
  impact parameter between 0fm and 10fm.

If a program does not adhere to these requirements, it will often be
simple enough to make a small wrapper script that enforce this.  For
example, `crmc` will write a lot of information to standard output.
We can filter that out via a shell script ([`crmc.sh`](crmc.sh)) like

    #!/bin/sh
    crmc $@ -o hepmc3 -f /dev/stdout | sed -n 's/^\(HepMC::\|[EAUWVP] \)/\1/p'

The `sed` command selects lines that begin with `HepMC::`, or one of
single characters `E` (event), `A` (attribute), `U` (units), `W`
(weight), `V` (vertex), or `P` (particle) followed by a space.  This
should in most cases be enough to filter out extra stuff written on
standard output.

The script above also passes any additional command line options on to
`crmc` via `$@`.  We can utilise this with `o2-sim` to set options to
the CRMC suite.  For example, if we want to simulate p-Pb collisions
using DpmJET, we can do

    o2-sim -g hepmc --configKeyValues "GeneratorFileOrCmd.cmd=crmc.sh -m 12 -i2212 -I 1002080820"


### Implementation details

Internally `GeneratorHepMC`

1. creates a unique temporary file name in the working directory,
2. then creates a FIFO (or named pipe, see
   [Wikipedia](https://en.wikipedia.org/wiki/Named_pipe)),
3. builds a command line, e.g.,

        eg options > fifo-name &

4. and executes that command line

## The configuration keys

The `GeneratorHepMC` (and sister generator `GeneratorTParticle`)
allows customisation of the execution via configuration keys passed
via `--configKeyValues`

- `HepMC.eventsToSkip=number` a number events to skip at the beginning
  of each file read.

- `HepMC.prune=boolean` if true, then prune events of particles that
  are not
  - beam particles (status = 4),
  - decayed (status = 2), nor
  - final state (status = 1)

  This reduces the event size. How much depend on the event
  generator. Use with caution, as it can potentially corrupt the event
  structure (though measures are taken to minimise that risk).

  In the future, we may want more granular control of which particles
  to keep.  For example, we could have the keys

  - `HepMC.keepStatus=list-of-status-codes`
  - `HepMC.keepPDGs=list-of-pdg-numbers`

- `HepMC.version` - when reading the events from files, this option is
   no longer needed.  The code itself figures out which format version
   the input file is in. If executing a child process through
   `GeneratorFileOrCmd.cmd` and the EG writes out HepMC2 format, then
   this _must_ be set to `2`. Otherwise, HepMC3 is assumed.

- `GeneratorFileOrCmd.fileNames=list` a comma separated list of HepMC
  files to read.

- `GeneratorFileOrCmd.cmd=command line` a command line to execute as a
  background child process.  If this is set (not the empty string),
  then `GeneratorFileOrCmd.fileNames` is ignored.

- A number of keys that specifies the command line option switch that
  the child program accepts for certain things.  If any of these are
  set to the empty string, then that switch and corresponding option
  value is not passed to the child program.

  - `GeneratorFileOrCmd.outputSwitch=switch` (default `>`) to specify
    output file.  The default of `>` assumes that the program write
    HepMC events, and _only_ those, to standard output.

  - `GeneratorFileOrCmd.seedSwitch=switch` (default `-s`) to specify
    the random number generator seed. The value passed is selected by
    the `o2-sim` option `--seed`

  - `GeneratorFileOrCmd.bMaxSwitch=switch` (default `-b`) to specify
     the upper limit on the impact parameters sampled.  The value
     passed is selected by the `o2-sim` option `--bMax`

  - `GeneratorFileOrCmd.nEventsSwitch=switch` (default `-n`) to
     specify the number of events to generate.  The value passed is
     selected by the `o2-sim` option `--nEvents` or (`-n`)

  - `GeneratorFileOrCmd.backgroundSwitch=switch` (default `&`) to
    specify how the program is put in the background.  Typically this
    should be `&`, but a program may itself fork to the background.

- Some options are deprecated

  - `HepMC.fileName` - use `GeneratorFileOrCmd.fileNames`

The command line build will now be

> _commandLine_ _nEventsSwitch_ _nEvents_ _seedSwitch_ _seed_
> _bMaxSwitch_ _bMax_ _outputSwitch_ _output_ _backgroundSwitch_

If any of the `Switch` keys are empty, then the corresponding option
is not propagated to the command line.  For example, if _bMaxSwitch_
is empty, then the build command line will be

> _commandLine_ _nEventsSwitch_ _nEvents_ _seedSwitch_ _seed_
> _outputSwitch_ _output_ _backgroundSwitch_

