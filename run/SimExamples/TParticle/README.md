<!-- doxy
\page refrunSimExamplesTParticle Example reading TParticle events
/doxy -->

Here are pointers on how to use `GeneratorTParticle` selected by the
option `-g tparticle` for `o2-sim`. 

## Reading TParticle files 

The generator `GeneratorTParticle` can read events from a ROOT file
containing a `TTree` with a branch holding a `TClonesArray` of
`TParticle` objects. These files can be produced by a standalone event
generator program (EG).

To make a simulation reading from the file `particles.root`, do 

    o2-sim -g tparticle --configKeyValues "GeneratorTParticle.fileName=particles.root"	...
	
See also [`read.sh`](read.sh).  Do 

    ./read.sh --help 
	
for a list of options.   This expects an input file with a `TTree`
with a single `TBranch` holding a `TClonesArray` of `TParticle`
objects.  One such example file can be made with [`myeg.sh`].  Do 

    ./myeg.sh --help 
	
for a list of options. 

### Configurations 

- The name of the `TTree` to read can be set by the configuration key
  `GeneratorTParticle.treeName`.  The default is `T` 
- The name of the `TBranch` holding the `TClonesArray` of `TParticle`
  objects can be set by the configuration key
  `GeneratorTParticle.branchName`.  The default is `Particles` 
  
For example 

    o2-sim -g tparticle --configKeyValues "GeneratorTParticle.fileName=particles.root;GeneratorTParticle.treeName=Events;GeneratorTParticle.branchName=Tracks"	...

## Reading TParticle events from child process 

`GeneratorTParticle` can not only read events from a file, but can
also spawn an child EG to produce events.  Suppose we have a program
named `eg` which is some EG that writes `TParticle` event records to a
file .  Then we can execute a simulation using this external
EG by

    o2-sim -g tgenerator --configKeyValues "GeneratorTParticle.progCmd=eg" 

See also [`child.sh`](child.sh).  Do 

    ./child.sh --help 
	
for a list of options.

There are some requirements on the program `eg`: 

- It _must_ accept the option `-n n-events` to set the number of
  events to produce to `n-events`. 

- It _must_ accept the option `-o output` to set the output file name.

If a program does not adhere to these requirements, it will often be
simple enough to make a small wrapper script that enforce this.

### Configurations 

Same as above. 

### Example EG 

The child-process feature allows us to use almost any EG for which we
have a `TGenerator` interface without compiling it in to O2.  Suppose
we have defined the class `MyGenerator` to produce events.  

    class MyGenerator : public TGenerator {
    public:
      MyGenerator();
      void Initialize(Long_t   projectile,
     		          Long_t   target,
    		          Double_t sqrts);
      void GenerateEvent();
      Int_t ImportParticles(TClonesArray* particles,Option_t*option="");
    };

and a steering class 

    struct MySteer { 
	  TGenerator*   generator;
	  TFile*        file;
	  TTree*        tree;
	  TClonesArray* particle;
	  Int_t         flushEvery;
	  MySteer(TGenerator* generator,
	          const TString& output,
			  Int_t flushEvery) 
	    : generator(generator)
		  file(TFile::Open(output,"RECREATE")),
		  tree("T","Particle tree"),
		  particles(new TClonesArray("TParticle")),
		  flushEvery(flushEvery)
	  {
	    tree->SetDirectory(file);
		tree->Branch("Particles",&particles);
      }
	  ~MySteer() { close(); }
	  void event() {
	    particles->Clear();
		generator->GenerateEvent();
		generator->ImportParticles(particles);
		tree->Fill();
	  }
	  void sync() { 
	    tree->AutoSave("SaveSelf FlushBaskets Overwrite");
	  }
	  void run(Int_t nev) {
	     for (Int_t iev = 0; iev < nev; iev++) {
		   event();
		   if (flushEvery > 0 and (iev % flushEvery == 0) and iev != 0)
		     sync();
		 }
	  }
	  void close() { 
	    if (not file) return;
		file->Write();
		file->Close();
		file = nullptr;
	  }
	};
	
Then we could make the script [`MyEG.cc` (complete code)](MyEG.cc) like

    void MyEG(Int_t nev,const TString& out,Int_t every=1)
    {
      MyGenerator* eg = new MyGenerator();
      eg->Initialize(2212, 2212, 5200);
    
	  MySteer steer(eg, out, every);
	  steer.run(nev);
    }
	
and a simple shell-script [`myeg.sh`](myeg.sh) to pass arguments to
the `MyEG.cc` script

    #!/bin/sh
    
    nev=1
    out=particles.root
    
    while test $# -gt 0 ; do
        case $1 in
    	-n) nev=$2 ; shift ;;
    	-o) out=$2 ; shift ;;
    	*) ;;
        esac
        shift
    done
    
    root -l MyEG.cc -- $nev \"$out\"

We can then do 

    o2-sim -g tgenerator --configKeyValues "GeneratorTParticle.progCmd=./myeg.sh" 

to produce events with our generator `MyGenerator`. 

	
### Implementation details 

Internally `GeneratorTParticle` 

1. creates a unique temporary file name in the working directory,
2. builds a command line, e.g., 

      eg options -o temporary-name & 
	  
3. and executes that command line 

## The future 

The `GeneratorTParticle` (and sister generator `GeneratorHepMC`) will
in the not so distant future be upgraded with new functionality to
more easily customise reading files and executing a child process.  In
particular

- New options that can be specified in `--configKeyValues` 

  - `GeneratorTParticle.treeName=name` the name of the `TTree` in the
    input files. 
	
  - `GeneratorTParticle.branchName=name` the name of the `TBranch` in
    the `TTree` that holds the `TClonesArray` of `TParticle` objects.
	
  - `FileOrCmd.fileNames=list` a comma separated list of HepMC files
    to read 
	
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

  - `GeneratorTParticle.fileName` - use `FileOrCmd.fileNames`
  - `GeneratorTParticle.progCmd` - use `FileOrCmd.cmd` 
  
The command line build will now be 

> _commandLine_ _nEventsSwitch_ _nEvents_ _seedSwitch_ _seed_
> _bMaxSwitch_ _bMax_ _outputSwitch_ _output_ _backgroundSwitch_

If any of the `Switch` keys are empty or set to `none`, then the
corresponding option is not propagated to the command line.  For
example, if _bMaxSwitch_ is empty, then the build command line will be

> _commandLine_ _nEventsSwitch_ _nEvents_ _seedSwitch_ _seed_
> _outputSwitch_ _output_ _backgroundSwitch_


### Header information 

The class `GeneratorTParticle` will take a key parameter, say
`headerName` which will indicate a branch that contains header
information.  Under that branch, the class will then search for leaves
(`TLeaf`) that correspond to standard header information keys (see
o2::dataformats::MCInfoKeys).  If any of those leaves are present,
then the corresponding keys will be set on the generated event header. 

Thus, as long as the generator observes the convention used, we can
also import auxiliary information (impact parameter, Npart, ...) from
the input files in addition to the particle information. 
