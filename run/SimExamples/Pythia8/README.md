<!-- doxy
\page refrunSimExamplesPythia Example of generating Pythia events
/doxy -->

Pythia8 is the only event generator guarantied to be included into O2.
This is because it is used for decays in the simulations.

The event generator is selected by

    o2-sim -g pythia8 ...

Pythia8 is heavily [documented](https://pythia8.org) at its
web-site. Please refer to that documentation for more.

## Configuration keys

Configuration keys are specified by

    o2-sim --configKeyValues "comma-separated-list-of-pairs"

for example

    o2-sim -g pythia8 \
      --configKeyValues "GeneratorPythia8.config=myconfig.cfg"

Available configuration keys for Pythia8 are

- `GeneratorPythia8.config=filename` Specifies the configuration file
  for Pythia8 to read.  The
  [format](https://pythia.org/latest-manual/SettingsScheme.html) and
  [settings](https://pythia.org/latest-manual) are heavily documented
  at the Pythia web-site.  Note, if a predefined configuration is
  selected (e.g, `pythia8pp`) - see below - then the file specified
  here will be read _after_ the default configuration.
- `GeneratorPythia8.hooksFileName=script-file-name` Specifies a ROOT
  script to be executed to define a `Pythia8::UserHooks` (see
  [documentation](https://pythia.org/latest-manual/UserHooks.html))
  object. See for example
  `${O2_ROOT}/share/Generators/egconfig/pythia8_userhooks_charm.C`.
- `GeneratorPythia8.hooksFuncname=function-name` Specifies the
  function in `GeneratorPythia8::hooksFileName` to call.  This
  function _must_ return a pointer to a `Pythia8::UserHooks`
  object. See for example
  `${O2_ROOT}/share/Generators/egconfig/pythia8_userhooks_charm.C`. The
  default value is `pythia8_userhooks`.
- `GeneratorPythia8.includePartonEvent=boolean` If `false` (default),
  prune the events for particles that are _not_
  - beam particles (HepMC status = 4),
  - decayed (HepMC status = 2), nor
  - final state (HepMC status = 1) This reduces the size of the events
  by roughly 30%. Note that setting this option to `false` _may_
  corrupt the events, and it should be used with caution.  If `true`,
  then the whole event is stored.

## Predefined configurations

The class `o2::eventgen::GeneratorFactory` (which parses the `o2-sim
-g` option) has a number of predefined configurations of Pythia. These
are listed below together with their configuration file (read from
`${O2_ROOT}/share/Generators/egconfig/').

- `alldet` (`pythia8_inel.cfg`) pp at 14 TeV, min. bias inelastic
  collisions, with an additional muon box generator in the MUON arm
  acceptance.
- `pythia8inel` (`pythia8_inel.cfg`) pp at 14 TeV, min. bias inelastic
  collisions.
- `pythia8hf` (`pythia8_hf.cfg`) pp at 14 TeV, with hard c-cbar and
  b-bar processes turned on.
- `pythia8powheg` (`pythia8_powheg.cfg`) pp at 13 TeV using POWHEG
  parton distribution functions (via LHEF files).
- `pythia8hi` (`pythia8_hi.cfg`) Pb-Pb at 5.52 TeV using the Angantyr
  model.

## Alternative

Rather than using the built-in Pythia8 event generator, we can also
use an external application that writes HepMC output.  For example

    /*
       g++ -std=c++17 -I${PYTHIA8_ROOT}/include \
       -I${HEPMC3_ROOT}/include \
       pythia.cc -o pythia \
       -L${PYTHIA8_ROOT} -L${HEPMC3_ROOT}/lib \
       -Wl,-rpath,${PYTHIA8_ROOT}/lib \
       -Wl,-rpath,${HEPMC3_ROOT}/lib \
       -lHepMC3 \
       -lpythia8
    */
    #include <HepMC3/WriterAscii.h>
    #include <Pythia8/Pythia.h>
    #include <Pythia8Plugins/HepMC3.h>
    #include <string>
    #include <fstream>

    struct BasePythia {
      Pythia8::Pythia _pythia{"",false};
      int             _seed = -1;
      std::string     _config;
      void silence() {
        _pythia.readString("Print:quiet                      = on");
        _pythia.readString("Init:showAllSettings             = off");
        _pythia.readString("Stat:showProcessLevel            = off");
        _pythia.readString("Init:showAllParticleData         = off");
        _pythia.readString("Init:showChangedParticleData     = off");
        _pythia.readString("Init:showChangedSettings         = off");
        _pythia.readString("Init:showMultipartonInteractions = off");
      }
      void seed() {
        _pythia.readString("Random:setSeed = "
                   +std::string(_seed >= 0 ? "yes" : "no"));
        _pythia.readString("Random:seed = "+std::to_string(_seed));
      }
      virtual void configure() = 0;
      void init()  {
        silence();
        seed();
        configure();
        if (not _config.empty()) _pythia.readFile(_config);
        _pythia.init();
      }
      bool loop(size_t nev, const std::string& filename) {
        std::ofstream*         file = (filename == "-" or filename.empty()
                                       ? nullptr
                                       : new std::ofstream(filename.c_str()));
        std::ostream&           out  = file ? *file : std::cout;
        HepMC3::GenEvent        event;
        HepMC3::WriterAscii     writer(out);
        HepMC3::Pythia8ToHepMC3 converter;

        for (size_t iev = 0; iev < nev; ++iev) {
          if (not _pythia.next()) return false;
          event    .clear();
          converter.fill_next_event(_pythia,event);
          writer   .write_event(event);
        }
        if (file) {
          file->close();
          delete file;
        }
        return true;
      }
    };

    struct Pythia : BasePythia {
      void configure() {
        _pythia.readString("Beams:idA                = 2212");
        _pythia.readString("Beams:idB                = 2212");
        _pythia.readString("Beams:eCM                = 5020.");
        _pythia.readString("SoftQCD:inelastic        = on");
        _pythia.readString("ParticleDecays:limitTau0 = on");
        _pythia.readString("ParticleDecays:tau0Max   = 10");
        _pythia.readString("Tune:ee                  = 7");
        _pythia.readString("Tune:pp                  = 14");
      }
    };

    struct Angantyr : BasePythia {
      enum Model {
        fixed = 0,
        random = 1,
        opacity = 2
      } _model = fixed;
      float _bMax = 15;
      virtual void configure() {
        _pythia.readString("Angantyr::CollisionModel = "+std::to_string(_model));
        _pythia.readString("HeavyIon:showInit        = off");
        _pythia.readString("HeavyIon:SigFitPrint     = off");
        _pythia.readString("HeavyIon:SigFitDefPar    = 0,0,0,0,0,0,0,0,0");
        _pythia.readString("HeavyIon:SigFitNGen      = 20");
        _pythia.readString("HeavyIon:bWidth          = "+std::to_string(_bMax));
      }
    };

    struct PbPb : Angantyr {
      void configure() {
        Angantyr::configure();
        _pythia.readString("Beams:idA                = 1000822080");
        _pythia.readString("Beams:idB                = 1000822080");
        _pythia.readString("Beams:eCM                = 5020.");
        _pythia.readString("Beams:frameType          = 1");
        _pythia.readString("ParticleDecays:limitTau0 = on");
        _pythia.readString("ParticleDecays:tau0Max   = 10");
      }
    };

    struct pPb : Angantyr {
      void configure() {
        Angantyr::configure();
        _pythia.readString("Beams:idA                = 2212");
        _pythia.readString("Beams:idB                = 1000822080");
        _pythia.readString("Beams:eA                 = 7000.");
        _pythia.readString("Beams:eB                 = 2760.");
        _pythia.readString("Beams:frameType          = 1");
        _pythia.readString("ParticleDecays:limitTau0 = on");
        _pythia.readString("ParticleDecays:tau0Max   = 10");
      }
    };

    int main(int argc, char** argv) {
      int seed           = -1;
      int nev            = 10;
      std::string output = "-";
      std::string system = "pp";
      std::string config = "";
      for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
          switch (argv[i][1]) {
          case 's': seed   = std::stoi(argv[++i]); break;
          case 'n': nev    = std::stoi(argv[++i]); break;
          case 'o': output = argv[++i]; break;
          case 'c': config = argv[++i]; break;
          case 'h':
        std::cout << "Usage: " << argv[0] << " [OPTIONS] [SYSTEM]\n\n"
              << "Options:\n"
              << "  -h           This help\n"
              << "  -n NUMBER    Number of events\n"
              << "  -o FILE      Output file\n"
              << "  -s SEED      Random number seed\n"
              << "  -c FILE      Configuratio file\n\n"
              << "Systems (all at sqrt(s)=5.02TeV):\n"
              << "  pp, p-Pb, or Pb-Pb\n"
              << std::endl;
        return 0;
          default:
        std::cerr << "Unknown option: " << argv[i] << std::endl;
        return 1;
          }
        } else
          system = argv[i];
      }

      BasePythia* eg = nullptr;
      if (system == "pp")    eg = new Pythia;
      if (system == "p-Pb")  eg = new pPb;
      if (system == "Pb-Pb") eg = new PbPb;
      if (not eg) {
        std::cerr << "Unknown system: " << system << std::endl;
        return 1;
      }
      eg->_seed   = seed;
      eg->_config = config;
      eg->init();
      bool ret = eg->loop(nev, output);

      delete eg;

      return ret ? 0 : 1;
    }

The above program sets up a relative generic event generator that uses
Pythia and writes the results to either file or standard output in the
HepMC format.  This can be used with the generator `hepmc` as f.ex.

    o2-sim -g hepmc \
      --configKeyValues="GeneratorFileOrCmd.cmd=./pythia pPb"

