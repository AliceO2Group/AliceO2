// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <Algorithm/RangeTokenizer.h>
#include <regex>
#include "SimulationDataFormat/InteractionSampler.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimConfig/InteractionDiamondParam.h"
#include "DataFormatsFT0/EventsPerBc.h"
#include <cmath>
#include <TRandom.h>
#include <numeric>
#include <fairlogger/Logger.h>
#include "Steer/MCKinematicsReader.h"
#include "CommonUtils/ConfigurableParam.h"
#include <CCDB/BasicCCDBManager.h>
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "SimConfig/SimConfig.h"
#include <filesystem>
#include <sstream>
#include <vector>
#include <numeric>

//
// Created by Sandro Wenzel on 13.07.21.
//

// A utility to create/engineer (later modify/display) collision contexts

// options struct filled from command line
struct Options {
  std::vector<std::string> interactionRates;
  std::string qedInteraction; // specification for QED contribution
  std::string outfilename;    //
  int orbits;                 // number of orbits to generate (can be a multiple of orbitsPerTF --> determine fraction or multiple of timeframes)
  long seed;                  //
  bool printContext = false;
  std::string bcpatternfile;
  int tfid = 0;          // tfid -> used to calculate start orbit for collisions
  double orbitsEarly = 0.;     // how many orbits from a prev timeframe should still be kept in the current timeframe
  double firstFractionalOrbit; // capture orbit and bunch crossing via decimal number
  uint32_t firstOrbit = 0; // first orbit in run (orbit offset)
  uint32_t firstBC = 0;    // first bunch crossing (relative to firstOrbit) of the first interaction;
  int orbitsPerTF = 256; // number of orbits per timeframe --> used to calculate start orbit for collisions
  bool useexistingkinematics = false;
  bool noEmptyTF = false; // prevent empty timeframes; the first interaction will be shifted backwards to fall within the range given by Options.orbits
  int maxCollsPerTF = -1; // the maximal number of hadronic collisions per TF (can be used to constrain number of collisions per timeframe to some maximal value)
  std::string configKeyValues = ""; // string to init config key values
  long timestamp = -1;              // timestamp for CCDB queries
  std::string individualTFextraction = ""; // triggers extraction of individuel timeframe components when non-null
                                           // format is path prefix
  std::string vertexModeString{"kNoVertex"}; // Vertex Mode; vertices will be assigned to collisions of mode != kNoVertex
  o2::conf::VertexMode vertexMode = o2::conf::VertexMode::kNoVertex;
  std::string external_path = ""; // optional external path where we can directly take the collision contexts
                                  // This is useful when someone else is creating the contexts (MC-data embedding) and we
                                  // merely want to pass these through. If this is given, we simply take the timeframe ID, number of orbits
                                  // and copy the right amount of timeframes into the destination folder (implies individualTFextraction)
  std::string nontrivial_mu_distribution = ""; // path to fetch a non-uniform MC(BC) distribution for the interaction sampler
                                               // can be: (a) ccdb, (b) a ROOT file with the histogram included
};

enum class InteractionLockMode {
  NOLOCK,
  EVERYN,
  MINTIMEDISTANCE
};

struct CcdbUrl {
  std::string server;   // may include http:// or https://
  std::string port;     // empty if none
  std::string fullPath; // everything after server[:port]/
};

std::optional<CcdbUrl> parseCcdbRegex(const std::string& url)
{
  static const std::regex re(
    R"(^(?:ccdb://)(https?://[^/:]+|[^/:]+)(?::(\d+))?/(.+)$)");
  std::smatch m;
  if (!std::regex_match(url, m, re)) {
    return std::nullopt;
  }

  CcdbUrl out;
  out.server = m[1].str();   // server (may include http:// or https://)
  out.port = m[2].str();     // optional port
  out.fullPath = m[3].str(); // remainder
  return out;
}

struct InteractionSpec {
  std::string name; // name (prefix for transport simulation); may also serve as unique identifier
  float interactionRate;
  std::pair<int, float> synconto; // if this interaction locks on another interaction; takes precedence over interactionRate
  InteractionLockMode syncmode = InteractionLockMode::NOLOCK;
  char syncmodeop = 0;         // syncmode operation ("@" --> embedd; "r" --> replace)
  int mcnumberasked = -1;      // number of MC events asked (but can be left -1) in which case it will be determined from timeframelength
  int mcnumberavail = -1;      // number of MC events avail (but can be left -1); if avail < asked there will be reuse of events
  bool randomizeorder = false; // whether order of events will be randomized
};

InteractionSpec parseInteractionSpec(std::string const& specifier, std::vector<InteractionSpec> const& existingPatterns, bool adjustEventCount)
{
  // An interaction specification is a command-separated string
  // of the following form:
  // SPEC=NAMESTRING,INTERACTIONSTRING[,MCNUMBERSTRING]
  //
  // where
  //
  // NAMESTRING : a simple named specifier for the interaction; matching to a simulation prefix used by o2-sim
  //
  // INTERACTIONSTRING: irate | @ID:[ed]FLOATVALUE
  //      - either: a simple number irate specifying the interaction rate in kHz
  //      -     or: a string such as @0:e5, saying that this interaction should match/sync
  //                with collisions of the 0-th interaction, but inject only every 5 collisions.
  //                Alternatively @0:d10000 means to inject but leaving a timedistance of at least 10000ns between signals
  //      -     or: a string r0:e5, saying that this interaction should sync with collisions of the 0-th interaction but
  //                **overwrite** every 5-th interaction with a collision from this interaction name
  // MCNUMBERSTRING: NUMBER1:r?NUMBER2 can specify how many collisions NUMBER1 to produce, taking from a sample of NUMBER2 available collisions
  //      - this option is only supported on the first interaction which is supposed to be the background interaction
  //      - if the 'r' character is present we randomize the order of the MC events

  // tokens are separated by comma
  std::vector<std::string> tokens = o2::RangeTokenizer::tokenize<std::string>(specifier);

  float rate = -1.;
  std::pair<int, float> synconto(-1, 1);

  // extract (kinematics prefix) name
  std::string name = tokens[0];

  // extract the MC number spec if given
  int collisionsasked = -1;
  int collisionsavail = -1;
  bool randomizeorder = false;
  if (tokens.size() > 2) {
    auto mctoken = tokens[2];
    std::regex re("([0-9]*):(r?)([0-9]*)$", std::regex_constants::extended);

    std::cmatch m;
    if (std::regex_match(mctoken.c_str(), m, re)) {
      collisionsasked = std::atoi(m[1].str().c_str());
      if (m[2].str().compare("r") == 0) {
        randomizeorder = true;
      }
      collisionsavail = std::atoi(m[3].str().c_str());
    } else {
      LOG(error) << "Could not parse " << mctoken << " as MCNUMBERSTRING";
      exit(1);
    }
  }

  if (adjustEventCount) {
    // if the number of collisionsavail has not been specified, we should
    // try to extract it from the kinematics directly
    o2::steer::MCKinematicsReader mcreader(name, o2::steer::MCKinematicsReader::Mode::kMCKine);
    if (collisionsavail > 0) {
      collisionsavail = std::min((size_t)collisionsavail, (size_t)mcreader.getNEvents(0));
    } else {
      collisionsavail = mcreader.getNEvents(0);
    }
  }
  LOG(info) << "Collisions avail for " << name << " " << collisionsavail;

  // extract interaction rate ... or locking
  auto& interactionToken = tokens[1];
  if (interactionToken[0] == '@' || interactionToken[0] == 'r') {
    try {
      // locking onto some other interaction
      std::regex re("[@r]([0-9]*):([ed])([0-9]*[.]?[0-9]?)$", std::regex_constants::extended);

      std::cmatch m;
      if (std::regex_match(interactionToken.c_str(), m, re)) {
        auto crossindex = std::atoi(m[1].str().c_str());
        auto mode = m[2].str();
        auto modevalue = std::atof(m[3].str().c_str());

        if (crossindex > existingPatterns.size()) {
          LOG(error) << "Reference to non-existent interaction spec";
          exit(1);
        }
        synconto = std::pair<int, float>(crossindex, modevalue);

        InteractionLockMode lockMode;
        if (mode.compare("e") == 0) {
          lockMode = InteractionLockMode::EVERYN;
        }
        if (mode.compare("d") == 0) {
          lockMode = InteractionLockMode::MINTIMEDISTANCE;
        }
        return InteractionSpec{name, rate, synconto, lockMode, interactionToken[0], collisionsasked, collisionsavail, randomizeorder};
      } else {
        LOG(error) << "Could not parse " << interactionToken << " as INTERACTIONSTRING";
        exit(1);
      }
    } catch (std::regex_error e) {
      LOG(error) << "Exception during regular expression match " << e.what();
      exit(1);
    }
  } else {
    rate = std::atof(interactionToken.c_str());
    return InteractionSpec{name, rate, synconto, InteractionLockMode::NOLOCK, 0, collisionsasked, collisionsavail, randomizeorder};
  }
}

bool parseOptions(int argc, char* argv[], Options& optvalues)
{
  namespace bpo = boost::program_options;
  bpo::options_description options(
    "A utility to create and manipulate digitization contexts (MC collision structure within a timeframe).\n\n"
    "Allowed options");

  options.add_options()(
    "interactions,i", bpo::value<std::vector<std::string>>(&optvalues.interactionRates)->multitoken(), "name,IRate|LockSpecifier")(
    "QEDinteraction", bpo::value<std::string>(&optvalues.qedInteraction)->default_value(""), "Interaction specifier for QED contribution (name,IRATE,maxeventnumber)")(
    "outfile,o", bpo::value<std::string>(&optvalues.outfilename)->default_value("collisioncontext.root"), "Outfile of collision context")(
    "orbits", bpo::value<int>(&optvalues.orbits)->default_value(-1),
    "Number of orbits to generate maximally (if given, can be used to determine the number of timeframes). "
    "Otherwise, the context will be generated by using collision numbers from the interaction specification.")(
    "seed", bpo::value<long>(&optvalues.seed)->default_value(0L), "Seed for random number generator (for time sampling etc). Default 0: Random")(
    "show-context", "Print generated collision context to terminal.")(
    "bcPatternFile", bpo::value<std::string>(&optvalues.bcpatternfile)->default_value(""), "Interacting BC pattern file (e.g. from CreateBCPattern.C); Use \"ccdb\" when fetching from CCDB.")(
    "orbitsPerTF", bpo::value<int>(&optvalues.orbitsPerTF)->default_value(256), "Orbits per timeframes")(
    "orbitsEarly", bpo::value<double>(&optvalues.orbitsEarly)->default_value(0.), "Number of orbits with extra collisions prefixed to each timeframe")(
    "use-existing-kine", "Read existing kinematics to adjust event counts")(
    "timeframeID", bpo::value<int>(&optvalues.tfid)->default_value(0), "Timeframe id of the first timeframe int this context. Allows to generate contexts for different start orbits")(
    "first-orbit", bpo::value<double>(&optvalues.firstFractionalOrbit)->default_value(0), "First (fractional) orbit in the run (HBFUtils.firstOrbit + BC from decimal)")(
    "maxCollsPerTF", bpo::value<int>(&optvalues.maxCollsPerTF)->default_value(-1), "Maximal number of MC collisions to put into one timeframe. By default no constraint.")(
    "noEmptyTF", bpo::bool_switch(&optvalues.noEmptyTF), "Enforce to have at least one collision")(
    "configKeyValues", bpo::value<std::string>(&optvalues.configKeyValues)->default_value(""), "Semicolon separated key=value strings (e.g.: 'TPC.gasDensity=1;...')")(
    "with-vertices", bpo::value<std::string>(&optvalues.vertexModeString)->default_value("kNoVertex"), "Assign vertices to collisions. Argument is the vertex mode. Defaults to no vertexing applied")(
    "timestamp", bpo::value<long>(&optvalues.timestamp)->default_value(-1L), "Timestamp for CCDB queries / anchoring")(
    "extract-per-timeframe", bpo::value<std::string>(&optvalues.individualTFextraction)->default_value(""),
    "Extract individual timeframe contexts. Format required: time_frame_prefix[:comma_separated_list_of_signals_to_offset]")(
    "import-external", bpo::value<std::string>(&optvalues.external_path)->default_value(""), "Take collision contexts (per timeframe) from external files for instance for data-anchoring use-case. Needs timeframeID and number of orbits to be given as well.")(
    "nontrivial-mu-distribution", bpo::value<std::string>(&optvalues.nontrivial_mu_distribution)->default_value(""), "Distribution for MU(BC)");

  options.add_options()("help,h", "Produce help message.");

  bpo::variables_map vm;
  try {
    bpo::store(bpo::command_line_parser(argc, argv).options(options).run(), vm);
    bpo::notify(vm);

    // help
    if (vm.count("help")) {
      std::cout << options << std::endl;
      return false;
    }
    if (vm.count("show-context")) {
      optvalues.printContext = true;
    }
    if (vm.count("use-existing-kine")) {
      optvalues.useexistingkinematics = true;
    }

    o2::conf::SimConfig::parseVertexModeString(optvalues.vertexModeString, optvalues.vertexMode);

    // fix the first orbit and bunch crossing
    // auto orbitbcpair = parseOrbitAndBC(optvalues.firstIRString);
    optvalues.firstOrbit = (uint32_t)optvalues.firstFractionalOrbit;
    optvalues.firstBC = (uint32_t)((optvalues.firstFractionalOrbit - 1. * optvalues.firstOrbit) * o2::constants::lhc::LHCMaxBunches);
    LOG(info) << "First orbit " << optvalues.firstOrbit;
    LOG(info) << "First BC " << optvalues.firstBC;

  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing options; Available options:\n";
    std::cerr << options << std::endl;
    return false;
  }
  return true;
}

bool copy_collision_context(const std::string& external_path, int this_tf_id, int target_tf_id)
{
  namespace fs = std::filesystem;
  try {
    fs::path filename;
    if (fs::exists(external_path) && fs::is_regular_file(external_path)) {
      std::cout << "external_path is an existing file: " << external_path << "\n";
      // use it directly
      filename = fs::path(external_path);
    } else {
      // Construct source file path
      filename = fs::path(external_path) / ("collission_context_" + std::to_string(this_tf_id) + ".root");
    }

    LOG(info) << "Checking existence of file: " << filename;

    if (fs::exists(filename)) {
      // Build destination path
      std::string path_prefix = "tf"; // Can be made configurable
      std::stringstream destination_path_stream;
      destination_path_stream << path_prefix << (target_tf_id) << "/collisioncontext.root";
      fs::path destination_path = destination_path_stream.str();

      // Ensure parent directory exists
      fs::path destination_dir = destination_path.parent_path();
      if (!fs::exists(destination_dir)) {
        fs::create_directories(destination_dir);
        LOG(info) << "Created directory: " << destination_dir;
      }

      // Copy file
      fs::copy_file(filename, destination_path, fs::copy_options::overwrite_existing);
      LOG(info) << "Copied file to: " << destination_path;
      return true;
    } else {
      LOG(warning) << "Source file does not exist: " << filename;
      return false;
    }
  } catch (const fs::filesystem_error& e) {
    LOG(error) << "Filesystem error: " << e.what();
    return false;
  } catch (const std::exception& e) {
    LOG(error) << "Unexpected error: " << e.what();
    return false;
  }
  return true;
}

int main(int argc, char* argv[])
{
  Options options;
  if (!parseOptions(argc, argv, options)) {
    exit(1);
  }

  // init params
  o2::conf::ConfigurableParam::updateFromString(options.configKeyValues);

  // See if this is external mode, which simplifies things
  if (options.external_path.size() > 0) {
    // in this mode, we don't actually have to do much work.
    // all we do is to
    // - determine how many timeframes are asked
    // - check if the right files are present in the external path (someone else needs to create/put them there)
    // - check if the given contexts are consistent with options given (orbitsPerTF, ...)
    // - copy the files into the MC destination folder (this implies timeframeextraction mode)
    // - return

    if (options.orbits < 0) {
      LOG(error) << "External mode; orbits need to be given";
      return 1;
    }

    if (options.orbitsPerTF == 0) {
      LOG(error) << "External mode; need to have orbitsPerTF";
      return 1;
    }

    if (options.individualTFextraction.size() == 0) {
      LOG(error) << "External mode: This requires --extract-per-timeframe";
      return 1;
    }

    // calculate number of timeframes
    auto num_timeframes = options.orbits / options.orbitsPerTF;
    LOG(info) << "External mode for " << num_timeframes << " consecutive timeframes; starting from " << options.tfid;

    // loop over all timeframe ids - check if file is present - (check consistency) - copy to final destination
    for (int i = 0; i < num_timeframes; ++i) {
      auto this_tf_id = options.tfid + i;
      if (!copy_collision_context(options.external_path, this_tf_id, i + 1)) {
        return 1;
      }
    }
    return 0;
  }

  // init random generator
  gRandom->SetSeed(options.seed);

  std::vector<InteractionSpec> ispecs;
  // building the interaction spec
  for (auto& i : options.interactionRates) {
    // this is created as output from
    ispecs.push_back(parseInteractionSpec(i, ispecs, options.useexistingkinematics));
  }

  std::vector<std::pair<o2::InteractionTimeRecord, std::vector<o2::steer::EventPart>>> collisions;
  std::vector<o2::BunchFilling> bunchFillings; // vector of bunch filling objects; generated by interaction samplers

  // now we generate the collision structure (interaction type by interaction type)
  bool usetimeframelength = options.orbits > 0;

  auto setBCFillingHelper = [&options](auto& sampler, auto& bcPatternString) {
    if (bcPatternString == "ccdb") {
      LOG(info) << "Fetch bcPattern information from CCDB";
      // fetch the GRP Object
      auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
      ccdb.setCaching(false);
      ccdb.setLocalObjectValidityChecking(true);
      auto grpLHC = ccdb.getForTimeStamp<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", options.timestamp);
      LOG(info) << "Fetched injection scheme " << grpLHC->getInjectionScheme() << " from CCDB";
      sampler.setBunchFilling(grpLHC->getBunchFilling());
    } else {
      sampler.setBunchFilling(bcPatternString);
    }
  };

  // this is the starting orbit from which on we construct interactions (it is possibly shifted by one tf to the left
  // in order to generate eventual "earlyOrbits"
  auto orbitstart = options.firstOrbit + options.tfid * options.orbitsPerTF;
  auto orbits_total = options.orbits;
  if (options.orbitsEarly > 0.) {
    orbitstart -= options.orbitsPerTF;
    orbits_total += options.orbitsPerTF;
  }

  for (int id = 0; id < ispecs.size(); ++id) {
    auto mode = ispecs[id].syncmode;
    if (mode == InteractionLockMode::NOLOCK) {
      auto sampler = std::make_unique<o2::steer::InteractionSampler>();
      std::unique_ptr<TH1F> mu_hist;

      // we check if there is a realistic bunch crossing distribution available
      const auto& mu_distr_source = options.nontrivial_mu_distribution;
      if (mu_distr_source.size() > 0) {
        if (mu_distr_source.find("ccdb") == 0) {
          auto ccdb_info_wrapper = parseCcdbRegex(mu_distr_source);
          if (!ccdb_info_wrapper.has_value()) {
            LOG(error) << "Could not parse CCDB path for mu(bc) distribution";
          } else {
            auto& ccdb_info = ccdb_info_wrapper.value();

            // for now construct a specific CCDBManager for this query
            o2::ccdb::CCDBManagerInstance ccdb_inst(ccdb_info.server + std::string(":") + ccdb_info.port);
            ccdb_inst.setFatalWhenNull(false);
            auto local_hist = ccdb_inst.getForTimeStamp<TH1F>(ccdb_info.fullPath, options.timestamp);
            if (local_hist) {
              // case in which CCDB object contains directly a ROOT histogram
              mu_hist.reset((TH1F*)local_hist->Clone("h2")); // we need to clone since ownership of local_hist is with TFile
            } else if (auto events_per_bc = ccdb_inst.getForTimeStamp<o2::ft0::EventsPerBc>(ccdb_info.fullPath, options.timestamp)) {
              // case in which CCDB object is from FT0 EventsPerBC calib (will be default)
              mu_hist = events_per_bc->toTH1F();
            } else {
              LOG(warn) << "No mu(bc) distribution found on CCDB. Using uniform one";
            }
          }
        } else {
          // we interpret the file as a ROOT file and open it to extract the wanted histogram
          auto mudistr_file = TFile::Open(mu_distr_source.c_str(), "OPEN");
          if (mudistr_file && !mudistr_file->IsZombie()) {
            auto local_hist = mudistr_file->Get<TH1F>("hBcTVX");
            mu_hist.reset((TH1F*)local_hist->Clone("h2")); // we need to clone since ownership of local_hist is with TFile
            mudistr_file->Close();
          }
        }
        if (mu_hist) {
          LOG(info) << "Found an external mu distribution with mean BC value " << mu_hist->GetMean();

          // do some checks

          // reset to correct interaction Sampler type
          sampler.reset(new o2::steer::NonUniformMuInteractionSampler());
        }
      }

      // for debug purposes: allows to instantiate trivial sampler
      if (const char* env = getenv("ALICEO2_ENFORCE_TRIVIAL_BC_SAMPLER")) {
        std::string spec(env);
        std::regex re(R"((\d+):(\d+))");
        std::smatch match;
        int every_n = 1, mult = 1;
        if (std::regex_match(spec, match, re)) {
          every_n = std::stoi(match[1]);
          mult = std::stoi(match[2]);
        } else {
          LOG(error) << "ALICEO2_ENFORCE_TRIVIAL_BC_SAMPLER format invalid, expected NUMBER_1:NUMBER_2";
          exit(1);
        }
        sampler.reset(new o2::steer::FixedSkipBC_InteractionSampler(every_n, mult));
      }

      sampler->setInteractionRate(ispecs[id].interactionRate);
      if (!options.bcpatternfile.empty()) {
        setBCFillingHelper(*sampler, options.bcpatternfile);
      }
      sampler->init();
      if (auto sampler_cast = dynamic_cast<o2::steer::NonUniformMuInteractionSampler*>(sampler.get())) {
        if (mu_hist) {
          sampler_cast->setBCIntensityScales(*mu_hist);
        }
      }

      o2::InteractionTimeRecord record;
      // this loop makes sure that the first collision is within the range of orbits asked (if noEmptyTF is enabled)
      do {
        sampler->setFirstIR(o2::InteractionRecord(options.firstBC, orbitstart));
        sampler->init();
        record = sampler->generateCollisionTime();
      } while (options.noEmptyTF && usetimeframelength && record.orbit >= orbitstart + orbits_total);
      int count = 0;
      do {
        if (usetimeframelength && record.orbit >= orbitstart + orbits_total) {
          break;
        }
        std::vector<o2::steer::EventPart> parts;
        parts.emplace_back(id, count);

        std::pair<o2::InteractionTimeRecord, std::vector<o2::steer::EventPart>> insertvalue(record, parts);
        auto iter = std::lower_bound(collisions.begin(), collisions.end(), insertvalue, [](std::pair<o2::InteractionTimeRecord, std::vector<o2::steer::EventPart>> const& a, std::pair<o2::InteractionTimeRecord, std::vector<o2::steer::EventPart>> const& b) { return a.first < b.first; });
        collisions.insert(iter, insertvalue);
        record = sampler->generateCollisionTime();
        count++;
      } while ((ispecs[id].mcnumberasked > 0 && count < ispecs[id].mcnumberasked)); // TODO: this loop should probably be replaced by a condition with usetimeframelength and number of orbits

      // we support randomization etc on non-injected/embedded interactions
      // and we can apply them here
      auto random_shuffle = [](auto first, auto last) {
        auto n = last - first;
        for (auto i = n - 1; i > 0; --i) {
          using std::swap;
          swap(first[i], first[(int)(gRandom->Rndm() * n)]);
        }
      };
      std::vector<int> eventindices(count);
      std::iota(eventindices.begin(), eventindices.end(), 0);
      // apply randomization of order if any
      if (ispecs[id].randomizeorder) {
        random_shuffle(eventindices.begin(), eventindices.end());
      }
      if (ispecs[id].mcnumberavail > 0) {
        // apply cutting to number of available entries
        for (auto& e : eventindices) {
          e = e % ispecs[id].mcnumberavail;
        }
      }
      // make these transformations final:
      for (auto& col : collisions) {
        for (auto& part : col.second) {
          if (part.sourceID == id) {
            part.entryID = eventindices[part.entryID];
          }
        }
      }

      // keep bunch filling information produced by these samplers
      bunchFillings.push_back(sampler->getBunchFilling());

    } else {
      // we are in some lock/sync mode and modify existing collisions
      int lastcol = -1;
      double lastcoltime = -1.;
      auto distanceval = ispecs[id].synconto.second;
      auto lockonto = ispecs[id].synconto.first;
      int eventcount = 0;

      for (int colid = 0; colid < collisions.size(); ++colid) {
        auto& col = collisions[colid];
        auto coltime = col.first.getTimeNS();

        bool rightinteraction = false;
        // we are locking only on collisions which have the referenced interaction present
        // --> there must be an EventPart with the right sourceID
        for (auto& eventPart : col.second) {
          if (eventPart.sourceID == lockonto) {
            rightinteraction = true;
            break;
          }
        }
        if (!rightinteraction) {
          continue;
        }

        bool inject = false;
        // we always start with first one
        if (lastcol == -1) {
          inject = true;
        }
        if (mode == InteractionLockMode::EVERYN && (colid - lastcol) >= distanceval) {
          inject = true;
        }
        if (mode == InteractionLockMode::MINTIMEDISTANCE && (coltime - lastcoltime) >= distanceval) {
          inject = true;
        }

        if (inject) {
          if (ispecs[id].syncmodeop == 'r') {
            LOG(debug) << "Replacing/overwriting another event ";
            // Syncing is replacing; which means we need to take out the original
            // event that we locked onto.
            // We take out this event part immediately (and complain if there is a problem).
            int index = 0;
            auto iter = std::find_if(col.second.begin(), col.second.end(), [lockonto](auto val) { return lockonto == val.sourceID; });
            if (iter != col.second.end()) {
              col.second.erase(iter);
            } else {
              LOG(error) << "Expected to replace another event part but did not find one for source " << lockonto << " and collision " << colid;
            }
          }

          if (ispecs[id].mcnumberavail >= 0) {
            col.second.emplace_back(id, eventcount % ispecs[id].mcnumberavail);
          } else {
            col.second.emplace_back(id, eventcount);
          }
          eventcount++;
          lastcol = colid;
          lastcoltime = coltime;
        }
      }
    }
  }

  // create DigitizationContext
  o2::steer::DigitizationContext digicontext;
  // we can fill this container
  auto& parts = digicontext.getEventParts();
  // we can fill this container
  auto& records = digicontext.getEventRecords();
  // copy over information
  size_t maxParts = 0;
  for (auto& p : collisions) {
    records.push_back(p.first);
    parts.push_back(p.second);
    maxParts = std::max(p.second.size(), maxParts);
  }
  digicontext.setNCollisions(collisions.size());
  digicontext.setMaxNumberParts(maxParts);
  // merge bunch filling info
  for (int i = 1; i < bunchFillings.size(); ++i) {
    bunchFillings[0].mergeWith(bunchFillings[i]);
  }
  digicontext.setBunchFilling(bunchFillings[0]);
  std::vector<std::string> prefixes;
  // Signal interaction rate
  float sgnIRate = -1.;
  for (auto& p : ispecs) {
    prefixes.push_back(p.name);
    // Set the interaction rate from the first pattern with a valid value.
    // This handles both simple signal-only productions (where "sgn" has the rate)
    // and embedding productions (where "bkg" has the rate and "sgn" syncs to it)
    if (sgnIRate < 0 && p.interactionRate > 0) {
      LOG(debug) << "Setting signal interaction rate to " << p.interactionRate << " Hz in the digitization context.";
      sgnIRate = p.interactionRate;
      digicontext.setDigitizerInteractionRate(p.interactionRate);
    }
  }
  digicontext.setSimPrefixes(prefixes);

  // <---- at this moment we have a dense collision context (not representing the final output we want)
  LOG(info) << "<<------ DENSE CONTEXT ---------";
  if (options.printContext) {
    digicontext.printCollisionSummary();
  }
  LOG(info) << "-------- DENSE CONTEXT ------->>";

  auto timeframeindices = digicontext.calcTimeframeIndices(orbitstart, options.orbitsPerTF, options.orbitsEarly);
  LOG(info) << "Fixed " << timeframeindices.size() << " timeframes ";
  for (auto p : timeframeindices) {
    LOG(info) << std::get<0>(p) << " " << std::get<1>(p) << " " << std::get<2>(p);
  }

  // apply max collision per timeframe filters + reindexing of event id (linearisation and compactification)
  digicontext.applyMaxCollisionFilter(timeframeindices, orbitstart, options.orbitsPerTF, options.maxCollsPerTF, options.orbitsEarly);

  LOG(info) << "Timeframe indices after collision filter";
  LOG(info) << "Fixed " << timeframeindices.size() << " timeframes ";
  for (auto p : timeframeindices) {
    LOG(info) << std::get<0>(p) << " " << std::get<1>(p) << " " << std::get<2>(p);
  }

  // <---- at this moment we have a dense collision context (not representing the final output we want)
  LOG(info) << "<<------ FILTERED CONTEXT ---------";
  if (options.printContext) {
    digicontext.printCollisionSummary();
  }
  LOG(info) << "-------- FILTERED CONTEXT ------->>";

  auto numTimeFrames = timeframeindices.size(); // digicontext.finalizeTimeframeStructure(orbitstart, options.orbitsPerTF, options.orbitsEarly);

  if (options.vertexMode != o2::conf::VertexMode::kNoVertex) {
    switch (options.vertexMode) {
      case o2::conf::VertexMode::kCCDB: {
        // fetch mean vertex from CCDB
        auto meanv = o2::ccdb::BasicCCDBManager::instance().getForTimeStamp<o2::dataformats::MeanVertexObject>("GLO/Calib/MeanVertex", options.timestamp);
        if (meanv) {
          LOG(info) << "Applying vertexing using CCDB mean vertex " << *meanv;
          digicontext.sampleInteractionVertices(*meanv);
        } else {
          LOG(fatal) << "No vertex available";
        }
        break;
      }

      case o2::conf::VertexMode::kDiamondParam: {
        // init this vertex from CCDB or InteractionDiamond parameter
        const auto& dparam = o2::eventgen::InteractionDiamondParam::Instance();
        o2::dataformats::MeanVertexObject meanv(dparam.position[0], dparam.position[1], dparam.position[2], dparam.width[0], dparam.width[1], dparam.width[2], dparam.slopeX, dparam.slopeY);
        LOG(info) << "Applying vertexing using DiamondParam mean vertex " << meanv;
        digicontext.sampleInteractionVertices(meanv);
        break;
      }
      default: {
        LOG(error) << "Unknown vertex mode ... Not generating vertices";
      }
    }
  }

  // we fill QED contributions to the context
  if (options.qedInteraction.size() > 0) {
    // TODO: use bcFilling information
    auto qedSpec = parseInteractionSpec(options.qedInteraction, ispecs, options.useexistingkinematics);
    std::cout << "### IRATE " << qedSpec.interactionRate << "\n";
    digicontext.fillQED(qedSpec.name, qedSpec.mcnumberasked, qedSpec.interactionRate);
  }

  if (options.printContext) {
    digicontext.printCollisionSummary();
  }
  digicontext.saveToFile(options.outfilename);

  // extract individual timeframes
  if (options.individualTFextraction.size() > 0) {
    // we are asked to extract individual timeframe components

    LOG(info) << "Extracting individual timeframe collision contexts";
    // extract prefix path to store these collision contexts
    // Function to check the pattern and extract tokens from b
    auto check_and_extract_tokens = [](const std::string& input, std::vector<std::string>& tokens) {
      // the regular expression pattern for expected input format
      const std::regex pattern(R"(^([a-zA-Z0-9]+)(:([a-zA-Z0-9]+(,[a-zA-Z0-9]+)*))?$)");
      std::smatch matches;

      // Check if the input matches the pattern
      if (std::regex_match(input, matches, pattern)) {
        // Clear any existing tokens in the vector
        tokens.clear();

        // matches[1] contains the part before the colon which we save first
        tokens.push_back(matches[1].str());
        // matches[2] contains the comma-separated list
        std::string b = matches[2].str();
        std::regex token_pattern(R"([a-zA-Z0-9]+)");
        auto tokens_begin = std::sregex_iterator(b.begin(), b.end(), token_pattern);
        auto tokens_end = std::sregex_iterator();

        // Iterate over the tokens and add them to the vector
        for (std::sregex_iterator i = tokens_begin; i != tokens_end; ++i) {
          tokens.push_back((*i).str());
        }
        return true;
      }
      LOG(error) << "Argument for --extract-per-timeframe does not match specification";
      return false;
    };

    std::vector<std::string> tokens;
    if (check_and_extract_tokens(options.individualTFextraction, tokens)) {
      auto path_prefix = tokens[0];
      std::vector<int> sources_to_offset{};

      LOG(info) << "PREFIX is " << path_prefix;

      for (int i = 1; i < tokens.size(); ++i) {
        LOG(info) << "Offsetting " << tokens[i];
        sources_to_offset.push_back(digicontext.findSimPrefix(tokens[i]));
      }

      auto first_timeframe = options.orbitsEarly > 0. ? 1 : 0;
      // now we are ready to loop over all timeframes
      int tf_output_counter = 1;
      for (int tf_id = first_timeframe; tf_id < numTimeFrames; ++tf_id) {
        auto copy = digicontext.extractSingleTimeframe(tf_id, timeframeindices, sources_to_offset);

        // each individual case gets QED interactions injected
        // This should probably be done inside the extraction itself
        if (digicontext.isQEDProvided()) {
          auto qedSpec = parseInteractionSpec(options.qedInteraction, ispecs, options.useexistingkinematics);
          copy.fillQED(qedSpec.name, qedSpec.mcnumberasked, qedSpec.interactionRate);
        }

        std::stringstream str;
        str << path_prefix << tf_output_counter++ << "/collisioncontext.root";
        copy.saveToFile(str.str());
        LOG(info) << "---- CollisionContext for timeframe " << tf_id << " -----";
        copy.printCollisionSummary();
      }
    }
  }

  return 0;
}
