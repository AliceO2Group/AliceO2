// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <SimConfig/SimConfig.h>
#include <DetectorsCommonDataFormats/DetID.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <FairLogger.h>
#include <thread>
#include <cmath>

using namespace o2::conf;
namespace bpo = boost::program_options;

void SimConfig::initOptions(boost::program_options::options_description& options)
{
  int nsimworkersdefault = std::max(1u, std::thread::hardware_concurrency() / 2);
  options.add_options()(
    "mcEngine,e", bpo::value<std::string>()->default_value("TGeant3"), "VMC backend to be used.")(
    "generator,g", bpo::value<std::string>()->default_value("boxgen"), "Event generator to be used.")(
    "trigger,t", bpo::value<std::string>()->default_value(""), "Event generator trigger to be used.")(
    "modules,m", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>({"all"}), "all modules"), "list of detectors")(
    "skipModules", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>({""}), ""), "list of detectors to skip (precendence over -m")(
    "nEvents,n", bpo::value<unsigned int>()->default_value(1), "number of events")(
    "startEvent", bpo::value<unsigned int>()->default_value(0), "index of first event to be used (when applicable)")(
    "extKinFile", bpo::value<std::string>()->default_value("Kinematics.root"),
    "name of kinematics file for event generator from file (when applicable)")(
    "embedIntoFile", bpo::value<std::string>()->default_value(""),
    "filename containing the reference events to be used for the embedding")(
    "bMax,b", bpo::value<float>()->default_value(0.), "maximum value for impact parameter sampling (when applicable)")(
    "isMT", bpo::value<bool>()->default_value(false), "multi-threaded mode (Geant4 only")(
    "outPrefix,o", bpo::value<std::string>()->default_value("o2sim"), "prefix of output files")(
    "logseverity", bpo::value<std::string>()->default_value("INFO"), "severity level for FairLogger")(
    "logverbosity", bpo::value<std::string>()->default_value("medium"), "level of verbosity for FairLogger (low, medium, high, veryhigh)")(
    "configKeyValues", bpo::value<std::string>()->default_value(""), "semicolon separated key=value strings (e.g.: 'TPC.gasDensity=1;...")(
    "configFile", bpo::value<std::string>()->default_value(""), "Path to an INI or JSON configuration file")(
    "chunkSize", bpo::value<unsigned int>()->default_value(500), "max size of primary chunk (subevent) distributed by server")(
    "chunkSizeI", bpo::value<int>()->default_value(-1), "internalChunkSize")(
    "seed", bpo::value<int>()->default_value(-1), "initial seed (default: -1 random)")(
    "field", bpo::value<std::string>()->default_value("-5"), "L3 field rounded to kGauss, allowed values +-2,+-5 and 0; +-5U for uniform field ")(
    "nworkers,j", bpo::value<int>()->default_value(nsimworkersdefault), "number of parallel simulation workers (only for parallel mode)")(
    "noemptyevents", "only writes events with at least one hit")(
    "CCDBUrl", bpo::value<std::string>()->default_value("ccdb-test.cern.ch:8080"), "URL for CCDB to be used.")(
    "timestamp", bpo::value<long>()->default_value(-1), "global timestamp value (for anchoring) - default is now")(
    "asservice", bpo::value<bool>()->default_value(false), "run in service/server mode");
}

bool SimConfig::resetFromParsedMap(boost::program_options::variables_map const& vm)
{
  using o2::detectors::DetID;
  mConfigData.mMCEngine = vm["mcEngine"].as<std::string>();
  mConfigData.mActiveDetectors = vm["modules"].as<std::vector<std::string>>();
  auto& active = mConfigData.mActiveDetectors;
  if (active.size() == 1 && active[0] == "all") {
    active.clear();
    for (int d = DetID::First; d <= DetID::Last; ++d) {
#ifdef ENABLE_UPGRADES
      if (d != DetID::IT3 && d != DetID::TRK && d != DetID::FT3) {
        active.emplace_back(DetID::getName(d));
      }
#else
      active.emplace_back(DetID::getName(d));
#endif
    }
    // add passive components manually (make a PassiveDetID for them!)
    active.emplace_back("HALL");
    active.emplace_back("MAG");
    active.emplace_back("DIPO");
    active.emplace_back("COMP");
    active.emplace_back("PIPE");
    active.emplace_back("ABSO");
    active.emplace_back("SHIL");
  }
  // now we take out detectors listed as skipped
  auto& skipped = vm["skipModules"].as<std::vector<std::string>>();
  for (auto& s : skipped) {
    auto iter = std::find(active.begin(), active.end(), s);
    if (iter != active.end()) {
      // take it out
      active.erase(iter);
    }
  }

  mConfigData.mGenerator = vm["generator"].as<std::string>();
  mConfigData.mTrigger = vm["trigger"].as<std::string>();
  mConfigData.mNEvents = vm["nEvents"].as<unsigned int>();
  mConfigData.mExtKinFileName = vm["extKinFile"].as<std::string>();
  mConfigData.mEmbedIntoFileName = vm["embedIntoFile"].as<std::string>();
  mConfigData.mStartEvent = vm["startEvent"].as<unsigned int>();
  mConfigData.mBMax = vm["bMax"].as<float>();
  mConfigData.mIsMT = vm["isMT"].as<bool>();
  mConfigData.mOutputPrefix = vm["outPrefix"].as<std::string>();
  mConfigData.mLogSeverity = vm["logseverity"].as<std::string>();
  mConfigData.mLogVerbosity = vm["logverbosity"].as<std::string>();
  mConfigData.mKeyValueTokens = vm["configKeyValues"].as<std::string>();
  mConfigData.mConfigFile = vm["configFile"].as<std::string>();
  mConfigData.mPrimaryChunkSize = vm["chunkSize"].as<unsigned int>();
  mConfigData.mInternalChunkSize = vm["chunkSizeI"].as<int>();
  mConfigData.mStartSeed = vm["seed"].as<int>();
  mConfigData.mSimWorkers = vm["nworkers"].as<int>();
  mConfigData.mTimestamp = vm["timestamp"].as<long>();
  mConfigData.mCCDBUrl = vm["CCDBUrl"].as<std::string>();
  mConfigData.mAsService = vm["asservice"].as<bool>();
  if (vm.count("noemptyevents")) {
    mConfigData.mFilterNoHitEvents = true;
  }
  mConfigData.mField = std::stoi((vm["field"].as<std::string>()).substr(0, (vm["field"].as<std::string>()).rfind("U")));
  mConfigData.mUniformField = (vm["field"].as<std::string>()).find("U") != std::string::npos;
  return true;
}

bool SimConfig::resetFromArguments(int argc, char* argv[])
{
  namespace bpo = boost::program_options;

  // Arguments parsing
  bpo::variables_map vm;
  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message.");
  initOptions(desc);

  try {
    bpo::store(parse_command_line(argc, argv, desc), vm);

    // help
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return false;
    }

    bpo::notify(vm);
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing command line arguments; Available options:\n";

    std::cerr << desc << std::endl;
    return false;
  }

  return resetFromParsedMap(vm);
}

namespace o2::conf
{
// returns a reconfig struct given a configuration string (boost program options format)
bool parseSimReconfigFromString(std::string const& argumentstring, SimReconfigData& data)
{
  namespace bpo = boost::program_options;

  bpo::options_description options("Allowed options");

  options.add_options()(
    "nEvents,n", bpo::value<unsigned int>(&data.nEvents)->default_value(1), "number of events")(
    "generator,g", bpo::value<std::string>(&data.generator)->default_value("boxgen"), "Event generator to be used.")(
    "trigger,t", bpo::value<std::string>(&data.trigger)->default_value(""), "Event generator trigger to be used.")(
    "startEvent", bpo::value<unsigned int>(&data.startEvent)->default_value(0), "index of first event to be used (when applicable)")(
    "extKinFile", bpo::value<std::string>(&data.extKinfileName)->default_value("Kinematics.root"),
    "name of kinematics file for event generator from file (when applicable)")(
    "embedIntoFile", bpo::value<std::string>(&data.embedIntoFileName)->default_value(""),
    "filename containing the reference events to be used for the embedding")(
    "bMax,b", bpo::value<float>(&data.mBMax)->default_value(0.), "maximum value for impact parameter sampling (when applicable)")(
    "outPrefix,o", bpo::value<std::string>(&data.outputPrefix)->default_value("o2sim"), "prefix of output files")(
    "outDir,d", bpo::value<std::string>(&data.outputDir), "directory where to put simulation output (created when non-existant")(
    "configKeyValues", bpo::value<std::string>(&data.keyValueTokens)->default_value(""), "semicolon separated key=value strings (e.g.: 'TPC.gasDensity=1;...")(
    "configFile", bpo::value<std::string>(&data.configFile)->default_value(""), "Path to an INI or JSON configuration file")(
    "chunkSize", bpo::value<unsigned int>(&data.primaryChunkSize)->default_value(500), "max size of primary chunk (subevent) distributed by server")(
    "seed", bpo::value<int>(&data.startSeed)->default_value(-1), "initial seed (default: -1 random)")(
    "stop", bpo::value<bool>(&data.stop)->default_value(false), "control command to shut down daemon");

  bpo::variables_map vm;
  try {
    bpo::store(bpo::command_line_parser(bpo::split_unix(argumentstring))
                 .options(options)
                 .run(),
               vm);
    bpo::notify(vm);
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing ReConfig data; Available options:\n";
    std::cerr << options << std::endl;
    return false;
  }
  return true;
}

} // namespace o2::conf

ClassImp(o2::conf::SimConfig);
