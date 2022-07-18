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

#include <SimConfig/SimConfig.h>
#include <DetectorsCommonDataFormats/DetID.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <FairLogger.h>
#include <thread>
#include <cmath>
#include <chrono>
#include <regex>

using namespace o2::conf;
namespace bpo = boost::program_options;

void SimConfig::initOptions(boost::program_options::options_description& options)
{
  int nsimworkersdefault = std::max(1u, std::thread::hardware_concurrency() / 2);
  options.add_options()(
    "mcEngine,e", bpo::value<std::string>()->default_value("TGeant4"), "VMC backend to be used.")(
    "generator,g", bpo::value<std::string>()->default_value("boxgen"), "Event generator to be used.")(
    "trigger,t", bpo::value<std::string>()->default_value(""), "Event generator trigger to be used.")(
    "modules,m", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>({"all"}), "all modules"), "list of modules included in geometry")(
    "skipModules", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>({""}), ""), "list of modules excluded in geometry (precendence over -m")(
    "readoutDetectors", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>(), ""), "list of detectors creating hits, all if not given; added to to active modules")(
    "skipReadoutDetectors", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>(), ""), "list of detectors to skip hit creation (precendence over --readoutDetectors")(
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
    "field", bpo::value<std::string>()->default_value("-5"), "L3 field rounded to kGauss, allowed values +-2,+-5 and 0; +-<intKGaus>U for uniform field; \"ccdb\" for taking it from CCDB ")(
    "nworkers,j", bpo::value<int>()->default_value(nsimworkersdefault), "number of parallel simulation workers (only for parallel mode)")(
    "noemptyevents", "only writes events with at least one hit")(
    "CCDBUrl", bpo::value<std::string>()->default_value("http://alice-ccdb.cern.ch"), "URL for CCDB to be used.")(
    "timestamp", bpo::value<uint64_t>(), "global timestamp value in ms (for anchoring) - default is now ... or beginning of run if ALICE run number was given")(
    "run", bpo::value<int>()->default_value(-1), "ALICE run number")(
    "asservice", bpo::value<bool>()->default_value(false), "run in service/server mode")(
    "noGeant", bpo::bool_switch(), "prohibits any Geant transport/physics (by using tight cuts)");
}

void SimConfig::determineActiveModules(std::vector<std::string> const& inputargs, std::vector<std::string> const& skippedModules, std::vector<std::string>& activeModules)
{
  using o2::detectors::DetID;

  // input args is a vector of module strings as obtained from the -m,--modules options
  // of SimConfig
  activeModules = inputargs;
  if (activeModules.size() == 1 && activeModules[0] == "all") {
    activeModules.clear();
    for (int d = DetID::First; d <= DetID::Last; ++d) {
#ifdef ENABLE_UPGRADES
      if (d != DetID::IT3 && d != DetID::TRK && d != DetID::FT3 && d != DetID::FCT) {
        activeModules.emplace_back(DetID::getName(d));
      }
#else
      activeModules.emplace_back(DetID::getName(d));
#endif
    }
    // add passive components manually (make a PassiveDetID for them!)
    activeModules.emplace_back("HALL");
    activeModules.emplace_back("MAG");
    activeModules.emplace_back("DIPO");
    activeModules.emplace_back("COMP");
    activeModules.emplace_back("PIPE");
    activeModules.emplace_back("ABSO");
    activeModules.emplace_back("SHIL");
  }
  // now we take out detectors listed as skipped
  for (auto& s : skippedModules) {
    auto iter = std::find(activeModules.begin(), activeModules.end(), s);
    if (iter != activeModules.end()) {
      // take it out
      activeModules.erase(iter);
    }
  }
}

void SimConfig::determineReadoutDetectors(std::vector<std::string> const& activeModules, std::vector<std::string> const& enableReadout, std::vector<std::string> const& disableReadout, std::vector<std::string>& readoutDetectors)
{
  using o2::detectors::DetID;

  readoutDetectors.clear();

  auto isDet = [](std::string const& s) {
    return DetID::nameToID(s.c_str()) >= DetID::First;
  };

  if (enableReadout.empty()) {
    // if no readout explicitly given, use all detectors from active modules
    for (auto& am : activeModules) {
      if (!isDet(am)) {
        // either we found a passive module or one with disabled readout ==> skip
        continue;
      }
      readoutDetectors.emplace_back(am);
    }
  } else {
    for (auto& er : enableReadout) {
      if (!isDet(er)) {
        // either we found a passive module or one with disabled readout ==> skip
        LOG(fatal) << "Enabled readout for " << er << " which is not a detector.";
      }
      if (std::find(activeModules.begin(), activeModules.end(), er) == activeModules.end()) {
        // add to active modules if not yet there
        LOG(fatal) << "Module " << er << " not constructed and cannot be used for readout (make sure it is contained in -m option).";
      }
      readoutDetectors.emplace_back(er);
    }
  }
  for (auto& dr : disableReadout) {
    if (!isDet(dr)) {
      // either we found a passive module or one with disabled readout ==> skip
      LOG(fatal) << "Disabled readout for " << dr << " which is not a detector.";
    }
    if (std::find(activeModules.begin(), activeModules.end(), dr) == activeModules.end()) {
      // add to active modules if not yet there
      LOG(fatal) << "Module " << dr << " not constructed, makes no sense to disable its readout (make sure it is contained in -m option).";
    }
    auto iter = std::find(readoutDetectors.begin(), readoutDetectors.end(), dr);
    if (iter != readoutDetectors.end()) {
      readoutDetectors.erase(iter);
    }
  }
}

bool SimConfig::resetFromParsedMap(boost::program_options::variables_map const& vm)
{
  using o2::detectors::DetID;
  mConfigData.mMCEngine = vm["mcEngine"].as<std::string>();

  // get final set of active Modules
  determineActiveModules(vm["modules"].as<std::vector<std::string>>(), vm["skipModules"].as<std::vector<std::string>>(), mConfigData.mActiveModules);
  const auto& activeModules = mConfigData.mActiveModules;

  // get final set of detectors which are readout
  determineReadoutDetectors(activeModules, vm["readoutDetectors"].as<std::vector<std::string>>(), vm["skipReadoutDetectors"].as<std::vector<std::string>>(), mConfigData.mReadoutDetectors);

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
  if (vm.count("timestamp")) {
    mConfigData.mTimestamp = vm["timestamp"].as<uint64_t>();
    mConfigData.mTimestampMode = kManual;
  } else {
    mConfigData.mTimestamp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
    mConfigData.mTimestampMode = kNow;
  }
  mConfigData.mRunNumber = vm["run"].as<int>();
  mConfigData.mCCDBUrl = vm["CCDBUrl"].as<std::string>();
  mConfigData.mAsService = vm["asservice"].as<bool>();
  mConfigData.mNoGeant = vm["noGeant"].as<bool>();
  if (vm.count("noemptyevents")) {
    mConfigData.mFilterNoHitEvents = true;
  }

  // analyse field options
  // either: "ccdb" or +-2[U],+-5[U] and 0[U]; +-<intKGaus>U
  auto& fieldstring = vm["field"].as<std::string>();
  std::regex re("(ccdb)|([+-]?[250]U?)");
  if (!std::regex_match(fieldstring, re)) {
    LOG(error) << "Invalid field option";
    return false;
  }
  if (fieldstring == "ccdb") {
    mConfigData.mFieldMode = SimFieldMode::kCCDB;
  } else if (fieldstring.find("U") != std::string::npos) {
    mConfigData.mFieldMode = SimFieldMode::kUniform;
  }
  if (fieldstring != "ccdb") {
    mConfigData.mField = std::stoi((vm["field"].as<std::string>()).substr(0, (vm["field"].as<std::string>()).rfind("U")));
  }
  if (!parseFieldString(fieldstring, mConfigData.mField, mConfigData.mFieldMode)) {
    return false;
  }

  return true;
}

bool SimConfig::parseFieldString(std::string const& fieldstring, int& fieldvalue, SimFieldMode& mode)
{
  // analyse field options
  // either: "ccdb" or +-2[U],+-5[U] and 0[U]; +-<intKGaus>U
  std::regex re("(ccdb)|([+-]?[250]U?)");
  if (!std::regex_match(fieldstring, re)) {
    LOG(error) << "Invalid field option";
    return false;
  }
  if (fieldstring == "ccdb") {
    mode = SimFieldMode::kCCDB;
  } else if (fieldstring.find("U") != std::string::npos) {
    mode = SimFieldMode::kUniform;
  }
  if (fieldstring != "ccdb") {
    fieldvalue = std::stoi(fieldstring.substr(0, fieldstring.rfind("U")));
  }
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
