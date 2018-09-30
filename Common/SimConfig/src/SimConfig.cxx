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

using namespace o2::conf;
namespace bpo = boost::program_options;

void SimConfig::initOptions(boost::program_options::options_description& options)
{
  options.add_options()(
    "mcEngine,e", bpo::value<std::string>()->default_value("TGeant3"), "VMC backend to be used.")(
    "generator,g", bpo::value<std::string>()->default_value("boxgen"), "Event generator to be used.")(
    "modules,m", bpo::value<std::vector<std::string>>()->multitoken()->default_value(
                   std::vector<std::string>({ "all" }), "all modules"),
    "list of detectors")("nEvents,n", bpo::value<unsigned int>()->default_value(1), "number of events")(
    "startEvent", bpo::value<unsigned int>()->default_value(0), "index of first event to be used (when applicable)")(
    "extKinFile", bpo::value<std::string>()->default_value("Kinematics.root"),
    "name of kinematics file for event generator from file (when applicable)")(
    "bMax,b", bpo::value<float>()->default_value(0.), "maximum value for impact parameter sampling (when applicable)")(
    "origin", bpo::value<std::vector<float>>()->multitoken()->default_value(std::vector<float>({0., 0., 0.}), "0 0 0"), "position of the interaction diamond: x y z (cm)")(
    "sigmaO", bpo::value<std::vector<float>>()->multitoken()->default_value(std::vector<float>({0.001, 0.001, 6.}), "0.001 0.001 6"), "width of the interaction diamond: x y z (cm)")(
    "isMT", bpo::value<bool>()->default_value(false), "multi-threaded mode (Geant4 only")(
    "outPrefix,o", bpo::value<std::string>()->default_value("o2sim"), "prefix of output files");
}

bool SimConfig::resetFromParsedMap(boost::program_options::variables_map const& vm)
{
  using o2::detectors::DetID;
  mConfigData.mMCEngine = vm["mcEngine"].as<std::string>();
  mConfigData.mActiveDetectors = vm["modules"].as<std::vector<std::string>>();
  if (mConfigData.mActiveDetectors.size() == 1 && mConfigData.mActiveDetectors[0] == "all") {
    mConfigData.mActiveDetectors.clear();
    for (int d = DetID::First; d <= DetID::Last; ++d) {
      mConfigData.mActiveDetectors.emplace_back(DetID::getName(d));
    }
  }
  mConfigData.mGenerator = vm["generator"].as<std::string>();
  mConfigData.mNEvents = vm["nEvents"].as<unsigned int>();
  mConfigData.mExtKinFileName = vm["extKinFile"].as<std::string>();
  mConfigData.mStartEvent = vm["startEvent"].as<unsigned int>();
  mConfigData.mBMax = vm["bMax"].as<float>();
  mConfigData.mOrigin = vm["origin"].as<std::vector<float>>();
  mConfigData.mSigmaO = vm["sigmaO"].as<std::vector<float>>();
  mConfigData.mIsMT = vm["isMT"].as<bool>();
  mConfigData.mOutputPrefix = vm["outPrefix"].as<std::string>();
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
    bpo::store(parse_command_line(argc, argv, desc, bpo::command_line_style::unix_style ^ bpo::command_line_style::allow_short), vm);
    //    bpo::store(parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << "exception caught\n";
    return false;
  }

  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return false;
  }

  return resetFromParsedMap(vm);
}

ClassImp(o2::conf::SimConfig);
