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
#include <boost/program_options.hpp>
#include <iostream>

using namespace o2::conf;

bool SimConfig::resetFromArguments(int argc, char* argv[])
{
  namespace bpo = boost::program_options;

  // Arguments parsing
  bpo::variables_map vm;
  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message.")(
    "mcEngine,e", bpo::value<std::string>()->default_value("TGeant3"), "VMC backend to be used.")(
    "generator,g", bpo::value<std::string>()->default_value("boxgen"), "Event generator to be used.")(
    "modules,m",
    bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>({ "EMCAL TOF TPC TRD" }), "EMCAL TOF TPC TRD"),
    "list of detectors")
    ("nEvents,n", bpo::value<unsigned int>()->default_value(1), "number of events")
    ("startEvent", bpo::value<unsigned int>()->default_value(0), "index of first event to be used (when applicable)")
    ("extKinFile", bpo::value<std::string>()->default_value("Kinematics.root"), "name of kinematics file for event generator from file (when applicable)")
    ("bMax,b", bpo::value<float>()->default_value(0.), "maximum value for impact parameter sampling (when applicable)");

  try {
    bpo::store(parse_command_line(argc, argv, desc), vm);
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

  mMCEngine = vm["mcEngine"].as<std::string>();
  mActiveDetectors = vm["modules"].as<std::vector<std::string>>();
  mGenerator = vm["generator"].as<std::string>();
  mNEvents = vm["nEvents"].as<unsigned int>();
  mExtKinFileName = vm["extKinFile"].as<std::string>();
  mStartEvent = vm["startEvent"].as<unsigned int>();
  mBMax = vm["bMax"].as<float>();
  
  return true;
}

ClassImp(o2::conf::SimConfig);
