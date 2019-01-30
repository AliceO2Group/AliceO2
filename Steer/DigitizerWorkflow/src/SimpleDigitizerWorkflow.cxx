// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsBase/Propagator.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DeviceSpec.h"
#include "SimReaderSpec.h"
#include "CollisionTimePrinter.h"
#include "DetectorsCommonDataFormats/DetID.h"

// for TPC
#include "TPCDigitizerSpec.h"
#include "TPCDigitRootWriterSpec.h"
#include "TPCBase/Sector.h"
#include "TPCBase/CDBInterface.h"
#include "TPCWorkflow/RecoWorkflow.h"
// needed in order to init the **SHARED** polyadist file (to be done before the digitizers initialize)
#include "TPCSimulation/GEMAmplification.h"

// for ITSMFT
#include "ITSMFTDigitizerSpec.h"
#include "ITSMFTDigitWriterSpec.h"

// for TOF
#include "TOFDigitizerSpec.h"
#include "TOFDigitWriterSpec.h"
#include "TOFClusterizerSpec.h"
#include "TOFClusterWriterSpec.h"

// for FIT
#include "FITDigitizerSpec.h"
#include "FITDigitWriterSpec.h"

// for EMCal
#include "EMCALDigitizerSpec.h"
#include "EMCALDigitWriterSpec.h"

// for HMPID
#include "HMPIDDigitizerSpec.h"
#include "HMPIDDigitWriterSpec.h"

// for TRD
#include "TRDDigitizerSpec.h"
#include "TRDDigitWriterSpec.h"

//for MUON MCH
#include "MCHDigitizerSpec.h"
#include "MCHDigitWriterSpec.h"

// GRP
#include "DataFormatsParameters/GRPObject.h"
#include "GRPUpdaterSpec.h"

#include <cstdlib>
// this is somewhat assuming that a DPL workflow will run on one node
#include <thread> // to detect number of hardware threads
#include <string>
#include <sstream>
#include <cmath>
#include <unistd.h> // for getppid

using namespace o2::framework;

// ------------------------------------------------------------------

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  // we customize the completion policy for the writer since it should stream immediately
  auto matcher = [](DeviceSpec const& device) {
    bool matched = device.name == "TPCDigitWriter";
    if (matched) {
      LOG(INFO) << "DPL completion policy for " << device.name << " customized";
    }
    return matched;
  };

  auto policy = [](gsl::span<o2::framework::PartRef const> const& inputs) {
    return CompletionPolicy::CompletionOp::Consume;
  };

  policies.push_back({ CompletionPolicy{ "process-any", matcher, policy } });
}

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // for the TPC it is useful to take at most half of the available (logical) cores due to memory requirements
  int defaultlanes = std::thread::hardware_concurrency() / 2;
  std::string laneshelp("Number of tpc processing lanes. A lane is a pipeline of algorithms.");
  workflowOptions.push_back(
    ConfigParamSpec{ "tpc-lanes", VariantType::Int, defaultlanes, { laneshelp } });

  std::string sectorshelp("Comma separated string of tpc sectors to treat. (Default is all)");
  workflowOptions.push_back(
    ConfigParamSpec{ "tpc-sectors", VariantType::String, "all", { sectorshelp } });

  std::string onlyhelp("Comma separated list of detectors to accept. Takes precedence over the skipDet option. (Default is none)");
  workflowOptions.push_back(
    ConfigParamSpec{ "onlyDet", VariantType::String, "none", { onlyhelp } });

  std::string skiphelp("Comma separate list of detectors to skip/ignore. (Default is none)");
  workflowOptions.push_back(
    ConfigParamSpec{ "skipDet", VariantType::String, "none", { skiphelp } });

  // we support only output type 'tracks' for the moment
  std::string tpcrthelp("Run TPC reco workflow to specified output type, currently supported: 'tracks'");
  workflowOptions.push_back(
    ConfigParamSpec{ "tpc-reco-type", VariantType::String, "", { tpcrthelp } });
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

// extract num TPC lanes, a lane is a streaming line of processors (digitizer-clusterizer-etc)
// by default this will be std::max(the number of physical cores, numberofsectors)
// as a temporary means to fully use a machine and as a way to play with different topologies
int getNumTPCLanes(std::vector<int> const& sectors, ConfigContext const& configcontext)
{
  auto lanes = configcontext.options().get<int>("tpc-lanes");
  if (lanes < 0) {
    LOG(FATAL) << "tpc-lanes needs to be positive\n";
    return 0;
  }
  // crosscheck with sectors
  return std::min(lanes, (int)sectors.size());
}

// ------------------------------------------------------------------

void initTPC()
{
  // We only want to do this for the DPL master
  // I am not aware of an easy way to query if "I am DPL master" so
  // using for the moment a mechanism defining/setting an environment variable
  // with the parent ID and query inside forks if this environment variable exists
  // (it assumes fundamentally that the master executes this function first)
  std::stringstream streamthis;
  std::stringstream streamparent;

  streamthis << "TPCGEMINIT_PID" << getpid();
  streamparent << "TPCGEMINIT_PID" << getppid();
  if (getenv(streamparent.str().c_str())) {
    LOG(DEBUG) << "GEM ALREADY INITIALIZED ... SKIPPING HERE";
    return;
  }

  LOG(DEBUG) << "INITIALIZING TPC GEMAmplification";
  setenv(streamthis.str().c_str(), "ON", 1);

  auto& cdb = o2::TPC::CDBInterface::instance();
  cdb.setUseDefaults();
  // by invoking this constructor we make sure that a common file will be created
  // in future we should take this from OCDB and just forward per message
  const static auto& ampl = o2::TPC::GEMAmplification::instance();
}

// ------------------------------------------------------------------

void extractTPCSectors(std::vector<int>& sectors, ConfigContext const& configcontext)
{
  auto sectorsstring = configcontext.options().get<std::string>("tpc-sectors");
  if (sectorsstring.compare("all") != 0) {
    // we expect them to be , separated
    std::stringstream ss(sectorsstring);
    std::vector<std::string> stringtokens;
    while (ss.good()) {
      std::string substr;
      getline(ss, substr, ',');
      stringtokens.push_back(substr);
    }
    // now try to convert each token to int
    for (auto& token : stringtokens) {
      try {
        auto s = std::stoi(token);
        sectors.emplace_back(s);
      } catch (std::invalid_argument e) {
      }
    }
    return;
  }

  // all sectors otherwise by default
  for (int s = 0; s < o2::TPC::Sector::MAXSECTOR; ++s) {
    sectors.emplace_back(s);
  }
}

// ------------------------------------------------------------------

bool wantCollisionTimePrinter()
{
  if (const char* f = std::getenv("DPL_COLLISION_TIME_PRINTER")) {
    return true;
  }
  return false;
}

// ------------------------------------------------------------------

std::shared_ptr<o2::parameters::GRPObject> readGRP(std::string inputGRP = "o2sim_grp.root")
{
  // init magnetic field
  o2::Base::Propagator::initFieldFromGRP(inputGRP);

  auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  if (!grp) {
    LOG(ERROR) << "This workflow needs a valid GRP file to start";
  }
  return std::shared_ptr<o2::parameters::GRPObject>(grp);
}

// ------------------------------------------------------------------

// Split a given string on a separator character
std::vector<std::string> splitString(std::string src, char sep)
{
  std::vector<std::string> fields;
  std::string token;
  std::istringstream ss(src);

  while (std::getline(ss, token, sep)) {
    if (!token.empty()) {
      fields.push_back(token);
    }
  }

  return fields;
}
// ------------------------------------------------------------------

// Filters detectors based on a white/black list provided via the onlyDet/skipDet CLI args
struct DetFilterer {
  // detlist:     A character-separated list of detectors
  // unsetVal:    The value when the option is unset
  // separator:   The character that separates the list of detectors defined in option
  // mustContain: The nature of this DetFilterer. If true, it is a white lister
  //              i.e. option defines the list of allowed detectors. If false
  //              it is a black lister i.e defines the list of disallowed detectors.
  DetFilterer(std::string detlist, std::string unsetVal, char separator, bool doWhiteListing)
  {
    // option is not set, nothing to do
    if (detlist.compare(unsetVal) == 0) {
      return;
    }

    std::vector<std::string> tokens = splitString(detlist, separator);

    // Convert a vector of strings to one of o2::detectors::DetID
    for (auto token : tokens) {
      ids.emplace_back(token.c_str());
    }

    isWhiteLister = doWhiteListing;
  }

  // isSet determines if a detector list was provided
  // against which to filter
  bool isSet()
  {
    return ids.size() > 0;
  }

  // accept determines if a given detector should be accepted
  bool accept(o2::detectors::DetID id)
  {
    bool found = std::find(ids.begin(), ids.end(), id) != ids.end();
    return found == isWhiteLister;
  }

 private:
  std::vector<o2::detectors::DetID> ids;
  bool isWhiteLister; // true = accept only detectors in the ids vector
};

// Helper function to define a white listing DetFilterer
DetFilterer whitelister(std::string optionVal, std::string unsetValue, char separator)
{
  return DetFilterer(optionVal, unsetValue, separator, true);
}

// Helper function to define a black listing DetFilterer
DetFilterer blacklister(std::string optionVal, std::string unsetValue, char separator)
{
  return DetFilterer(optionVal, unsetValue, separator, false);
}

// ------------------------------------------------------------------

/// This function is required to be implemented to define the workflow
/// specifications
WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Reserve one entry which fill be filled with the SimReaderSpec
  // at the end. This places the processor at the beginning of the
  // workflow in the upper left corner of the GUI.
  WorkflowSpec specs(1);

  // First, read the GRP to detect which components need instantiations
  // (for the moment this assumes the file o2sim_grp.root to be in the current directory)
  const auto grp = readGRP();
  if (!grp) {
    return specs;
  }

  // onlyDet takes precedence on skipDet
  DetFilterer filterers[2] = {
    whitelister(configcontext.options().get<std::string>("onlyDet"), "none", ','),
    blacklister(configcontext.options().get<std::string>("skipDet"), "none", ',')
  };

  auto accept = [&configcontext, &filterers](o2::detectors::DetID id) {
    for (auto& f : filterers) {
      if (f.isSet()) {
        return f.accept(id);
      }
    }

    // accept all if neither onlyDet/skipDet are provided
    return true;
  };

  // lambda to extract detectors which are enabled in the workflow
  // will complain if user gave wrong input in construction of DetID
  auto isEnabled = [&configcontext, &filterers, accept, grp](o2::detectors::DetID id) {
    auto accepted = accept(id);
    bool is_ingrp = grp->isDetReadOut(id);
    LOG(INFO) << id.getName()
              << " is in grp? " << (is_ingrp ? "yes" : "no") << ";"
              << " is skipped? " << (!accepted ? "yes" : "no");
    return accepted && is_ingrp;
  };

  std::vector<o2::detectors::DetID> detList; // list of participating detectors
  int fanoutsize = 0;
  if (wantCollisionTimePrinter()) {
    specs.emplace_back(o2::steer::getCollisionTimePrinter(fanoutsize++));
  }

  // the TPC part
  // we need to init this anyway since TPC is treated a bit special (for the moment)
  initTPC();
  // keeps track of which subchannels correspond to tpc channels
  auto tpclanes = std::make_shared<std::vector<int>>();
  // keeps track of which tpc sectors to process
  auto tpcsectors = std::make_shared<std::vector<int>>();

  if (isEnabled(o2::detectors::DetID::TPC)) {
    extractTPCSectors(*tpcsectors.get(), configcontext);
    auto lanes = getNumTPCLanes(*tpcsectors.get(), configcontext);
    detList.emplace_back(o2::detectors::DetID::TPC);

    for (int l = 0; l < lanes; ++l) {
      specs.emplace_back(o2::TPC::getTPCDigitizerSpec(fanoutsize, (l == 0)));
      tpclanes->emplace_back(fanoutsize); // this records that TPC is "listening under this subchannel"
      fanoutsize++;
    }

    auto tpcRecoOutputType = configcontext.options().get<std::string>("tpc-reco-type");
    if (tpcRecoOutputType.empty()) {
      // for writing digits to disc
      specs.emplace_back(o2::TPC::getTPCDigitRootWriterSpec(lanes));
    } else {
      // attach the TPC reco workflow
      auto tpcRecoWorkflow = o2::TPC::RecoWorkflow::getWorkflow(true, lanes, "digitizer", tpcRecoOutputType.c_str());
      specs.insert(specs.end(), tpcRecoWorkflow.begin(), tpcRecoWorkflow.end());
    }
  }

  // the ITS part
  if (isEnabled(o2::detectors::DetID::ITS)) {
    detList.emplace_back(o2::detectors::DetID::ITS);
    // connect the ITS digitization
    specs.emplace_back(o2::ITSMFT::getITSDigitizerSpec(fanoutsize++));
    // connect ITS digit writer
    specs.emplace_back(o2::ITSMFT::getITSDigitWriterSpec());
  }

  // the MFT part
  if (isEnabled(o2::detectors::DetID::MFT)) {
    detList.emplace_back(o2::detectors::DetID::MFT);
    // connect the MFT digitization
    specs.emplace_back(o2::ITSMFT::getMFTDigitizerSpec(fanoutsize++));
    // connect MFT digit writer
    specs.emplace_back(o2::ITSMFT::getMFTDigitWriterSpec());
  }

  // the TOF part
  if (isEnabled(o2::detectors::DetID::TOF)) {
    detList.emplace_back(o2::detectors::DetID::TOF);
    // connect the TOF digitization
    specs.emplace_back(o2::tof::getTOFDigitizerSpec(fanoutsize++));
    // add TOF digit writer
    specs.emplace_back(o2::tof::getTOFDigitWriterSpec());
    // add TOF clusterer
    specs.emplace_back(o2::tof::getTOFClusterizerSpec());
    // add TOF cluster writer
    specs.emplace_back(o2::tof::getTOFClusterWriterSpec());
  }

  // the FIT part
  if (isEnabled(o2::detectors::DetID::FIT)) {
    detList.emplace_back(o2::detectors::DetID::FIT);
    // connect the FIT digitization
    specs.emplace_back(o2::fit::getFITDigitizerSpec(fanoutsize++));
    // connect the FIT digit writer
    specs.emplace_back(o2::fit::getFITDigitWriterSpec());
  }

  // the EMCal part
  if (isEnabled(o2::detectors::DetID::EMC)) {
    detList.emplace_back(o2::detectors::DetID::EMC);
    // connect the EMCal digitization
    specs.emplace_back(o2::emcal::getEMCALDigitizerSpec(fanoutsize++));
    // connect the EMCal digit writer
    specs.emplace_back(o2::emcal::getEMCALDigitWriterSpec());
  }

  // add HMPID
  if (isEnabled(o2::detectors::DetID::HMP)) {
    detList.emplace_back(o2::detectors::DetID::HMP);
    // connect the HMP digitization
    specs.emplace_back(o2::hmpid::getHMPIDDigitizerSpec(fanoutsize++));
    // connect the HMP digit writer
    specs.emplace_back(o2::hmpid::getHMPIDDigitWriterSpec());
  }

  // add TRD
  if (isEnabled(o2::detectors::DetID::TRD)) {
    detList.emplace_back(o2::detectors::DetID::TRD);
    // connect the TRD digitization
    specs.emplace_back(o2::trd::getTRDDigitizerSpec(fanoutsize++));
    // connect the TRD digit writer
    specs.emplace_back(o2::trd::getTRDDigitWriterSpec());
  }

  //add MUON MCH
  if (isEnabled(o2::detectors::DetID::MCH)) {
    detList.emplace_back(o2::detectors::DetID::MCH);
    //connect the MUON MCH digitization
    specs.emplace_back(o2::mch::getMCHDigitizerSpec(fanoutsize++));
    //connect the MUON MCH digit writer
    specs.emplace_back(o2::mch::getMCHDigitWriterSpec());
  }

  // GRP updater: must come after all detectors since requires their list
  specs.emplace_back(o2::parameters::getGRPUpdaterSpec(detList));

  // The SIM Reader. NEEDS TO BE LAST
  specs[0] = o2::steer::getSimReaderSpec(fanoutsize, tpcsectors, tpclanes);

  return specs;
}
