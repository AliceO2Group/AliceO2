// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/program_options.hpp>

#include "Framework/RootSerializationSupport.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "SimReaderSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "CommonUtils/ConfigurableParam.h"

// for TPC
#include "TPCDigitizerSpec.h"
#include "TPCDigitRootWriterSpec.h"
#include "TPCBase/Sector.h"
#include "TPCBase/CDBInterface.h"
// needed in order to init the **SHARED** polyadist file (to be done before the digitizers initialize)
#include "TPCSimulation/GEMAmplification.h"

// for ITSMFT
#include "ITSMFTDigitizerSpec.h"
#include "ITSMFTWorkflow/DigitWriterSpec.h"

// for TOF
#include "TOFDigitizerSpec.h"
#include "TOFWorkflowUtils/TOFDigitWriterSpec.h"

// for FT0
#include "FT0DigitizerSpec.h"
#include "FT0DigitWriterSpec.h"

// for FV0
#include "FV0DigitizerSpec.h"
#include "FV0DigitWriterSpec.h"

// for FDD
#include "FDDDigitizerSpec.h"
#include "FDDDigitWriterSpec.h"

// for EMCal
#include "EMCALDigitizerSpec.h"
#include "EMCALDigitWriterSpec.h"

// for HMPID
#include "HMPIDDigitizerSpec.h"
#include "HMPIDDigitWriterSpec.h"

// for TRD
#include "TRDWorkflow/TRDDigitizerSpec.h"
#include "TRDWorkflow/TRDDigitWriterSpec.h"
#include "TRDWorkflow/TRDTrapSimulatorSpec.h"
#include "TRDWorkflow/TRDTrackletWriterSpec.h"

//for MUON MCH
#include "MCHDigitizerSpec.h"
#include "MCHDigitWriterSpec.h"

// for MID
#include "MIDDigitizerSpec.h"
#include "MIDDigitWriterSpec.h"

// for PHOS
#include "PHOSDigitizerSpec.h"
#include "PHOSDigitWriterSpec.h"

// for CPV
#include "CPVDigitizerSpec.h"
#include "CPVDigitWriterSpec.h"

// for ZDC
#include "ZDCDigitizerSpec.h"
#include "ZDCDigitWriterSpec.h"

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

bool gIsMaster = false; // a global variable indicating if this is the master workflow process
                        // (an individual DPL processor will still build the workflow but is not
                        //  considered master)

// ------------------------------------------------------------------

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  // we customize the completion policy for the writer since it should stream immediately
  policies.push_back(CompletionPolicyHelpers::defineByName("TPCDigitWriter", CompletionPolicy::CompletionOp::Consume));
  policies.push_back(CompletionPolicyHelpers::defineByName("TPCDigitizer.*", CompletionPolicy::CompletionOp::Consume));
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-cluster-decoder.*", CompletionPolicy::CompletionOp::Consume));
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-clusterer.*", CompletionPolicy::CompletionOp::Consume));
}

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // for the TPC it is useful to take at most half of the available (logical) cores due to memory requirements
  int defaultlanes = std::max(1u, std::thread::hardware_concurrency() / 2);
  std::string laneshelp("Number of tpc processing lanes. A lane is a pipeline of algorithms.");
  workflowOptions.push_back(
    ConfigParamSpec{"tpc-lanes", VariantType::Int, defaultlanes, {laneshelp}});

  std::string sectorshelp("List of TPC sectors, comma separated ranges, e.g. 0-3,7,9-15");
  std::string sectorDefault = "0-" + std::to_string(o2::tpc::Sector::MAXSECTOR - 1);
  workflowOptions.push_back(
    ConfigParamSpec{"tpc-sectors", VariantType::String, sectorDefault.c_str(), {sectorshelp}});

  std::string onlyhelp("Comma separated list of detectors to accept. Takes precedence over the skipDet option. (Default is none)");
  workflowOptions.push_back(
    ConfigParamSpec{"onlyDet", VariantType::String, "none", {onlyhelp}});

  std::string skiphelp("Comma separate list of detectors to skip/ignore. (Default is none)");
  workflowOptions.push_back(
    ConfigParamSpec{"skipDet", VariantType::String, "none", {skiphelp}});

  // we support only output type 'tracks' for the moment
  std::string tpcrthelp("deprecated option, please connect workflows on the command line by pipe");
  workflowOptions.push_back(
    ConfigParamSpec{"tpc-reco-type", VariantType::String, "", {tpcrthelp}});

  std::string simhelp("Comma separated list of simulation prefixes (for background, signal productions)");
  workflowOptions.push_back(
    ConfigParamSpec{"sims", VariantType::String, "o2sim", {simhelp}});

  // option allowing to set parameters
  std::string keyvaluehelp("Semicolon separated key=value strings (e.g.: 'TPC.gasDensity=1;...')");
  workflowOptions.push_back(
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {keyvaluehelp}});
  workflowOptions.push_back(
    ConfigParamSpec{"configFile", VariantType::String, "", {"configuration file for configurable parameters"}});

  // option to disable MC truth
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"disable  mc-truth"}});

  // option to use/not use CCDB for TOF
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb-tof", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}});

  // option to use or not use the Trap Simulator after digitisation (debate of digitization or reconstruction is for others)
  workflowOptions.push_back(ConfigParamSpec{"enable-trd-trapsim", VariantType::Bool, false, {"enable the trap simulation of the TRD"}});
}

void customize(std::vector<o2::framework::DispatchPolicy>& policies)
{
  using DispatchOp = o2::framework::DispatchPolicy::DispatchOp;
  // we customize all devices to dispatch data immediately
  auto matcher = [](auto const& spec) {
    return spec.name == "SimReader";
  };
  policies.push_back({"prompt-for-simreader", matcher, DispatchOp::WhenReady});
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
    if (gIsMaster) {
      LOG(FATAL) << "tpc-lanes needs to be positive\n";
    }
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

  auto& cdb = o2::tpc::CDBInterface::instance();
  cdb.setUseDefaults();
  // by invoking this constructor we make sure that a common file will be created
  // in future we should take this from OCDB and just forward per message
  const static auto& ampl = o2::tpc::GEMAmplification::instance();
}

// ------------------------------------------------------------------

std::shared_ptr<o2::parameters::GRPObject> readGRP(std::string inputGRP)
{
  auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  if (!grp) {
    LOG(ERROR) << "This workflow needs a valid GRP file to start";
    return nullptr;
  }
  if (gIsMaster) {
    grp->print();
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

// Finding out if the current process is the master DPL driver process,
// first setting up the topology. Might be important to know when we write
// files (to prevent that multiple processes write the same file)
bool isMasterWorkflowDefinition(ConfigContext const& configcontext)
{
  int argc = configcontext.argc();
  auto argv = configcontext.argv();
  bool ismaster = true;
  for (int argi = 0; argi < argc; ++argi) {
    // when channel-config is present it means that this is started as
    // as FairMQDevice which means it is already a forked process
    if (strcmp(argv[argi], "--channel-config") == 0) {
      ismaster = false;
      break;
    }
  }
  return ismaster;
}

// ------------------------------------------------------------------

/// This function is required to be implemented to define the workflow
/// specifications
WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // check if we merely construct the topology to create help options
  // if this is the case we don't need to read from GRP
  bool helpasked = configcontext.helpOnCommandLine();
  bool ismaster = isMasterWorkflowDefinition(configcontext);
  gIsMaster = ismaster;

  // Reserve one entry which fill be filled with the SimReaderSpec
  // at the end. This places the processor at the beginning of the
  // workflow in the upper left corner of the GUI.
  WorkflowSpec specs(1);

  using namespace o2::conf;
  ConfigurableParam::updateFromFile(configcontext.options().get<std::string>("configFile"));

  // Update the (declared) parameters if changed from the command line
  // Note: In the future this should be done only on a dedicated processor managing
  // the parameters and then propagated automatically to all devices
  ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  // which sim productions to overlay and digitize
  auto simPrefixes = splitString(configcontext.options().get<std::string>("sims"), ',');

  // First, read the GRP to detect which components need instantiations
  auto grpfile = o2::base::NameConf::getGRPFileName(simPrefixes[0]);
  std::shared_ptr<o2::parameters::GRPObject const> grp(nullptr);
  if (!helpasked) {
    grp = readGRP(grpfile.c_str());
    if (!grp) {
      return WorkflowSpec{};
    }
  }

  // update the digitization configuration with the right geometry file
  // we take the geometry from the first simPrefix (could actually check if they are
  // all compatible)
  auto geomfilename = o2::base::NameConf::getGeomFileName(simPrefixes[0]);
  ConfigurableParam::setValue("DigiParams.digitizationgeometry", geomfilename);
  ConfigurableParam::setValue("DigiParams.grpfile", grpfile);
  LOG(INFO) << "MC-TRUTH " << !configcontext.options().get<bool>("disable-mc");
  bool mctruth = !configcontext.options().get<bool>("disable-mc");
  ConfigurableParam::setValue("DigiParams", "mctruth", mctruth);

  // write the configuration used for the digitizer workflow
  if (ismaster) {
    o2::conf::ConfigurableParam::writeINI(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE));
  }

  // onlyDet takes precedence on skipDet
  DetFilterer filterers[2] = {
    whitelister(configcontext.options().get<std::string>("onlyDet"), "none", ','),
    blacklister(configcontext.options().get<std::string>("skipDet"), "none", ',')};

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
  auto isEnabled = [&configcontext, &filterers, accept, grp, helpasked](o2::detectors::DetID id) {
    if (helpasked) {
      return true;
    }
    auto accepted = accept(id);
    bool is_ingrp = grp->isDetReadOut(id);
    if (gIsMaster) {
      LOG(INFO) << id.getName()
                << " is in grp? " << (is_ingrp ? "yes" : "no") << ";"
                << " is skipped? " << (!accepted ? "yes" : "no");
    }
    return accepted && is_ingrp;
  };

  std::vector<o2::detectors::DetID> detList; // list of participating detectors

  // the TPC part
  // we need to init this anyway since TPC is treated a bit special (for the moment)
  if (!helpasked && ismaster) {
    initTPC();
  }

  // keeps track of which tpc sectors to process
  std::vector<int> tpcsectors;

  if (isEnabled(o2::detectors::DetID::TPC)) {
    tpcsectors = o2::RangeTokenizer::tokenize<int>(configcontext.options().get<std::string>("tpc-sectors"));
    // only one lane for the help printout
    auto lanes = helpasked ? 1 : getNumTPCLanes(tpcsectors, configcontext);
    detList.emplace_back(o2::detectors::DetID::TPC);

    WorkflowSpec tpcPipelines = o2::tpc::getTPCDigitizerSpec(lanes, tpcsectors, mctruth);
    specs.insert(specs.end(), tpcPipelines.begin(), tpcPipelines.end());

    if (configcontext.options().get<std::string>("tpc-reco-type").empty() == false) {
      throw std::runtime_error("option 'tpc-reco-type' is deprecated, please connect workflows on the command line by pipe");
    }
    // for writing digits to disc
    specs.emplace_back(o2::tpc::getTPCDigitRootWriterSpec(tpcsectors, mctruth));
  }

  // first 36 channels are reserved for the TPC
  const int firstOtherChannel = 36;
  int fanoutsize = firstOtherChannel;

  // the ITS part
  if (isEnabled(o2::detectors::DetID::ITS)) {
    detList.emplace_back(o2::detectors::DetID::ITS);
    // connect the ITS digitization
    specs.emplace_back(o2::itsmft::getITSDigitizerSpec(fanoutsize++, mctruth));
    // connect ITS digit writer
    specs.emplace_back(o2::itsmft::getITSDigitWriterSpec(mctruth));
  }

  // the MFT part
  if (isEnabled(o2::detectors::DetID::MFT)) {
    detList.emplace_back(o2::detectors::DetID::MFT);
    // connect the MFT digitization
    specs.emplace_back(o2::itsmft::getMFTDigitizerSpec(fanoutsize++, mctruth));
    // connect MFT digit writer
    specs.emplace_back(o2::itsmft::getMFTDigitWriterSpec(mctruth));
  }

  // the TOF part
  if (isEnabled(o2::detectors::DetID::TOF)) {
    auto useCCDB = configcontext.options().get<bool>("use-ccdb-tof");
    detList.emplace_back(o2::detectors::DetID::TOF);
    // connect the TOF digitization
    specs.emplace_back(o2::tof::getTOFDigitizerSpec(fanoutsize++, useCCDB, mctruth));
    // add TOF digit writer
    specs.emplace_back(o2::tof::getTOFDigitWriterSpec(mctruth));
  }

  // the FT0 part
  if (isEnabled(o2::detectors::DetID::FT0)) {
    detList.emplace_back(o2::detectors::DetID::FT0);
    // connect the FIT digitization
    specs.emplace_back(o2::ft0::getFT0DigitizerSpec(fanoutsize++, mctruth));
    // connect the FIT digit writer
    specs.emplace_back(o2::ft0::getFT0DigitWriterSpec(mctruth));
  }

  // the FV0 part
  if (isEnabled(o2::detectors::DetID::FV0)) {
    detList.emplace_back(o2::detectors::DetID::FV0);
    // connect the FV0 digitization
    specs.emplace_back(o2::fv0::getFV0DigitizerSpec(fanoutsize++, mctruth));
    // connect the FV0 digit writer
    specs.emplace_back(o2::fv0::getFV0DigitWriterSpec(mctruth));
  }

  // the EMCal part
  if (isEnabled(o2::detectors::DetID::EMC)) {
    detList.emplace_back(o2::detectors::DetID::EMC);
    // connect the EMCal digitization
    specs.emplace_back(o2::emcal::getEMCALDigitizerSpec(fanoutsize++, mctruth));
    // connect the EMCal digit writer
    specs.emplace_back(o2::emcal::getEMCALDigitWriterSpec(mctruth));
  }

  // add HMPID
  if (isEnabled(o2::detectors::DetID::HMP)) {
    detList.emplace_back(o2::detectors::DetID::HMP);
    // connect the HMP digitization
    specs.emplace_back(o2::hmpid::getHMPIDDigitizerSpec(fanoutsize++, mctruth));
    // connect the HMP digit writer
    specs.emplace_back(o2::hmpid::getHMPIDDigitWriterSpec(mctruth));
  }

  // add ZDC
  if (isEnabled(o2::detectors::DetID::ZDC)) {
    detList.emplace_back(o2::detectors::DetID::ZDC);
    // connect the ZDC digitization
    specs.emplace_back(o2::zdc::getZDCDigitizerSpec(fanoutsize++, mctruth));
    // connect the ZDC digit writer
    specs.emplace_back(o2::zdc::getZDCDigitWriterSpec(mctruth));
  }

  // add TRD
  if (isEnabled(o2::detectors::DetID::TRD)) {
    detList.emplace_back(o2::detectors::DetID::TRD);
    // connect the TRD digitization
    specs.emplace_back(o2::trd::getTRDDigitizerSpec(fanoutsize++, mctruth));
    // connect the TRD digit writer
    specs.emplace_back(o2::trd::getTRDDigitWriterSpec(mctruth));
    auto enableTrapSim = configcontext.options().get<bool>("enable-trd-trapsim");
    if (enableTrapSim) {
      // connect the TRD Trap SimulatorA
      specs.emplace_back(o2::trd::getTRDTrapSimulatorSpec());
      // connect to the device to write out the tracklets.
      specs.emplace_back(o2::trd::getTRDTrackletWriterSpec());
    }
  }

  //add MUON MCH
  if (isEnabled(o2::detectors::DetID::MCH)) {
    detList.emplace_back(o2::detectors::DetID::MCH);
    //connect the MUON MCH digitization
    specs.emplace_back(o2::mch::getMCHDigitizerSpec(fanoutsize++, mctruth));
    //connect the MUON MCH digit writer
    specs.emplace_back(o2::mch::getMCHDigitWriterSpec(mctruth));
  }

  // add MID
  if (isEnabled(o2::detectors::DetID::MID)) {
    detList.emplace_back(o2::detectors::DetID::MID);
    // connect the MID digitization
    specs.emplace_back(o2::mid::getMIDDigitizerSpec(fanoutsize++, mctruth));
    // connect the MID digit writer
    specs.emplace_back(o2::mid::getMIDDigitWriterSpec(mctruth));
  }

  // add FDD
  if (isEnabled(o2::detectors::DetID::FDD)) {
    detList.emplace_back(o2::detectors::DetID::FDD);
    // connect the FDD digitization
    specs.emplace_back(o2::fdd::getFDDDigitizerSpec(fanoutsize++, mctruth));
    // connect the FDD digit writer
    specs.emplace_back(o2::fdd::getFDDDigitWriterSpec(mctruth));
  }

  // the PHOS part
  if (isEnabled(o2::detectors::DetID::PHS)) {
    detList.emplace_back(o2::detectors::DetID::PHS);
    // connect the PHOS digitization
    specs.emplace_back(o2::phos::getPHOSDigitizerSpec(fanoutsize++, mctruth));
    // add PHOS writer
    specs.emplace_back(o2::phos::getPHOSDigitWriterSpec(mctruth));
  }

  // the CPV part
  if (isEnabled(o2::detectors::DetID::CPV)) {
    detList.emplace_back(o2::detectors::DetID::CPV);
    // connect the CPV digitization
    specs.emplace_back(o2::cpv::getCPVDigitizerSpec(fanoutsize++, mctruth));
    // add PHOS writer
    specs.emplace_back(o2::cpv::getCPVDigitWriterSpec(mctruth));
  }

  // GRP updater: must come after all detectors since requires their list
  specs.emplace_back(o2::parameters::getGRPUpdaterSpec(grpfile, detList));

  // The SIM Reader. NEEDS TO BE LAST
  specs[0] = o2::steer::getSimReaderSpec({firstOtherChannel, fanoutsize}, simPrefixes, tpcsectors);

  return specs;
}
