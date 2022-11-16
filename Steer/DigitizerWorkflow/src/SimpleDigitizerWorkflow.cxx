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
#include <boost/lexical_cast.hpp>

#include "Framework/RootSerializationSupport.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/InputSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "SimReaderSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CCDB/BasicCCDBManager.h"

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

// #ifdef ENABLE_UPGRADES
// // for ITS3
// #include "ITS3DigitizerSpec.h"
// #include "ITS3Workflow/DigitWriterSpec.h"
// #endif

// for TOF
#include "TOFDigitizerSpec.h"
#include "TOFWorkflowIO/TOFDigitWriterSpec.h"

// for FT0
#include "FT0DigitizerSpec.h"
#include "FT0DigitWriterSpec.h"

// for CTP
#include "CTPDigitizerSpec.h"
#include "CTPWorkflowIO/DigitWriterSpec.h"

// for FV0
#include "FV0DigitizerSpec.h"
#include "FV0DigitWriterSpec.h"

// for FDD
#include "FDDDigitizerSpec.h"
#include "FDDWorkflow/DigitWriterSpec.h"

// for EMCal
#include "EMCALWorkflow/EMCALDigitizerSpec.h"
#include "EMCALWorkflow/EMCALDigitWriterSpec.h"

// for HMPID
#include "HMPIDDigitizerSpec.h"
#include "HMPIDDigitWriterSpec.h"

// for TRD
#include "TRDWorkflow/TRDDigitizerSpec.h"
#include "TRDWorkflowIO/TRDDigitWriterSpec.h"
#include "TRDWorkflow/TRDTrapSimulatorSpec.h"
#include "TRDWorkflowIO/TRDTrackletWriterSpec.h"

// for MUON MCH
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
#include "ZDCWorkflow/ZDCDigitWriterDPLSpec.h"

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
#include <type_traits>
#include "DetectorsBase/DPLWorkflowUtils.h"
#include "Framework/CCDBParamSpec.h"

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

  std::string onlyctxhelp("Produce only the digitization context; Don't actually digitize");
  workflowOptions.push_back(ConfigParamSpec{"only-context", o2::framework::VariantType::Bool, false, {onlyctxhelp}});

  // we support only output type 'tracks' for the moment
  std::string tpcrthelp("deprecated option, please connect workflows on the command line by pipe");
  workflowOptions.push_back(
    ConfigParamSpec{"tpc-reco-type", VariantType::String, "", {tpcrthelp}});

  // Option to write TPC digits internaly, without forwarding to a special writer instance.
  // This is useful in GRID productions with small available memory.
  workflowOptions.push_back(ConfigParamSpec{"tpc-chunked-writer", o2::framework::VariantType::Bool, false, {"Write independent TPC digit chunks as soon as they can be flushed."}});

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

  // option to disable INI file writing
  workflowOptions.push_back(ConfigParamSpec{"disable-write-ini", o2::framework::VariantType::Bool, false, {"disable  INI config write"}});

  // option to use/not use CCDB for TOF
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb-tof", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"ccdb-tof-sa", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects via CCDBManager (standalone)"}});

  // option to use/not use CCDB for FT0
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb-ft0", o2::framework::VariantType::Bool, true, {"enable access to ccdb ft0 calibration objects"}});

  // option to use/not use CCDB for EMCAL
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb-emc", o2::framework::VariantType::Bool, false, {"enable access to ccdb EMCAL simulation objects"}});

  // option to use or not use the Trap Simulator after digitisation (debate of digitization or reconstruction is for others)
  workflowOptions.push_back(ConfigParamSpec{"disable-trd-trapsim", VariantType::Bool, false, {"disable the trap simulation of the TRD"}});
  workflowOptions.push_back(ConfigParamSpec{"trd-digit-downscaling", VariantType::Int, 1, {"only keep TRD digits for every n-th trigger"}});

  workflowOptions.push_back(ConfigParamSpec{"combine-devices", VariantType::Bool, false, {"combined multiple DPL worker/writer devices"}});
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

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  // we customize the time information sent in DPL headers
  policies.push_back(o2::framework::CallbacksPolicy{
    [](o2::framework::DeviceSpec const& spec, o2::framework::ConfigContext const& context) -> bool {
      return true;
    },
    [](o2::framework::CallbackService& service, o2::framework::InitContext& context) {
      // simple linear enumeration from already updated HBFUtils (set via config key values)
      service.set(o2::framework::CallbackService::Id::NewTimeslice,
                  [](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph) {
                    const auto& hbfu = o2::raw::HBFUtils::Instance();
                    const auto offset = int64_t(hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).orbit);
                    const auto increment = int64_t(hbfu.nHBFPerTF);
                    const auto startTime = hbfu.startTime;
                    const auto orbitFirst = hbfu.orbitFirst;
                    dh.firstTForbit = offset + increment * dh.tfCounter;
                    LOG(info) << "Setting firstTForbit to " << dh.firstTForbit;
                    dh.runNumber = hbfu.runNumber;
                    LOG(info) << "Setting runNumber to " << dh.runNumber;
                    dph.creation = startTime + (dh.firstTForbit - orbitFirst) * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
                    LOG(info) << "Setting timeframe creation time to " << dph.creation;
                  });
    }} // end of struct
  );
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
      LOG(fatal) << "tpc-lanes needs to be positive\n";
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
    LOG(debug) << "GEM ALREADY INITIALIZED ... SKIPPING HERE";
    return;
  }

  LOG(debug) << "INITIALIZING TPC GEMAmplification";
  setenv(streamthis.str().c_str(), "ON", 1);

  auto& cdb = o2::tpc::CDBInterface::instance();
  cdb.setUseDefaults();
  // by invoking this constructor we make sure that a common file will be created
  // in future we should take this from OCDB and just forward per message
  const static auto& ampl = o2::tpc::GEMAmplification::instance();
}

// ------------------------------------------------------------------
void publish_master_env(const char* key, const char* value)
{
  // publish env variables as process master
  std::stringstream str;
  str << "O2SIMDIGIINTERNAL_" << getpid() << "_" << key;
  LOG(info) << "Publishing master key " << str.str();
  setenv(str.str().c_str(), value, 1);
}

const char* get_master_env(const char* key)
{
  // access internal env variables published by master process
  std::stringstream str;
  str << "O2SIMDIGIINTERNAL_" << getppid() << "_" << key;
  // LOG(info) << "Looking up master key " << str.str();
  return getenv(str.str().c_str());
}

std::shared_ptr<o2::parameters::GRPObject> readGRP(std::string const& inputGRP)
{
  auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  if (!grp) {
    LOG(error) << "This workflow needs a valid GRP file to start";
    return nullptr;
  }
  if (gIsMaster) {
    grp->print();
  }
  return std::shared_ptr<o2::parameters::GRPObject>(grp);
}

// ------------------------------------------------------------------

// Split a given string on a separator character
std::vector<std::string> splitString(std::string const& src, char sep)
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
  DetFilterer(std::string const& detlist, std::string const& unsetVal, char separator, bool doWhiteListing)
  {
    // option is not set, nothing to do
    if (detlist.compare(unsetVal) == 0) {
      return;
    }

    std::vector<std::string> tokens = splitString(detlist, separator);

    // Convert a vector of strings to one of o2::detectors::DetID
    for (auto& token : tokens) {
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
  // check if we merely construct the topology to create help options
  // if this is the case we don't need to read from GRP
  bool helpasked = configcontext.helpOnCommandLine();
  bool ismaster = isMasterWorkflowDefinition(configcontext);
  gIsMaster = ismaster;

  std::string dplProcessName = whoAmI(configcontext);
  bool isDPLinternal = isInternalDPL(dplProcessName);
  bool isDumpWorkflow = isDumpWorkflowInvocation(configcontext);
  bool initServices = !isDPLinternal && !isDumpWorkflow && !ismaster;
  // Reserve one entry which will be filled with the SimReaderSpec
  // at the end. This places the processor at the beginning of the
  // workflow in the upper left corner of the GUI.
  WorkflowSpec specs(1);
  WorkflowSpec digitizerSpecs; // collecting everything producing digits
  WorkflowSpec writerSpecs;    // collecting everything writing digits to files

  using namespace o2::conf;
  ConfigurableParam::updateFromFile(configcontext.options().get<std::string>("configFile"));

  // Update the (declared) parameters if changed from the command line
  // Note: In the future this should be done only on a dedicated processor managing
  // the parameters and then propagated automatically to all devices
  ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  const auto& hbfu = o2::raw::HBFUtils::Instance();

  // which sim productions to overlay and digitize
  auto simPrefixes = splitString(configcontext.options().get<std::string>("sims"), ',');
  // First, read the GRP to detect which components need instantiations
  std::shared_ptr<o2::parameters::GRPObject const> grp(nullptr);

  // lambda to access the GRP time start
  auto getGRPStartTime = [](o2::parameters::GRPObject const* grp) {
    const auto GRPTIMEKEY = "GRPTIMESTART";
    if (gIsMaster && grp) {
      // we publish a couple of things as environment variables
      // this saves loading from ROOT file and hence duplicated file reading and
      // initialization of the ROOT engine in each DPL device
      auto t = grp->getTimeStart();
      publish_master_env(GRPTIMEKEY, std::to_string(t).c_str());
      return t;
    } else {
      auto tstr = get_master_env(GRPTIMEKEY);
      if (!tstr) {
        LOG(fatal) << "Expected env value not found";
      }
      // LOG(info) << "Found entry " << tstr;
      return boost::lexical_cast<uint64_t>(tstr);
    }
  };

  if (!helpasked) {
    if (gIsMaster) {
      grp = readGRP(simPrefixes[0]);
      if (!grp) {
        return WorkflowSpec{};
      }
      getGRPStartTime(grp.get());
    }
    if (!hbfu.startTime) { // HBFUtils.startTime was not set from the command line, set it from GRP
      hbfu.setValue("HBFUtils.startTime", std::to_string(getGRPStartTime(grp.get())));
    }
  }

  auto grpfile = o2::base::NameConf::getGRPFileName(simPrefixes[0]);
  if (initServices) {
    // init on a high level, the time for the CCDB queries
    // we expect that digitizers do not play with the manager themselves
    // this will only be needed until digitizers take CCDB objects via DPL mechanism
    o2::ccdb::BasicCCDBManager::instance().setTimestamp(hbfu.startTime);
    // activate caching
    o2::ccdb::BasicCCDBManager::instance().setCaching(true);
    // without this, caching does not seem to work
    o2::ccdb::BasicCCDBManager::instance().setLocalObjectValidityChecking(true);
  }
  // update the digitization configuration with the right geometry file
  // we take the geometry from the first simPrefix (could actually check if they are
  // all compatible)
  ConfigurableParam::setValue("DigiParams.digitizationgeometry_prefix", simPrefixes[0]);
  ConfigurableParam::setValue("DigiParams.grpfile", grpfile);

  LOG(info) << "MC-TRUTH " << !configcontext.options().get<bool>("disable-mc");
  bool mctruth = !configcontext.options().get<bool>("disable-mc");
  ConfigurableParam::setValue("DigiParams", "mctruth", mctruth);

  // write the configuration used for the digitizer workflow
  // (In the case, in which we call multiple processes to do digitization,
  //  only one of them should write this file ... but take the complete configKeyValue line)
  if (ismaster) {
    if (!configcontext.options().get<bool>("disable-write-ini")) {
      o2::conf::ConfigurableParam::writeINI(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE));
    }
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
    auto isInGRPReadout = [grp](o2::detectors::DetID id) {
      std::stringstream str;
      str << "GRPDETKEY_" << id.getName();
      if (gIsMaster and grp.get() != nullptr) {
        auto ok = grp->isDetReadOut(id);
        if (ok) {
          publish_master_env(str.str().c_str(), "ON");
        }
        return ok;
      } else {
        // we should have published important GRP info as
        // environment variables in order to not having to read GRP via ROOT
        // in all the processes
        return get_master_env(str.str().c_str()) != nullptr;
      }
    };

    if (helpasked) {
      return true;
    }
    if (configcontext.options().get<bool>("only-context")) {
      // no detector necessary if we are asked to produce only the digitization context
      return false;
    }
    auto accepted = accept(id);
    bool is_ingrp = isInGRPReadout(id);
    if (gIsMaster) {
      LOG(info) << id.getName()
                << " is in grp? " << (is_ingrp ? "yes" : "no") << ";"
                << " is skipped? " << (!accepted ? "yes" : "no");
    }
    return accepted && is_ingrp;
  };

  std::vector<o2::detectors::DetID> detList; // list of participating detectors

  // keeps track of which tpc sectors to process
  std::vector<int> tpcsectors;

  if (isEnabled(o2::detectors::DetID::TPC)) {
    if (!helpasked && ismaster) {
      initTPC();
    }

    tpcsectors = o2::RangeTokenizer::tokenize<int>(configcontext.options().get<std::string>("tpc-sectors"));
    // only one lane for the help printout
    auto lanes = helpasked ? 1 : getNumTPCLanes(tpcsectors, configcontext);
    detList.emplace_back(o2::detectors::DetID::TPC);

    auto internalwrite = configcontext.options().get<bool>("tpc-chunked-writer");
    WorkflowSpec tpcPipelines = o2::tpc::getTPCDigitizerSpec(lanes, tpcsectors, mctruth, internalwrite);
    specs.insert(specs.end(), tpcPipelines.begin(), tpcPipelines.end());

    if (configcontext.options().get<std::string>("tpc-reco-type").empty() == false) {
      throw std::runtime_error("option 'tpc-reco-type' is deprecated, please connect workflows on the command line by pipe");
    }
    if (!internalwrite) {
      // for writing digits to disc
      specs.emplace_back(o2::tpc::getTPCDigitRootWriterSpec(tpcsectors, mctruth));
    }
  }

  // first 36 channels are reserved for the TPC
  const int firstOtherChannel = 36;
  int fanoutsize = firstOtherChannel;

  // the ITS part
  if (isEnabled(o2::detectors::DetID::ITS)) {
    detList.emplace_back(o2::detectors::DetID::ITS);
    // connect the ITS digitization
    digitizerSpecs.emplace_back(o2::itsmft::getITSDigitizerSpec(fanoutsize++, mctruth));
    // connect ITS digit writer
    writerSpecs.emplace_back(o2::itsmft::getITSDigitWriterSpec(mctruth));
  }

  // #ifdef ENABLE_UPGRADES
  //   // the ITS3 part
  //   if (isEnabled(o2::detectors::DetID::IT3)) {
  //     detList.emplace_back(o2::detectors::DetID::IT3);
  //     // connect the ITS digitization
  //     specs.emplace_back(o2::its3::getITS3DigitizerSpec(fanoutsize++, mctruth));
  //     // // connect ITS digit writer
  //     specs.emplace_back(o2::its3::getITS3DigitWriterSpec(mctruth));
  //   }
  // #endif

  // the MFT part
  if (isEnabled(o2::detectors::DetID::MFT)) {
    detList.emplace_back(o2::detectors::DetID::MFT);
    // connect the MFT digitization
    digitizerSpecs.emplace_back(o2::itsmft::getMFTDigitizerSpec(fanoutsize++, mctruth));
    // connect MFT digit writer
    writerSpecs.emplace_back(o2::itsmft::getMFTDigitWriterSpec(mctruth));
  }

  // the TOF part
  if (isEnabled(o2::detectors::DetID::TOF)) {
    auto useCCDB = configcontext.options().get<bool>("use-ccdb-tof");
    auto CCDBsa = configcontext.options().get<bool>("ccdb-tof-sa");
    auto ccdb_url_tof = o2::base::NameConf::getCCDBServer();
    auto timestamp = o2::raw::HBFUtils::Instance().startTime / 1000;
    detList.emplace_back(o2::detectors::DetID::TOF);
    // connect the TOF digitization
    // printf("TOF Setting: use-ccdb = %d ---- ccdb url=%s  ----   timestamp=%ld\n", useCCDB, ccdb_url_tof.c_str(), timestamp);

    if (CCDBsa) {
      useCCDB = true;
    }
    digitizerSpecs.emplace_back(o2::tof::getTOFDigitizerSpec(fanoutsize++, useCCDB, mctruth, ccdb_url_tof.c_str(), timestamp, CCDBsa));
    // add TOF digit writer
    writerSpecs.emplace_back(o2::tof::getTOFDigitWriterSpec(mctruth));
  }

  // the FT0 part
  if (isEnabled(o2::detectors::DetID::FT0)) {
    auto useCCDB = configcontext.options().get<bool>("use-ccdb-ft0");
    auto timestamp = o2::raw::HBFUtils::Instance().startTime;
    detList.emplace_back(o2::detectors::DetID::FT0);
    // connect the FT0 digitization
    specs.emplace_back(o2::ft0::getFT0DigitizerSpec(fanoutsize++, mctruth, !useCCDB));
    // connect the FIT digit writer
    writerSpecs.emplace_back(o2::ft0::getFT0DigitWriterSpec(mctruth));
  }

  // the FV0 part
  if (isEnabled(o2::detectors::DetID::FV0)) {
    detList.emplace_back(o2::detectors::DetID::FV0);
    // connect the FV0 digitization
    digitizerSpecs.emplace_back(o2::fv0::getFV0DigitizerSpec(fanoutsize++, mctruth));
    // connect the FV0 digit writer
    writerSpecs.emplace_back(o2::fv0::getFV0DigitWriterSpec(mctruth));
  }

  // the EMCal part
  if (isEnabled(o2::detectors::DetID::EMC)) {
    auto useCCDB = configcontext.options().get<bool>("use-ccdb-emc");
    detList.emplace_back(o2::detectors::DetID::EMC);
    // connect the EMCal digitization
    digitizerSpecs.emplace_back(o2::emcal::getEMCALDigitizerSpec(fanoutsize++, mctruth, useCCDB));
    // connect the EMCal digit writer
    writerSpecs.emplace_back(o2::emcal::getEMCALDigitWriterSpec(mctruth));
  }

  // add HMPID
  if (isEnabled(o2::detectors::DetID::HMP)) {
    detList.emplace_back(o2::detectors::DetID::HMP);
    // connect the HMP digitization
    digitizerSpecs.emplace_back(o2::hmpid::getHMPIDDigitizerSpec(fanoutsize++, mctruth));
    // connect the HMP digit writer
    writerSpecs.emplace_back(o2::hmpid::getHMPIDDigitWriterSpec(mctruth));
  }

  // add ZDC
  if (isEnabled(o2::detectors::DetID::ZDC)) {
    detList.emplace_back(o2::detectors::DetID::ZDC);
    // connect the ZDC digitization
    digitizerSpecs.emplace_back(o2::zdc::getZDCDigitizerSpec(fanoutsize++, mctruth));
    // connect the ZDC digit writer
    writerSpecs.emplace_back(o2::zdc::getZDCDigitWriterDPLSpec(mctruth, true));
  }

  // add TRD
  if (isEnabled(o2::detectors::DetID::TRD)) {
    detList.emplace_back(o2::detectors::DetID::TRD);
    // connect the TRD digitization
    specs.emplace_back(o2::trd::getTRDDigitizerSpec(fanoutsize++, mctruth));
    auto disableTrapSim = configcontext.options().get<bool>("disable-trd-trapsim");
    auto trdDigitDownscaling = configcontext.options().get<int>("trd-digit-downscaling");
    if (!disableTrapSim) {
      // connect the TRD TRAP simulator
      specs.emplace_back(o2::trd::getTRDTrapSimulatorSpec(mctruth, trdDigitDownscaling));
      // connect to the device to write out the tracklets.
      specs.emplace_back(o2::trd::getTRDTrackletWriterSpec(mctruth));
      // connect the TRD digit writer expecting input from TRAP simulation
      specs.emplace_back(o2::trd::getTRDDigitWriterSpec(mctruth, false));
    } else {
      // connect the TRD digit writer expecting input from TRD digitizer
      specs.emplace_back(o2::trd::getTRDDigitWriterSpec(mctruth, true));
    }
  }

  // add MUON MCH
  if (isEnabled(o2::detectors::DetID::MCH)) {
    detList.emplace_back(o2::detectors::DetID::MCH);
    // connect the MUON MCH digitization
    digitizerSpecs.emplace_back(o2::mch::getMCHDigitizerSpec(fanoutsize++, mctruth));
    // connect the MUON MCH digit writer
    writerSpecs.emplace_back(o2::mch::getMCHDigitWriterSpec(mctruth));
  }

  // add MID
  if (isEnabled(o2::detectors::DetID::MID)) {
    detList.emplace_back(o2::detectors::DetID::MID);
    // connect the MID digitization
    digitizerSpecs.emplace_back(o2::mid::getMIDDigitizerSpec(fanoutsize++, mctruth));
    // connect the MID digit writer
    writerSpecs.emplace_back(o2::mid::getMIDDigitWriterSpec(mctruth));
  }

  // add FDD
  if (isEnabled(o2::detectors::DetID::FDD)) {
    detList.emplace_back(o2::detectors::DetID::FDD);
    // connect the FDD digitization
    digitizerSpecs.emplace_back(o2::fdd::getFDDDigitizerSpec(fanoutsize++, mctruth));
    // connect the FDD digit writer
    writerSpecs.emplace_back(o2::fdd::getFDDDigitWriterSpec(mctruth));
  }

  // the PHOS part
  if (isEnabled(o2::detectors::DetID::PHS)) {
    detList.emplace_back(o2::detectors::DetID::PHS);
    // connect the PHOS digitization
    digitizerSpecs.emplace_back(o2::phos::getPHOSDigitizerSpec(fanoutsize++, mctruth));
    // add PHOS writer
    writerSpecs.emplace_back(o2::phos::getPHOSDigitWriterSpec(mctruth));
  }

  // the CPV part
  if (isEnabled(o2::detectors::DetID::CPV)) {
    detList.emplace_back(o2::detectors::DetID::CPV);
    // connect the CPV digitization
    digitizerSpecs.emplace_back(o2::cpv::getCPVDigitizerSpec(fanoutsize++, mctruth));
    // add PHOS writer
    writerSpecs.emplace_back(o2::cpv::getCPVDigitWriterSpec(mctruth));
  }
  // the CTP part
  if (isEnabled(o2::detectors::DetID::CTP)) {
    detList.emplace_back(o2::detectors::DetID::CTP);
    // connect the CTP digitization
    specs.emplace_back(o2::ctp::getCTPDigitizerSpec(fanoutsize++, detList));
    // connect the CTP digit writer
    specs.emplace_back(o2::ctp::getDigitWriterSpec(false));
  }
  // GRP updater: must come after all detectors since requires their list
  if (!configcontext.options().get<bool>("only-context")) {
    writerSpecs.emplace_back(o2::parameters::getGRPUpdaterSpec(simPrefixes[0], detList));
  }

  bool combine = configcontext.options().get<bool>("combine-devices");
  if (!combine) {
    for (auto& s : digitizerSpecs) {
      specs.push_back(s);
    }
    for (auto& s : writerSpecs) {
      specs.push_back(s);
    }
  } else {
    std::vector<DataProcessorSpec> remaining;
    specs.push_back(specCombiner("Digitizations", digitizerSpecs, remaining));
    specs.push_back(specCombiner("Writers", writerSpecs, remaining));
    for (auto& s : remaining) {
      specs.push_back(s);
    }
  }

  // For reasons of offering homegenous behaviour (consistent options to outside scripts),
  // we require that at least one of the devices above listens to the DPL CCDB fetcher.
  // Verify this or insert a dummy channel in one of the devices. (This cannot be done in the SimReader
  // as the SimReader is the source device injecting the timing information).
  // In future this code can serve as a check that all digitizers access CCDB via the DPL fetcher.
  bool haveCCDBInputSpec = false;
  for (auto spec : specs) {
    for (auto in : spec.inputs) {
      if (in.lifetime == Lifetime::Condition) {
        haveCCDBInputSpec = true;
        break;
      }
    }
  }
  if (!haveCCDBInputSpec && specs.size() > 0) {
    LOG(info) << "No one uses DPL CCDB .. injecting a dummy CCDB query into " << specs.back().name;
    specs.back().inputs.emplace_back("_dummyOrbitReset", "CTP", "ORBITRESET", 0, Lifetime::Condition,
                                     ccdbParamSpec("CTP/Calib/OrbitReset"));
  }

  // The SIM Reader. NEEDS TO BE LAST
  specs[0] = o2::steer::getSimReaderSpec({firstOtherChannel, fanoutsize}, simPrefixes, tpcsectors);
  return specs;
}
