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

/// \file
/// \author Julian Myrcha

#include "EveWorkflow/O2DPLDisplay.h"
#include "EveWorkflow/EveWorkflowHelper.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "CommonUtils/NameConf.h"
#include "TRDBase/GeometryFlat.h"
#include "TOFBase/Geo.h"
#include "TPCFastTransform.h"
#include "TRDBase/Geometry.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Cluster.h"
#include <unistd.h>
#include <climits>

using namespace o2::event_visualisation;
using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::globaltracking;
using namespace o2::tpc;
using namespace o2::trd;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"jsons-folder", VariantType::String, "jsons", {"name of the folder to store json files"}},
    {"eve-hostname", VariantType::String, "", {"name of the host allowed to produce files (empty means no limit)"}},
    {"eve-dds-collection-index", VariantType::Int, -1, {"number of dpl collection allowed to produce files (-1 means no limit)"}},
    {"number-of_files", VariantType::Int, 300, {"maximum number of json files in folder"}},
    {"number-of_tracks", VariantType::Int, -1, {"maximum number of track stored in json file (-1 means no limit)"}},
    {"time-interval", VariantType::Int, 5000, {"time interval in milliseconds between stored files"}},
    {"disable-mc", VariantType::Bool, false, {"disable visualization of MC data"}},
    {"display-clusters", VariantType::String, "ITS,TPC,TRD,TOF", {"comma-separated list of clusters to display"}},
    {"display-tracks", VariantType::String, "TPC,ITS,ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF", {"comma-separated list of tracks to display"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}},
    {"skipOnEmptyInput", VariantType::Bool, false, {"Just don't run the ED when no input is provided"}},
    {"filter-its-rof", VariantType::Bool, false, {"don't display tracks outside ITS readout frame"}},
    {"no-empty-output", VariantType::Bool, false, {"don't create files with no tracks/clusters"}},
    {"filter-time-min", VariantType::Float, -1, {"display tracks only in [min, max] microseconds time range in each time frame, requires --filter-time-max to be specified as well"}},
    {"filter-time-max", VariantType::Float, -1, {"display tracks only in [min, max] microseconds time range in each time frame, requires --filter-time-min to be specified as well"}},
  };

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // main method must be included here (otherwise customize not used)
void O2DPLDisplaySpec::init(InitContext& ic)
{
  LOG(info) << "------------------------    O2DPLDisplay::init version " << this->mWorkflowVersion << "    ------------------------------------";
  mData.init();

  mData.mConfig->configProcessing.runMC = mUseMC;
}

void O2DPLDisplaySpec::run(ProcessingContext& pc)
{
  if (!this->mEveHostNameMatch) {
    return;
  }
  LOG(info) << "------------------------    O2DPLDisplay::run version " << this->mWorkflowVersion << "    ------------------------------------";
  // filtering out any run which occur before reaching next time interval
  auto currentTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = currentTime - this->mTimeStamp;
  if (elapsed < this->mTimeInteval) {
    return; // skip this run - it is too often
  }
  this->mTimeStamp = currentTime;
  updateTimeDependentParams(pc);

  EveWorkflowHelper::FilterSet enabledFilters;

  if (this->mFilterITSROF) {
    enabledFilters.set(EveWorkflowHelper::Filter::ITSROF);
  }

  if (this->mFilterTime) {
    enabledFilters.set(EveWorkflowHelper::Filter::TimeBracket);
  }

  if (this->mNumberOfTracks != -1) {
    enabledFilters.set(EveWorkflowHelper::Filter::TotalNTracks);
  }

  EveWorkflowHelper helper(enabledFilters, this->mNumberOfTracks, this->mTimeBracket);

  helper.getRecoContainer().collectData(pc, *mDataRequest);
  helper.selectTracks(&(mData.mConfig->configCalib), mClMask, mTrkMask, mTrkMask);

  helper.prepareITSClusters(mData.mITSDict);
  helper.prepareMFTClusters(mData.mMFTDict);

  const auto& ref = pc.inputs().getFirstValid(true);
  const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
  const auto* dph = DataRefUtils::getHeader<DataProcessingHeader*>(ref);

  helper.draw();

  if (!(this->mNoEmptyOutput && helper.isEmpty())) {
    helper.save(this->mJsonPath, this->mNumberOfFiles, this->mTrkMask, this->mClMask, this->mWorkflowVersion, dh->runNumber, dh->firstTForbit, dph->creation);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  LOGP(info, "Visualization of TF:{} at orbit {} took {} s.", dh->tfCounter, dh->firstTForbit, std::chrono::duration_cast<std::chrono::microseconds>(endTime - currentTime).count() * 1e-6);
}

void O2DPLDisplaySpec::endOfStream(EndOfStreamContext& ec)
{
}

void O2DPLDisplaySpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldictITS"); // called by the RecoContainer
  // pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldictMFT"); // called by the RecoContainer
}

void O2DPLDisplaySpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "ITS cluster dictionary updated";
    mData.setITSDict((const o2::itsmft::TopologyDictionary*)obj);
  }
  if (matcher == ConcreteDataMatcher("MFT", "CLUSDICT", 0)) {
    LOG(info) << "MFT cluster dictionary updated";
    mData.setMFTDict((const o2::itsmft::TopologyDictionary*)obj);
  }
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  LOG(info) << "------------------------    defineDataProcessing " << O2DPLDisplaySpec::mWorkflowVersion << "    ------------------------------------";

  WorkflowSpec specs;

  std::string jsonFolder = cfgc.options().get<std::string>("jsons-folder");
  std::string eveHostName = cfgc.options().get<std::string>("eve-hostname");
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  bool useMC = !cfgc.options().get<bool>("disable-mc");

  char hostname[_POSIX_HOST_NAME_MAX];
  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  bool eveHostNameMatch = eveHostName.empty() || eveHostName == hostname;

  int eveDDSColIdx = cfgc.options().get<int>("eve-dds-collection-index");
  if (eveDDSColIdx != -1) {
    char* colIdx = getenv("DDS_COLLECTION_INDEX");
    int myIdx = colIdx ? atoi(colIdx) : -1;
    if (myIdx == eveDDSColIdx) {
      LOG(important) << "Restricting DPL Display to collection index, my index " << myIdx << ", enabled " << int(myIdx == eveDDSColIdx);
    } else {
      LOG(info) << "Restricting DPL Display to collection index, my index " << myIdx << ", enabled " << int(myIdx == eveDDSColIdx);
    }
    eveHostNameMatch &= myIdx == eveDDSColIdx;
  }

  std::chrono::milliseconds timeInterval(cfgc.options().get<int>("time-interval"));
  int numberOfFiles = cfgc.options().get<int>("number-of_files");
  int numberOfTracks = cfgc.options().get<int>("number-of_tracks");

  GID::mask_t allowedTracks = GID::getSourcesMask("ITS,TPC,MFT,MCH,ITS-TPC,ITS-TPC-TOF,TPC-TRD,ITS-TPC-TRD,MID");
  GID::mask_t allowedClusters = GID::getSourcesMask("ITS,TPC,MFT,MCH,TRD,TOF,MID,TRD");

  GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-tracks")) & allowedTracks;
  GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-clusters")) & allowedClusters;
  if (!srcTrk.any() && !srcCl.any()) {
    if (cfgc.options().get<bool>("skipOnEmptyInput")) {
      LOG(info) << "No valid inputs for event display, disabling event display";
      return std::move(specs);
    }
    throw std::runtime_error("No input configured");
  }

  bool filterTime;
  EveWorkflowHelper::TBracket timeBracket;

  if (cfgc.options().isDefault("filter-time-min") && cfgc.options().isDefault("filter-time-max")) {
    filterTime = false;
  } else if (!cfgc.options().isDefault("filter-time-min") && !cfgc.options().isDefault("filter-time-max")) {
    filterTime = true;

    auto filterTimeMin = cfgc.options().get<float>("filter-time-min");
    auto filterTimeMax = cfgc.options().get<float>("filter-time-max");

    timeBracket = EveWorkflowHelper::TBracket{filterTimeMin, filterTimeMax};

    if (timeBracket.isInvalid()) {
      throw std::runtime_error("Filter time bracket is invalid");
    }
  } else {
    throw std::runtime_error("Both filter times, min and max, have to be specified at the same time");
  }

  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTrk, useMC);
  dataRequest->requestClusters(srcCl, useMC);

  auto filterITSROF = cfgc.options().get<bool>("filter-its-rof");

  if (filterITSROF) {
    dataRequest->requestIRFramesITS();
  }

  InputHelper::addInputSpecs(cfgc, specs, srcCl, srcTrk, srcTrk, useMC);

  auto noEmptyFiles = cfgc.options().get<bool>("no-empty-output");

  specs.emplace_back(DataProcessorSpec{
    "o2-eve-display",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<O2DPLDisplaySpec>(useMC, srcTrk, srcCl, dataRequest, jsonFolder, timeInterval, numberOfFiles, numberOfTracks, eveHostNameMatch, noEmptyFiles, filterITSROF, filterTime, timeBracket)}});

  return std::move(specs);
}
