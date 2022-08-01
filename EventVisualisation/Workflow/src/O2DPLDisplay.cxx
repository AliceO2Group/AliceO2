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

#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "EveWorkflow/O2DPLDisplay.h"
#include "EveWorkflow/EveWorkflowHelper.h"
#include "EventVisualisationBase/ConfigurationManager.h"
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

using namespace o2::event_visualisation;
using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::globaltracking;
using namespace o2::tpc;
using namespace o2::trd;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"jsons-folder", VariantType::String, "jsons", {"name of the folder to store json files"}},
    {"use-json-format", VariantType::Bool, false, {"instead of root format (default) use json format"}},
    {"eve-hostname", VariantType::String, "", {"name of the host allowed to produce files (empty means no limit)"}},
    {"eve-dds-collection-index", VariantType::Int, -1, {"number of dpl collection allowed to produce files (-1 means no limit)"}},
    {"number-of_files", VariantType::Int, 150, {"maximum number of json files in folder"}},
    {"number-of_tracks", VariantType::Int, -1, {"maximum number of track stored in json file (-1 means no limit)"}},
    {"time-interval", VariantType::Int, 5000, {"time interval in milliseconds between stored files"}},
    {"disable-mc", VariantType::Bool, false, {"disable visualization of MC data"}},
    {"display-clusters", VariantType::String, "ITS,TPC,TRD,TOF", {"comma-separated list of clusters to display"}},
    {"display-tracks", VariantType::String, "TPC,ITS,ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF", {"comma-separated list of tracks to display"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
    {"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings ..."}},
    {"skipOnEmptyInput", VariantType::Bool, false, {"don't run the ED when no input is provided"}},
    {"min-its-tracks", VariantType::Int, -1, {"don't create file if less than the specified number of ITS tracks is present"}},
    {"min-tracks", VariantType::Int, 1, {"don't create file if less than the specified number of all tracks is present"}},
    {"filter-its-rof", VariantType::Bool, false, {"don't display tracks outside ITS readout frame"}},
    {"filter-time-min", VariantType::Float, -1.f, {"display tracks only in [min, max] microseconds time range in each time frame, requires --filter-time-max to be specified as well"}},
    {"filter-time-max", VariantType::Float, -1.f, {"display tracks only in [min, max] microseconds time range in each time frame, requires --filter-time-min to be specified as well"}},
    {"remove-tpc-abs-eta", VariantType::Float, 0.f, {"remove TPC tracks in [-eta, +eta] range"}},
    {"track-sorting", VariantType::Bool, true, {"sort track by track time before applying filters"}},
    {"only-nth-event", VariantType::Int, 0, {"process only every nth event"}},
    {"primary-vertex-mode", VariantType::Bool, false, {"produce jsons with individual primary vertices, not total time frame data"}},
    {"max-primary-vertices", VariantType::Int, 5, {"maximum number of primary vertices to draw per time frame"}},
    {"primary-vertex-triggers", VariantType::Bool, false, {"instead of drawing vertices with tracks (and maybe calorimeter triggers), draw vertices with calorimeter triggers (and maybe tracks)"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // main method must be included here (otherwise customize not used)
void O2DPLDisplaySpec::init(InitContext& ic)
{
  LOG(info) << "------------------------    O2DPLDisplay::init version " << o2_eve_version << "    ------------------------------------";
  mData.mConfig.configProcessing.runMC = mUseMC;
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
}

void O2DPLDisplaySpec::run(ProcessingContext& pc)
{
  if (!this->mEveHostNameMatch) {
    return;
  }
  if (this->mOnlyNthEvent && this->mEventCounter++ % this->mOnlyNthEvent != 0) {
    return;
  }
  LOG(info) << "------------------------    O2DPLDisplay::run version " << o2_eve_version << "    ------------------------------------";
  // filtering out any run which occur before reaching next time interval
  auto currentTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = currentTime - this->mTimeStamp;
  if (elapsed < this->mTimeInterval) {
    return; // skip this run - it is too often
  }
  this->mTimeStamp = currentTime;
  o2::globaltracking::RecoContainer recoCont;
  recoCont.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc); // Make sure that this is called after the RecoContainer collect data, since some condition objects are fetched there

  EveWorkflowHelper::FilterSet enabledFilters;

  enabledFilters.set(EveWorkflowHelper::Filter::ITSROF, this->mFilterITSROF);
  enabledFilters.set(EveWorkflowHelper::Filter::TimeBracket, this->mFilterTime);
  enabledFilters.set(EveWorkflowHelper::Filter::EtaBracket, this->mRemoveTPCEta);
  enabledFilters.set(EveWorkflowHelper::Filter::TotalNTracks, this->mNumberOfTracks != -1);

  EveWorkflowHelper helper(enabledFilters, this->mNumberOfTracks, this->mTimeBracket, this->mEtaBracket, this->mPrimaryVertexMode);
  helper.setRecoContainer(&recoCont);

  helper.setITSROFs();
  helper.selectTracks(&(mData.mConfig.configCalib), mClMask, mTrkMask, mTrkMask);
  helper.selectTowers();

  helper.prepareITSClusters(mData.mITSDict);
  helper.prepareMFTClusters(mData.mMFTDict);

  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();

  std::size_t filesSaved = 0;

  auto processData = [&](const auto& dataMap) {
    for (const auto& keyVal : dataMap) {
      if (filesSaved >= mMaxPrimaryVertices) {
        break;
      }

      const auto pv = keyVal.first;
      helper.draw(pv, mTrackSorting);

      bool save = true;

      if (this->mMinITSTracks != -1 && helper.mEvent.getDetectorTrackCount(detectors::DetID::ITS) < this->mMinITSTracks) {
        save = false;
      }

      if (this->mMinTracks != -1 && helper.mEvent.getTrackCount() < this->mMinTracks) {
        save = false;
      }

      if (save) {
        helper.mEvent.setClMask(this->mClMask.to_ulong());
        helper.mEvent.setTrkMask(this->mTrkMask.to_ulong());
        helper.mEvent.setRunNumber(tinfo.runNumber);
        helper.mEvent.setTfCounter(tinfo.tfCounter);
        helper.mEvent.setFirstTForbit(tinfo.firstTForbit);
        helper.mEvent.setPrimaryVertex(pv);
        helper.save(this->mJsonPath, this->mExt, this->mNumberOfFiles, this->mTrkMask, this->mClMask, tinfo.runNumber, tinfo.creation);
        filesSaved++;
      }

      helper.clear();
    }
  };

  if (mPrimaryVertexTriggers) {
    processData(helper.mPrimaryVertexTriggerGIDs);
  } else {
    processData(helper.mPrimaryVertexTrackGIDs);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  LOGP(info, "Visualization of TF:{} at orbit {} took {} s.", tinfo.tfCounter, tinfo.firstTForbit, std::chrono::duration_cast<std::chrono::microseconds>(endTime - currentTime).count() * 1e-6);

  LOGP(info, "PVs with tracks: {}", helper.mPrimaryVertexTrackGIDs.size());
  LOGP(info, "PVs with triggers: {}", helper.mPrimaryVertexTriggerGIDs.size());
  LOGP(info, "Data files saved: {}", filesSaved);

  std::unordered_map<o2::dataformats::GlobalTrackID, std::size_t> savedDataTypes;

  for (int i = 0; i < GID::Source::NSources; i++) {
    savedDataTypes[i] = 0;
  }

  for (const auto& gid : helper.mTotalAcceptedDataTypes) {
    savedDataTypes[gid.getSource()] += 1;
  }

  std::vector<std::string> sourceStats;
  sourceStats.reserve(GID::Source::NSources);

  const auto combinedMask = mTrkMask | mClMask;

  for (int i = 0; i < GID::Source::NSources; i++) {
    if (combinedMask[i]) {
      sourceStats.emplace_back(fmt::format("{}/{} {}", savedDataTypes.at(i), helper.mTotalDataTypes.at(i), GID::getSourceName(i)));
    }
  }

  LOGP(info, "Tracks: {}", fmt::join(sourceStats, ", "));
}

void O2DPLDisplaySpec::endOfStream(EndOfStreamContext& ec)
{
}

void O2DPLDisplaySpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    auto grpECS = o2::base::GRPGeomHelper::instance().getGRPECS(); // RS
    mData.init();
  }
  // pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldictITS"); // called by the RecoContainer
  // pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldictMFT"); // called by the RecoContainer
}

void O2DPLDisplaySpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "ITS cluster dictionary updated";
    mData.setITSDict((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "CLUSDICT", 0)) {
    LOG(info) << "MFT cluster dictionary updated";
    mData.setMFTDict((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  LOG(info) << "------------------------    defineDataProcessing " << o2_eve_version << "    ------------------------------------";

  WorkflowSpec specs;

  std::string jsonFolder = cfgc.options().get<std::string>("jsons-folder");
  std::string ext = ".root"; // root files are default format
  auto useJsonFormat = cfgc.options().get<bool>("use-json-format");
  if (useJsonFormat) {
    ext = ".json";
  }
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

  GID::mask_t allowedTracks = GID::getSourcesMask(O2DPLDisplaySpec::allowedTracks);
  GID::mask_t allowedClusters = GID::getSourcesMask(O2DPLDisplaySpec::allowedClusters);

  GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-tracks")) & allowedTracks;
  GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-clusters")) & allowedClusters;
  if (!srcTrk.any() && !srcCl.any()) {
    if (cfgc.options().get<bool>("skipOnEmptyInput")) {
      LOG(info) << "No valid inputs for event display, disabling event display";
      return std::move(specs);
    }
    throw std::runtime_error("No input configured");
  }

  auto isRangeEnabled = [&opts = cfgc.options()](const char* min_name, const char* max_name) {
    EveWorkflowHelper::Bracket bracket{opts.get<float>(min_name), opts.get<float>(max_name)};
    bool optEnabled = false;

    if (bracket.getMin() < 0 && bracket.getMax() < 0) {
      optEnabled = false;
    } else if (bracket.getMin() >= 0 && bracket.getMax() >= 0) {
      optEnabled = true;

      if (bracket.isInvalid()) {
        throw std::runtime_error(fmt::format("{}, {} bracket is invalid", min_name, max_name));
      }
    } else {
      throw std::runtime_error(fmt::format("Both boundaries, {} and {}, have to be specified at the same time", min_name, max_name));
    }

    return std::make_tuple(optEnabled, bracket);
  };

  const auto [filterTime, timeBracket] = isRangeEnabled("filter-time-min", "filter-time-max");

  const auto etaRange = cfgc.options().get<float>("remove-tpc-abs-eta");

  bool removeTPCEta = false;
  EveWorkflowHelper::Bracket etaBracket;

  if (etaRange != 0.f) {
    etaBracket = EveWorkflowHelper::Bracket{-etaRange, etaRange};
    removeTPCEta = true;
  }

  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTrk, useMC);
  dataRequest->requestClusters(srcCl, useMC);

  auto filterITSROF = cfgc.options().get<bool>("filter-its-rof");

  if (filterITSROF) {
    dataRequest->requestIRFramesITS();
    InputHelper::addInputSpecsIRFramesITS(cfgc, specs);
  }

  auto primaryVertexMode = cfgc.options().get<bool>("primary-vertex-mode");
  auto maxPrimaryVertices = cfgc.options().get<int>("max-primary-vertices");

  InputHelper::addInputSpecs(cfgc, specs, srcCl, srcTrk, srcTrk, useMC);
  if (primaryVertexMode) {
    dataRequest->requestPrimaryVertertices(useMC);
    InputHelper::addInputSpecsPVertex(cfgc, specs, useMC);
  }

  auto minITSTracks = cfgc.options().get<int>("min-its-tracks");
  auto minTracks = cfgc.options().get<int>("min-tracks");
  auto onlyNthEvent = cfgc.options().get<int>("only-nth-event");
  auto tracksSorting = cfgc.options().get<bool>("track-sorting");
  auto primaryVertexTriggers = cfgc.options().get<bool>("primary-vertex-triggers");

  if (numberOfTracks == -1) {
    tracksSorting = false; // do not sort if all tracks are allowed
  }
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true); // query only once all objects except mag.field

  specs.emplace_back(DataProcessorSpec{
    "o2-eve-export",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<O2DPLDisplaySpec>(useMC, srcTrk, srcCl, dataRequest, ggRequest, jsonFolder, ext, timeInterval, numberOfFiles, numberOfTracks, eveHostNameMatch, minITSTracks, minTracks, filterITSROF, filterTime, timeBracket, removeTPCEta, etaBracket, tracksSorting, onlyNthEvent, primaryVertexMode, maxPrimaryVertices, primaryVertexTriggers)}});

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return std::move(specs);
}
