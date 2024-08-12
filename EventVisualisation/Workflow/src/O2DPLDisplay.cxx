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
#include "EMCALCalib/CellRecalibrator.h"
#include "EMCALWorkflow/CalibLoader.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <EventVisualisationBase/DirectoryLoader.h>
#include "DataFormatsMCH/Cluster.h"
#include <unistd.h>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

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
    {"use-json-format", VariantType::Bool, false, {"instead of eve format (default) use json format"}},
    {"use-root-format", VariantType::Bool, false, {"instead of eve format (default) use root format"}},
    {"eve-hostname", VariantType::String, "", {"name of the host allowed to produce files (empty means no limit)"}},
    {"eve-dds-collection-index", VariantType::Int, -1, {"number of dpl collection allowed to produce files (-1 means no limit)"}},
    {"number-of_files", VariantType::Int, 150, {"maximum number of json files in folder"}},
    {"number-of_tracks", VariantType::Int, -1, {"maximum number of track stored in json file (-1 means no limit)"}},
    {"number-of_bytes", VariantType::Int, 3000000, {"number of bytes stored in time interval which stops producing new data file (-1 means no limit)"}},
    {"time-interval", VariantType::Int, 5000, {"time interval in milliseconds between stored files"}},
    {"disable-mc", VariantType::Bool, false, {"disable visualization of MC data"}},
    {"disable-write", VariantType::Bool, false, {"disable writing output files"}},
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
    {"primary-vertex-triggers", VariantType::Bool, false, {"instead of drawing vertices with tracks (and maybe calorimeter triggers), draw vertices with calorimeter triggers (and maybe tracks)"}},
    {"no-calibrate-emcal", VariantType::Bool, false, {"Do not apply on-the-fly EMCAL calibration"}},
    {"emcal-max-celltime", VariantType::Float, 100.f, {"Max. EMCAL cell time (in ns)"}},
    {"emcal-min-cellenergy", VariantType::Float, 0.3f, {"Min. EMCAL cell energy (in GeV)"}},
    {"primary-vertex-min-z", VariantType::Float, -o2::constants::math::VeryBig, {"minimum z position for primary vertex"}},
    {"primary-vertex-max-z", VariantType::Float, o2::constants::math::VeryBig, {"maximum z position for primary vertex"}},
    {"primary-vertex-min-x", VariantType::Float, -o2::constants::math::VeryBig, {"minimum x position for primary vertex"}},
    {"primary-vertex-max-x", VariantType::Float, o2::constants::math::VeryBig, {"maximum x position for primary vertex"}},
    {"primary-vertex-min-y", VariantType::Float, -o2::constants::math::VeryBig, {"minimum y position for primary vertex"}},
    {"primary-vertex-max-y", VariantType::Float, o2::constants::math::VeryBig, {"maximum y position for primary vertex"}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // main method must be included here (otherwise customize not used)
void O2DPLDisplaySpec::init(InitContext& ic)
{
  LOGF(info, "------------------------    O2DPLDisplay::init version ", o2_eve_version, "    ------------------------------------");
  mData.mConfig.configProcessing.runMC = mUseMC;
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  if (mEMCALCalibLoader) {
    mEMCALCalibrator = std::make_unique<o2::emcal::CellRecalibrator>();
  }
}

void O2DPLDisplaySpec::run(ProcessingContext& pc)
{
  if (!this->mEveHostNameMatch) {
    return;
  }
  if (this->mOnlyNthEvent && this->mEventCounter++ % this->mOnlyNthEvent != 0) {
    return;
  }
  LOGF(info, "------------------------    O2DPLDisplay::run version ", o2_eve_version, "    ------------------------------------");
  // filtering out any run which occur before reaching next time interval
  auto currentTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = currentTime - this->mTimeStamp;
  if (elapsed < this->mTimeInterval) {
    return; // skip this run - it is too often
  }
  this->mTimeStamp = currentTime; // next run AFTER period counted from last run, even if there will be not any save
  o2::globaltracking::RecoContainer recoCont;
  recoCont.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc); // Make sure that this is called after the RecoContainer collect data, since some condition objects are fetched there
  if (mEMCALCalibLoader) {
    mEMCALCalibLoader->checkUpdates(pc);
    if (mEMCALCalibLoader->hasUpdateBadChannelMap()) {
      mEMCALCalibrator->setBadChannelMap(mEMCALCalibLoader->getBadChannelMap());
    }
    if (mEMCALCalibLoader->hasUpdateTimeCalib()) {
      mEMCALCalibrator->setTimeCalibration(mEMCALCalibLoader->getTimeCalibration());
    }
    if (mEMCALCalibLoader->hasUpdateGainCalib()) {
      mEMCALCalibrator->setGainCalibration(mEMCALCalibLoader->getGainCalibration());
    }
  }

  EveWorkflowHelper::FilterSet enabledFilters;

  enabledFilters.set(EveWorkflowHelper::Filter::ITSROF, this->mFilterITSROF);
  enabledFilters.set(EveWorkflowHelper::Filter::TimeBracket, this->mFilterTime);
  enabledFilters.set(EveWorkflowHelper::Filter::EtaBracket, this->mRemoveTPCEta);
  enabledFilters.set(EveWorkflowHelper::Filter::TotalNTracks, this->mNumberOfTracks != -1);
  EveWorkflowHelper helper(enabledFilters, this->mNumberOfTracks, this->mTimeBracket, this->mEtaBracket, this->mPrimaryVertexMode);
  helper.setRecoContainer(&recoCont);
  if (mEMCALCalibrator) {
    helper.setEMCALCellRecalibrator(mEMCALCalibrator.get());
  }
  helper.setMaxEMCALCellTime(mEMCALMaxCellTime);
  helper.setMinEMCALCellEnergy(mEMCALMinCellEnergy);

  helper.setITSROFs();
  helper.selectTracks(&(mData.mConfig.configCalib), mClMask, mTrkMask, mTrkMask);
  helper.selectTowers();
  helper.prepareITSClusters(mData.mITSDict);
  helper.prepareMFTClusters(mData.mMFTDict);

  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();

  std::size_t filesSaved = 0;
  const std::vector<std::string> dirs = o2::event_visualisation::DirectoryLoader::allFolders(this->mJsonPath);
  const std::string marker = "_";
  const std::vector<std::string> exts = {
    ".json", ".root", ".eve"};
  auto processData = [&](const auto& dataMap) {
    for (const auto& keyVal : dataMap) {
      if (filesSaved >= mMaxPrimaryVertices) {
        break;
      }
      if (this->mNumberOfBytes != -1) {
        auto periodStart =
          duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - this->mTimeInterval.count();
        if (!DirectoryLoader::canCreateNextFile(
              dirs, marker, exts, periodStart, this->mNumberOfBytes)) {
          LOGF(info, "Already too much data (> %d) to transfer in this period - event will not be not saved ...", this->mNumberOfBytes);
          break;
        }
      }
      const auto pv = keyVal.first;
      bool save = false;
      if (mPrimaryVertexMode) {
        auto primaryVertex = recoCont.getPrimaryVertices()[pv];
        auto primaryVertex_X = primaryVertex.getX();
        auto primaryVertex_Y = primaryVertex.getY();
        auto primaryVertex_Z = primaryVertex.getZ();
        if ((primaryVertex_X >= mPrimaryVertexMinX) & (primaryVertex_X <= mPrimaryVertexMaxX) & (primaryVertex_Y >= mPrimaryVertexMinY) & (primaryVertex_Y <= mPrimaryVertexMaxY) & (primaryVertex_Z >= mPrimaryVertexMinZ) & (primaryVertex_Z <= mPrimaryVertexMaxZ)) {
          helper.draw(pv, mTrackSorting);
          save = true;
        }
      } else {
        helper.draw(pv, mTrackSorting);
        save = true;
      }

      if (this->mMinITSTracks != -1 && helper.mEvent.getDetectorTrackCount(detectors::DetID::ITS) < this->mMinITSTracks) {
        save = false;
      }

      if (this->mMinTracks != -1 && helper.mEvent.getTrackCount() < this->mMinTracks) {
        save = false;
      }

      if (this->mDisableWrite) {
        save = false;
      }

      if (save) {
        helper.mEvent.setClMask(this->mClMask.to_ulong());
        helper.mEvent.setTrkMask(this->mTrkMask.to_ulong());
        helper.mEvent.setRunNumber(tinfo.runNumber);
        helper.mEvent.setTfCounter(tinfo.tfCounter);
        helper.mEvent.setFirstTForbit(tinfo.firstTForbit);
        helper.mEvent.setRunType(this->mRunType);
        helper.mEvent.setPrimaryVertex(pv);
        helper.mEvent.setCreationTime(tinfo.creation);
        helper.save(this->mJsonPath, this->mExt, this->mNumberOfFiles);
        filesSaved++;
        currentTime = std::chrono::high_resolution_clock::now(); // time AFTER save
        this->mTimeStamp = currentTime;                          // next run AFTER period counted from last save
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
    mRunType = grpECS->getRunType();
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
  if (mEMCALCalibLoader && mEMCALCalibLoader->finalizeCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOGF(info, "ITS cluster dictionary updated");
    mData.setITSDict((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "CLUSDICT", 0)) {
    LOGF(info, "MFT cluster dictionary updated");
    mData.setMFTDict((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  LOGF(info, "------------------------    defineDataProcessing ", o2_eve_version, "    ------------------------------------");

  WorkflowSpec specs;

  auto jsonFolder = cfgc.options().get<std::string>("jsons-folder");
  std::string ext = ".eve"; // root files are default format
  auto useJsonFormat = cfgc.options().get<bool>("use-json-format");
  if (useJsonFormat) {
    ext = ".json";
  }
  auto useROOTFormat = cfgc.options().get<bool>("use-root-format");
  if (useROOTFormat) {
    ext = ".root";
  }
  auto eveHostName = cfgc.options().get<std::string>("eve-hostname");
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  bool useMC = !cfgc.options().get<bool>("disable-mc");
  bool disableWrite = cfgc.options().get<bool>("disable-write");

  char hostname[_POSIX_HOST_NAME_MAX];
  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  bool eveHostNameMatch = eveHostName.empty() || eveHostName == hostname;

  int eveDDSColIdx = cfgc.options().get<int>("eve-dds-collection-index");
  if (eveDDSColIdx != -1) {
    char* colIdx = getenv("DDS_COLLECTION_INDEX");
    int myIdx = colIdx ? atoi(colIdx) : -1;
    if (myIdx == eveDDSColIdx) {
      LOGF(important, "Restricting DPL Display to collection index, my index ", myIdx, ", enabled ", int(myIdx == eveDDSColIdx));
    } else {
      LOGF(info, "Restricting DPL Display to collection index, my index ", myIdx, ", enabled ", int(myIdx == eveDDSColIdx));
    }
    eveHostNameMatch &= myIdx == eveDDSColIdx;
  }

  std::chrono::milliseconds timeInterval(cfgc.options().get<int>("time-interval"));
  int numberOfFiles = cfgc.options().get<int>("number-of_files");
  int numberOfTracks = cfgc.options().get<int>("number-of_tracks");
  int numberOfBytes = cfgc.options().get<int>("number-of_bytes");

  GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-tracks"));
  GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-clusters"));

  if (srcTrk[GID::MFTMCH] && srcTrk[GID::MCHMID]) {
    srcTrk |= GID::getSourceMask(GID::MFTMCHMID);
  }

  const GID::mask_t allowedTracks = GID::getSourcesMask(O2DPLDisplaySpec::allowedTracks);
  const GID::mask_t allowedClusters = GID::getSourcesMask(O2DPLDisplaySpec::allowedClusters);

  srcTrk &= allowedTracks;
  srcCl &= allowedClusters;

  if (!srcTrk.any() && !srcCl.any()) {
    if (cfgc.options().get<bool>("skipOnEmptyInput")) {
      LOGF(info, "No valid inputs for event display, disabling event display");
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
    dataRequest->requestPrimaryVertices(useMC);
    InputHelper::addInputSpecsPVertex(cfgc, specs, useMC);
  }

  auto minITSTracks = cfgc.options().get<int>("min-its-tracks");
  auto minTracks = cfgc.options().get<int>("min-tracks");
  auto onlyNthEvent = cfgc.options().get<int>("only-nth-event");
  auto tracksSorting = cfgc.options().get<bool>("track-sorting");
  auto primaryVertexTriggers = cfgc.options().get<bool>("primary-vertex-triggers");
  auto primaryVertexMinZ = cfgc.options().get<float>("primary-vertex-min-z");
  auto primaryVertexMaxZ = cfgc.options().get<float>("primary-vertex-max-z");
  auto primaryVertexMinX = cfgc.options().get<float>("primary-vertex-min-x");
  auto primaryVertexMaxX = cfgc.options().get<float>("primary-vertex-max-x");
  auto primaryVertexMinY = cfgc.options().get<float>("primary-vertex-min-y");
  auto primaryVertexMaxY = cfgc.options().get<float>("primary-vertex-max-y");
  auto maxEMCALCellTime = cfgc.options().get<float>("emcal-max-celltime");
  auto minEMCALCellEnergy = cfgc.options().get<float>("emcal-min-cellenergy");

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

  std::shared_ptr<o2::emcal::CalibLoader> emcalCalibLoader;
  if (!cfgc.options().get<bool>("no-calibrate-emcal")) {
    emcalCalibLoader = std::make_shared<o2::emcal::CalibLoader>();
    emcalCalibLoader->enableTimeCalib(true);
    emcalCalibLoader->enableBadChannelMap(true);
    emcalCalibLoader->enableGainCalib(true);
    emcalCalibLoader->defineInputSpecs(dataRequest->inputs);
  }

  specs.emplace_back(DataProcessorSpec{
    "o2-eve-export",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<O2DPLDisplaySpec>(disableWrite, useMC, srcTrk, srcCl, dataRequest, ggRequest, emcalCalibLoader, jsonFolder, ext, timeInterval, numberOfFiles, numberOfTracks, numberOfBytes, eveHostNameMatch, minITSTracks, minTracks, filterITSROF, filterTime, timeBracket, removeTPCEta, etaBracket, tracksSorting, onlyNthEvent, primaryVertexMode, maxPrimaryVertices, primaryVertexTriggers, primaryVertexMinZ, primaryVertexMaxZ, primaryVertexMinX, primaryVertexMaxX, primaryVertexMinY, primaryVertexMaxY, maxEMCALCellTime, minEMCALCellEnergy)}});

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return std::move(specs);
}
