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
#include "EventVisualisationBase/EveConfParam.h"
#include "EventVisualisationBase/ConfigurationManager.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "CommonUtils/NameConf.h"
#include "TRDBase/GeometryFlat.h"
#include "TOFBase/Geo.h"
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
#include "DataFormatsITSMFT/DPLAlpideParamInitializer.h"
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
    {"receiver-hostname", VariantType::String, "arcbs04.cern.ch", {"name of the host where visualisation data is transmitted (only eve format)"}},
    {"receiver-port", VariantType::Int, 8001, {"port number of the host where visualisation data is transmitted (only eve format)"}},
    {"receiver-timeout", VariantType::Int, 300, {"socket connection timeout (ms)"}},
    {"use-only-files", VariantType::Bool, false, {"do not transmit visualisation data using sockets (only eve format)"}},
    {"use-only-sockets", VariantType::Bool, false, {"do not store visualisation data using filesystem"}},
    {"use-json-format", VariantType::Bool, false, {"instead of eve format (default) use json format"}},
    {"use-root-format", VariantType::Bool, false, {"instead of eve format (default) use root format"}},
    {"eve-hostname", VariantType::String, "", {"name of the host allowed to produce files (empty means no limit)"}},
    {"eve-dds-collection-index", VariantType::Int, -1, {"number of dpl collection allowed to produce files (-1 means no limit)"}},
    {"time-interval", VariantType::Int, 5000, {"time interval in milliseconds between stored files"}},
    {"disable-mc", VariantType::Bool, false, {"disable visualization of MC data"}},
    {"disable-write", VariantType::Bool, false, {"disable writing output files"}},
    {"display-clusters", VariantType::String, "ITS,TPC,TRD,TOF", {"comma-separated list of clusters to display"}},
    {"display-tracks", VariantType::String, "TPC,ITS,ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF", {"comma-separated list of tracks to display"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
    {"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings, e.g. EveConfParam content..."}},
    {"skipOnEmptyInput", VariantType::Bool, false, {"don't run the ED when no input is provided"}},
  };
  o2::itsmft::DPLAlpideParamInitializer::addConfigOption(options);
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
  const auto& conf = EveConfParam::Instance();

  if (!mEveHostNameMatch) {
    return;
  }
  if (conf.onlyNthEvent > 1 && mEventCounter++ % conf.onlyNthEvent) {
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

  EveWorkflowHelper helper;
  helper.setTPCVDrift(&mTPCVDriftHelper.getVDriftObject());
  helper.setRecoContainer(&recoCont);
  if (mEMCALCalibrator) {
    helper.setEMCALCellRecalibrator(mEMCALCalibrator.get());
  }

  helper.setITSROFs();
  helper.selectTracks(&(mData.mConfig.configCalib), mClMask, mTrkMask, mTrkMask);
  helper.selectTowers();
  helper.prepareITSClusters(mData.mITSDict);
  helper.prepareMFTClusters(mData.mMFTDict);

  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();

  std::size_t filesSaved = 0;
  const std::vector<std::string> dirs = o2::event_visualisation::DirectoryLoader::allFolders(this->mJsonPath);
  const std::string marker = "_";
  const std::vector<std::string> exts = {".json", ".root", ".eve"};
  auto processData = [&](const auto& dataMap) {
    for (const auto& keyVal : dataMap) {
      if (conf.maxPVs > 0 && filesSaved >= conf.maxPVs) {
        break;
      }
      if (conf.maxBytes > 0) {
        auto periodStart =
          duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - this->mTimeInterval.count();
        if (!DirectoryLoader::canCreateNextFile(dirs, marker, exts, periodStart, conf.maxBytes)) {
          LOGF(info, "Already too much data (> %d) to transfer in this period - event will not be not saved ...", conf.maxBytes);
          break;
        }
      }
      const auto pv = keyVal.first;
      bool save = false;
      if (conf.PVMode) {
        helper.draw(pv, conf.trackSorting);
        save = true;
      } else {
        helper.draw(pv, conf.trackSorting);
        save = true;
      }

      if (conf.minITSTracks > -1 && helper.mEvent.getDetectorTrackCount(detectors::DetID::ITS) < conf.minITSTracks) {
        save = false;
      }

      if (conf.minTracks > -1 && helper.mEvent.getTrackCount() < conf.minTracks) {
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
        helper.save(this->mJsonPath, this->mExt, conf.maxFiles, this->mReceiverHostname, this->mReceiverPort, this->mReceiverTimeout, this->mUseOnlyFiles, this->mUseOnlySockets);
        filesSaved++;
        currentTime = std::chrono::high_resolution_clock::now(); // time AFTER save
        this->mTimeStamp = currentTime;                          // next run AFTER period counted from last save
      }

      helper.clear();
    }
  };
  if (conf.PVTriggersMode) {
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

  // FIXME: find out why this does not work with 11.1.1
//  LOGP(info, "Tracks: {}", fmt::join(sourceStats, ", "));
}

void O2DPLDisplaySpec::endOfStream(EndOfStreamContext& ec)
{
}

void O2DPLDisplaySpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  mTPCVDriftHelper.extractCCDBInputs(pc);
  if (mTPCVDriftHelper.isUpdated()) {
    mTPCVDriftHelper.acknowledgeUpdate();
  }
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
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {

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
  const auto& conf = EveConfParam::Instance();

  bool useMC = !cfgc.options().get<bool>("disable-mc");
  bool disableWrite = cfgc.options().get<bool>("disable-write");

  auto receiverHostname = cfgc.options().get<std::string>("receiver-hostname");
  auto receiverPort = cfgc.options().get<int>("receiver-port");
  auto receiverTimeout = cfgc.options().get<int>("receiver-timeout");
  auto useOnlyFiles = cfgc.options().get<bool>("use-only-files");
  auto useOnlySockets = cfgc.options().get<bool>("use-only-sockets");

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

  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTrk, useMC);
  dataRequest->requestClusters(srcCl, useMC);

  if (conf.filterITSROF) {
    dataRequest->requestIRFramesITS();
    InputHelper::addInputSpecsIRFramesITS(cfgc, specs);
  }

  InputHelper::addInputSpecs(cfgc, specs, srcCl, srcTrk, srcTrk, useMC);
  if (conf.PVMode) {
    dataRequest->requestPrimaryVertices(useMC);
    InputHelper::addInputSpecsPVertex(cfgc, specs, useMC);
  }
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true); // query only once all objects except mag.field

  std::shared_ptr<o2::emcal::CalibLoader> emcalCalibLoader;
  if (conf.calibrateEMC) {
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
    AlgorithmSpec{adaptFromTask<O2DPLDisplaySpec>(disableWrite, useMC, srcTrk, srcCl, dataRequest, ggRequest,
                                                  emcalCalibLoader, jsonFolder, ext, timeInterval, eveHostNameMatch,
                                                  receiverHostname, receiverPort, receiverTimeout, useOnlyFiles, useOnlySockets)}});

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return std::move(specs);
}
