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

/// @file   GPUWorkflowTPC.cxx
/// @author David Rohr

#include "GPUWorkflow/GPUWorkflowSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/WorkflowSpec.h" // o2::framework::mergeInputs
#include "Framework/DataRefUtils.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/SerializationMethods.h"
#include "Framework/Logger.h"
#include "Framework/CallbackService.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/RawDataTypes.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"
#include "TPCReconstruction/TPCTrackingDigitsPreCheck.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCFastTransform.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonUtils/NameConf.h"
#include "TPCBase/RDHUtils.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2InterfaceQA.h"
#include "GPUO2Interface.h"
#include "CalibdEdxContainer.h"
#include "GPUNewCalibValues.h"
#include "TPCPadGainCalib.h"
#include "TPCZSLinkMapping.h"
#include "display/GPUDisplayInterface.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Utils.h"
#include "TPCBase/CDBInterface.h"
#include "TPCCalibration/VDriftHelper.h"
#include "CorrectionMapsHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "TPCCalibration/IDCContainer.h"
#include "TPCBase/DeadChannelMapCreator.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Algorithm/Parser.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/GeometryFlat.h"
#include "ITSBase/GeometryTGeo.h"
#include "CommonUtils/VerbosityConfig.h"
#include "CommonUtils/DebugStreamer.h"
#include <filesystem>
#include <memory> // for make_shared
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <regex>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <chrono>
#include "GPUReconstructionConvert.h"
#include "DetectorsRaw/RDHUtils.h"
#include <TStopwatch.h>
#include <TObjArray.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH1D.h>
#include <TGraphAsymmErrors.h>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::gpu;
using namespace o2::base;
using namespace o2::dataformats;

namespace o2::gpu
{

void GPURecoWorkflowSpec::initFunctionTPCCalib(InitContext& ic)
{
  mTPCDeadChannelMapCreator.reset(new o2::tpc::DeadChannelMapCreator());
  const auto deadMapSource = (mSpecConfig.tpcDeadMapSources > -1) ? static_cast<tpc::SourcesDeadMap>(mSpecConfig.tpcDeadMapSources) : tpc::SourcesDeadMap::All;
  mTPCDeadChannelMapCreator->init();
  mTPCDeadChannelMapCreator->setSource(deadMapSource);

  mCalibObjects.mdEdxCalibContainer.reset(new o2::tpc::CalibdEdxContainer());
  mTPCVDriftHelper.reset(new o2::tpc::VDriftHelper());
  mCalibObjects.mFastTransformHelper.reset(new o2::tpc::CorrectionMapsLoader());
  mCalibObjects.mFastTransform = std::move(o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  mCalibObjects.mFastTransformRef = std::move(o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  mCalibObjects.mFastTransformMShape = std::move(o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  mCalibObjects.mFastTransformHelper->setCorrMap(mCalibObjects.mFastTransform.get()); // just to reserve the space
  mCalibObjects.mFastTransformHelper->setCorrMapRef(mCalibObjects.mFastTransformRef.get());
  mCalibObjects.mFastTransformHelper->setLumiScaleType(mSpecConfig.lumiScaleType);
  mCalibObjects.mFastTransformHelper->setCorrMapMShape(mCalibObjects.mFastTransformMShape.get());
  mCalibObjects.mFastTransformHelper->setLumiScaleMode(mSpecConfig.lumiScaleMode);
  mCalibObjects.mFastTransformHelper->enableMShapeCorrection(mSpecConfig.enableMShape);
  if (mSpecConfig.outputTracks) {
    mCalibObjects.mFastTransformHelper->init(ic);
  }
  if (mConfParam->dEdxDisableTopologyPol) {
    LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB");
    mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);
  }

  if (mConfParam->dEdxDisableThresholdMap) {
    LOGP(info, "Disabling loading of threshold map from CCDB");
    mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalThresholdMap);
  }

  if (mConfParam->dEdxDisableGainMap) {
    LOGP(info, "Disabling loading of gain map from CCDB");
    mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalGainMap);
  }

  if (mConfParam->dEdxDisableResidualGainMap) {
    LOGP(info, "Disabling loading of residual gain map from CCDB");
    mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalResidualGainMap);
  }

  if (mConfParam->dEdxDisableResidualGain) {
    LOGP(info, "Disabling loading of residual gain calibration from CCDB");
    mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTimeGain);
  }

  if (mConfParam->dEdxUseFullGainMap) {
    LOGP(info, "Using the full gain map for correcting the cluster charge during calculation of the dE/dx");
    mCalibObjects.mdEdxCalibContainer->setUsageOfFullGainMap(true);
  }

  if (mConfParam->gainCalibDisableCCDB) {
    LOGP(info, "Disabling loading the TPC pad gain calibration from the CCDB");
    mUpdateGainMapCCDB = false;
  }

  // load from file
  if (!mConfParam->dEdxPolTopologyCorrFile.empty() || !mConfParam->dEdxCorrFile.empty() || !mConfParam->dEdxSplineTopologyCorrFile.empty()) {
    if (!mConfParam->dEdxPolTopologyCorrFile.empty()) {
      LOGP(info, "Loading dE/dx polynomial track topology correction from file: {}", mConfParam->dEdxPolTopologyCorrFile);
      mCalibObjects.mdEdxCalibContainer->loadPolTopologyCorrectionFromFile(mConfParam->dEdxPolTopologyCorrFile);

      LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB as it was already loaded from input file");
      mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);

      if (std::filesystem::exists(mConfParam->thresholdCalibFile)) {
        LOG(info) << "Loading tpc zero supression map from file " << mConfParam->thresholdCalibFile;
        const auto* thresholdMap = o2::tpc::utils::readCalPads(mConfParam->thresholdCalibFile, "ThresholdMap")[0];
        mCalibObjects.mdEdxCalibContainer->setZeroSupresssionThreshold(*thresholdMap);

        LOGP(info, "Disabling loading of threshold map from CCDB as it was already loaded from input file");
        mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalThresholdMap);
      } else {
        if (not mConfParam->thresholdCalibFile.empty()) {
          LOG(warn) << "Couldn't find tpc zero supression file " << mConfParam->thresholdCalibFile << ". Not setting any zero supression.";
        }
        LOG(info) << "Setting default zero supression map";
        mCalibObjects.mdEdxCalibContainer->setDefaultZeroSupresssionThreshold();
      }
    } else if (!mConfParam->dEdxSplineTopologyCorrFile.empty()) {
      LOGP(info, "Loading dE/dx spline track topology correction from file: {}", mConfParam->dEdxSplineTopologyCorrFile);
      mCalibObjects.mdEdxCalibContainer->loadSplineTopologyCorrectionFromFile(mConfParam->dEdxSplineTopologyCorrFile);

      LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB as splines were loaded from input file");
      mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);
    }
    if (!mConfParam->dEdxCorrFile.empty()) {
      LOGP(info, "Loading dEdx correction from file: {}", mConfParam->dEdxCorrFile);
      mCalibObjects.mdEdxCalibContainer->loadResidualCorrectionFromFile(mConfParam->dEdxCorrFile);

      LOGP(info, "Disabling loading of residual gain calibration from CCDB as it was already loaded from input file");
      mCalibObjects.mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTimeGain);
    }
  }

  if (mConfParam->dEdxPolTopologyCorrFile.empty() && mConfParam->dEdxSplineTopologyCorrFile.empty()) {
    // setting default topology correction to allocate enough memory
    LOG(info) << "Setting default dE/dx polynomial track topology correction to allocate enough memory";
    mCalibObjects.mdEdxCalibContainer->setDefaultPolTopologyCorrection();
  }

  GPUO2InterfaceConfiguration& config = *mConfig.get();
  mConfig->configCalib.dEdxCalibContainer = mCalibObjects.mdEdxCalibContainer.get();

  if (std::filesystem::exists(mConfParam->gainCalibFile)) {
    LOG(info) << "Loading tpc gain correction from file " << mConfParam->gainCalibFile;
    const auto* gainMap = o2::tpc::utils::readCalPads(mConfParam->gainCalibFile, "GainMap")[0];
    mCalibObjects.mTPCPadGainCalib = GPUO2Interface::getPadGainCalib(*gainMap);

    LOGP(info, "Disabling loading the TPC gain correction map from the CCDB as it was already loaded from input file");
    mUpdateGainMapCCDB = false;
  } else {
    if (not mConfParam->gainCalibFile.empty()) {
      LOG(warn) << "Couldn't find tpc gain correction file " << mConfParam->gainCalibFile << ". Not applying any gain correction.";
    }
    mCalibObjects.mTPCPadGainCalib = GPUO2Interface::getPadGainCalibDefault();
    mCalibObjects.mTPCPadGainCalib->getGainCorrection(30, 5, 5);
  }
  mConfig->configCalib.tpcPadGain = mCalibObjects.mTPCPadGainCalib.get();

  mTPCZSLinkMapping.reset(new TPCZSLinkMapping{tpc::Mapper::instance()});
  mConfig->configCalib.tpcZSLinkMapping = mTPCZSLinkMapping.get();
}

void GPURecoWorkflowSpec::finaliseCCDBTPC(ConcreteDataMatcher& matcher, void* obj)
{
  const o2::tpc::CalibdEdxContainer* dEdxCalibContainer = mCalibObjects.mdEdxCalibContainer.get();

  auto copyCalibsToBuffer = [this, dEdxCalibContainer]() {
    if (!(mdEdxCalibContainerBufferNew)) {
      mdEdxCalibContainerBufferNew = std::make_unique<o2::tpc::CalibdEdxContainer>();
      mdEdxCalibContainerBufferNew->cloneFromObject(*dEdxCalibContainer, nullptr);
    }
  };

  if (matcher == ConcreteDataMatcher(gDataOriginTPC, "PADGAINFULL", 0)) {
    LOGP(info, "Updating gain map from CCDB");
    const auto* gainMap = static_cast<o2::tpc::CalDet<float>*>(obj);

    if (dEdxCalibContainer->isCorrectionCCDB(o2::tpc::CalibsdEdx::CalGainMap) && mSpecConfig.outputTracks) {
      copyCalibsToBuffer();
      const float minGain = 0;
      const float maxGain = 2;
      mdEdxCalibContainerBufferNew.get()->setGainMap(*gainMap, minGain, maxGain);
    }

    if (mUpdateGainMapCCDB && mSpecConfig.caClusterer) {
      mTPCPadGainCalibBufferNew = GPUO2Interface::getPadGainCalib(*gainMap);
    }

  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "PADGAINRESIDUAL", 0)) {
    LOGP(info, "Updating residual gain map from CCDB");
    copyCalibsToBuffer();
    const auto* gainMapResidual = static_cast<std::unordered_map<string, o2::tpc::CalDet<float>>*>(obj);
    const float minResidualGain = 0.7f;
    const float maxResidualGain = 1.3f;
    mdEdxCalibContainerBufferNew.get()->setGainMapResidual(gainMapResidual->at("GainMap"), minResidualGain, maxResidualGain);
  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "PADTHRESHOLD", 0)) {
    LOGP(info, "Updating threshold map from CCDB");
    copyCalibsToBuffer();
    const auto* thresholdMap = static_cast<std::unordered_map<string, o2::tpc::CalDet<float>>*>(obj);
    mdEdxCalibContainerBufferNew.get()->setZeroSupresssionThreshold(thresholdMap->at("ThresholdMap"));
  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "TOPOLOGYGAIN", 0) && !(dEdxCalibContainer->isTopologyCorrectionSplinesSet())) {
    LOGP(info, "Updating Q topology correction from CCDB");
    copyCalibsToBuffer();
    const auto* topologyCorr = static_cast<o2::tpc::CalibdEdxTrackTopologyPolContainer*>(obj);
    o2::tpc::CalibdEdxTrackTopologyPol calibTrackTopology;
    calibTrackTopology.setFromContainer(*topologyCorr);
    mdEdxCalibContainerBufferNew->setPolTopologyCorrection(calibTrackTopology);
  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "TIMEGAIN", 0)) {
    LOGP(info, "Updating residual gain correction from CCDB");
    copyCalibsToBuffer();
    const auto* residualCorr = static_cast<o2::tpc::CalibdEdxCorrection*>(obj);
    mdEdxCalibContainerBufferNew->setResidualCorrection(*residualCorr);
  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "IDCPADFLAGS", 0)) {
    copyCalibsToBuffer();
    const auto* padFlags = static_cast<o2::tpc::CalDet<o2::tpc::PadFlags>*>(obj);
    mTPCDeadChannelMapCreator->setDeadChannelMapIDCPadStatus(*padFlags);
    mTPCDeadChannelMapCreator->finalizeDeadChannelMap();
    mdEdxCalibContainerBufferNew.get()->setDeadChannelMap(mTPCDeadChannelMapCreator->getDeadChannelMap());
    LOGP(info, "Updating dead channel map with IDC pad flags: {} / {} dead pads from pad flags / total",
         mTPCDeadChannelMapCreator->getDeadChannelMapIDC().getSum<int>(), mTPCDeadChannelMapCreator->getDeadChannelMap().getSum<int>());
  } else if (matcher == ConcreteDataMatcher(gDataOriginTPC, "TPCRUNINFO", 0)) {
    copyCalibsToBuffer();
    mTPCDeadChannelMapCreator->loadFEEConfigViaRunInfoTS(mCreationForCalib);
    mTPCDeadChannelMapCreator->finalizeDeadChannelMap();
    mdEdxCalibContainerBufferNew.get()->setDeadChannelMap(mTPCDeadChannelMapCreator->getDeadChannelMap());
    LOGP(info, "Updating dead channel map with the FEE info loaded via TPCRUNINFO for creation time {}: {} / {} dead pads from FEE info / total",
         mCreationForCalib, mTPCDeadChannelMapCreator->getDeadChannelMapFEE().getSum<int>(), mTPCDeadChannelMapCreator->getDeadChannelMap().getSum<int>());
  } else if (mTPCVDriftHelper->accountCCDBInputs(matcher, obj)) {
  } else if (mCalibObjects.mFastTransformHelper->accountCCDBInputs(matcher, obj)) {
  }
}

template <>
bool GPURecoWorkflowSpec::fetchCalibsCCDBTPC<GPUCalibObjectsConst>(ProcessingContext& pc, GPUCalibObjectsConst& newCalibObjects, calibObjectStruct& oldCalibObjects)
{
  // update calibrations for clustering and tracking
  mCreationForCalib = pc.services().get<o2::framework::TimingInfo>().creation;
  bool mustUpdate = false;
  if ((mSpecConfig.outputTracks || mSpecConfig.caClusterer) && !mConfParam->disableCalibUpdates) {
    const o2::tpc::CalibdEdxContainer* dEdxCalibContainer = mCalibObjects.mdEdxCalibContainer.get();

    // this calibration is defined for clustering and tracking
    if (dEdxCalibContainer->isCorrectionCCDB(o2::tpc::CalibsdEdx::CalGainMap) || mUpdateGainMapCCDB) {
      pc.inputs().get<o2::tpc::CalDet<float>*>("tpcgain");
    }

    // these calibrations are only defined for the tracking
    if (mSpecConfig.outputTracks) {
      // update the calibration objects in case they changed in the CCDB
      if (dEdxCalibContainer->isCorrectionCCDB(o2::tpc::CalibsdEdx::CalThresholdMap)) {
        pc.inputs().get<std::unordered_map<std::string, o2::tpc::CalDet<float>>*>("tpcthreshold");
      }

      if (mTPCDeadChannelMapCreator->useSource(tpc::SourcesDeadMap::IDCPadStatus)) {
        pc.inputs().get<o2::tpc::CalDet<tpc::PadFlags>*>("tpcidcpadflags");
      }

      if (mTPCDeadChannelMapCreator->useSource(tpc::SourcesDeadMap::FEEConfig)) {
        pc.inputs().get<char*>("tpcruninfo");
      }

      if (dEdxCalibContainer->isCorrectionCCDB(o2::tpc::CalibsdEdx::CalResidualGainMap)) {
        pc.inputs().get<std::unordered_map<std::string, o2::tpc::CalDet<float>>*>("tpcgainresidual");
      }

      if (dEdxCalibContainer->isCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol)) {
        pc.inputs().get<o2::tpc::CalibdEdxTrackTopologyPolContainer*>("tpctopologygain");
      }

      if (dEdxCalibContainer->isCorrectionCCDB(o2::tpc::CalibsdEdx::CalTimeGain)) {
        pc.inputs().get<o2::tpc::CalibdEdxCorrection*>("tpctimegain");
      }

      if (mSpecConfig.outputTracks) {
        mTPCVDriftHelper->extractCCDBInputs(pc);
        mCalibObjects.mFastTransformHelper->extractCCDBInputs(pc);
      }
      if (mTPCVDriftHelper->isUpdated() || mCalibObjects.mFastTransformHelper->isUpdated()) {
        const auto& vd = mTPCVDriftHelper->getVDriftObject();
        LOGP(info, "Updating{}TPC fast transform map and/or VDrift factor of {} wrt reference {} and TDrift offset {} wrt reference {} from source {}",
             mCalibObjects.mFastTransformHelper->isUpdated() ? " new " : " old ",
             vd.corrFact, vd.refVDrift, vd.timeOffsetCorr, vd.refTimeOffset, mTPCVDriftHelper->getSourceName());

        bool mustUpdateHelper = false;
        if (mTPCVDriftHelper->isUpdated() || mCalibObjects.mFastTransformHelper->isUpdatedMap()) {
          oldCalibObjects.mFastTransform = std::move(mCalibObjects.mFastTransform);
          mCalibObjects.mFastTransform.reset(new TPCFastTransform);
          mCalibObjects.mFastTransform->cloneFromObject(*mCalibObjects.mFastTransformHelper->getCorrMap(), nullptr);
          o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mCalibObjects.mFastTransform, 0, vd.corrFact, vd.refVDrift, vd.getTimeOffset());
          newCalibObjects.fastTransform = mCalibObjects.mFastTransform.get();
          mustUpdateHelper = true;
        }
        if (mTPCVDriftHelper->isUpdated() || mCalibObjects.mFastTransformHelper->isUpdatedMapRef()) {
          oldCalibObjects.mFastTransformRef = std::move(mCalibObjects.mFastTransformRef);
          mCalibObjects.mFastTransformRef.reset(new TPCFastTransform);
          mCalibObjects.mFastTransformRef->cloneFromObject(*mCalibObjects.mFastTransformHelper->getCorrMapRef(), nullptr);
          o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mCalibObjects.mFastTransformRef, 0, vd.corrFact, vd.refVDrift, vd.getTimeOffset());
          newCalibObjects.fastTransformRef = mCalibObjects.mFastTransformRef.get();
          mustUpdateHelper = true;
        }
        if (mTPCVDriftHelper->isUpdated() || mCalibObjects.mFastTransformHelper->isUpdatedMapMShape()) {
          oldCalibObjects.mFastTransformMShape = std::move(mCalibObjects.mFastTransformMShape);
          mCalibObjects.mFastTransformMShape.reset(new TPCFastTransform);
          mCalibObjects.mFastTransformMShape->cloneFromObject(*mCalibObjects.mFastTransformHelper->getCorrMapMShape(), nullptr);
          o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mCalibObjects.mFastTransformMShape, 0, vd.corrFact, vd.refVDrift, vd.getTimeOffset());
          newCalibObjects.fastTransformMShape = mCalibObjects.mFastTransformMShape.get();
          mustUpdateHelper = true;
        }
        if (mustUpdateHelper || mCalibObjects.mFastTransformHelper->isUpdatedLumi()) {
          oldCalibObjects.mFastTransformHelper = std::move(mCalibObjects.mFastTransformHelper);
          mCalibObjects.mFastTransformHelper.reset(new o2::tpc::CorrectionMapsLoader);
          mCalibObjects.mFastTransformHelper->copySettings(*oldCalibObjects.mFastTransformHelper);
          mCalibObjects.mFastTransformHelper->setCorrMap(mCalibObjects.mFastTransform.get());
          mCalibObjects.mFastTransformHelper->setCorrMapRef(mCalibObjects.mFastTransformRef.get());
          mCalibObjects.mFastTransformHelper->setCorrMapMShape(mCalibObjects.mFastTransformMShape.get());
          mCalibObjects.mFastTransformHelper->acknowledgeUpdate();
          newCalibObjects.fastTransformHelper = mCalibObjects.mFastTransformHelper.get();
        }
        mustUpdate = true;
        mTPCVDriftHelper->acknowledgeUpdate();
        mCalibObjects.mFastTransformHelper->acknowledgeUpdate();
      }
    }

    if (mdEdxCalibContainerBufferNew) {
      oldCalibObjects.mdEdxCalibContainer = std::move(mCalibObjects.mdEdxCalibContainer);
      mCalibObjects.mdEdxCalibContainer = std::move(mdEdxCalibContainerBufferNew);
      newCalibObjects.dEdxCalibContainer = mCalibObjects.mdEdxCalibContainer.get();
      mustUpdate = true;
    }

    if (mTPCPadGainCalibBufferNew) {
      oldCalibObjects.mTPCPadGainCalib = std::move(mCalibObjects.mTPCPadGainCalib);
      mCalibObjects.mTPCPadGainCalib = std::move(mTPCPadGainCalibBufferNew);
      newCalibObjects.tpcPadGain = mCalibObjects.mTPCPadGainCalib.get();
      mustUpdate = true;
    }
  }
  return mustUpdate;
}

void GPURecoWorkflowSpec::doTrackTuneTPC(GPUTrackingInOutPointers& ptrs, char* buffout)
{
  using TrackTunePar = o2::globaltracking::TrackTuneParams;
  const auto& trackTune = TrackTunePar::Instance();
  if (ptrs.nOutputTracksTPCO2 && trackTune.sourceLevelTPC &&
      (trackTune.useTPCInnerCorr || trackTune.useTPCOuterCorr ||
       trackTune.tpcCovInnerType != TrackTunePar::AddCovType::Disable || trackTune.tpcCovOuterType != TrackTunePar::AddCovType::Disable)) {
    if (((const void*)ptrs.outputTracksTPCO2) != ((const void*)buffout)) {
      throw std::runtime_error("Buffer does not match span");
    }
    o2::tpc::TrackTPC* tpcTracks = reinterpret_cast<o2::tpc::TrackTPC*>(buffout);
    float scale = mCalibObjects.mFastTransformHelper->getInstLumiCTP();
    if (scale < 0.f) {
      scale = 0.f;
    }
    auto diagInner = trackTune.getCovInnerTotal(scale);
    auto diagOuter = trackTune.getCovOuterTotal(scale);

    for (unsigned int itr = 0; itr < ptrs.nOutputTracksTPCO2; itr++) {
      auto& trc = tpcTracks[itr];
      if (trackTune.useTPCInnerCorr) {
        trc.updateParams(trackTune.tpcParInner);
      }
      if (trackTune.tpcCovInnerType != TrackTunePar::AddCovType::Disable) {
        trc.updateCov(diagInner, trackTune.tpcCovInnerType == TrackTunePar::AddCovType::WithCorrelations);
      }
      if (trackTune.useTPCOuterCorr) {
        trc.getParamOut().updateParams(trackTune.tpcParOuter);
      }
      if (trackTune.tpcCovOuterType != TrackTunePar::AddCovType::Disable) {
        trc.getParamOut().updateCov(diagOuter, trackTune.tpcCovOuterType == TrackTunePar::AddCovType::WithCorrelations);
      }
    }
  }
}

} // namespace o2::gpu
