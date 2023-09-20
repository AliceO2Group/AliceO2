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
#include "DPLUtils/DPLRawParser.h"
#include "DPLUtils/DPLRawPageSequencer.h"
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
  mdEdxCalibContainer.reset(new o2::tpc::CalibdEdxContainer());
  mTPCVDriftHelper.reset(new o2::tpc::VDriftHelper());
  mFastTransformHelper.reset(new o2::tpc::CorrectionMapsLoader());
  mFastTransform = std::move(o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  mFastTransformRef = std::move(o2::tpc::TPCFastTransformHelperO2::instance()->create(0));
  mFastTransformHelper->setCorrMap(mFastTransform.get()); // just to reserve the space
  mFastTransformHelper->setCorrMapRef(mFastTransformRef.get());
  mFastTransformHelper->setLumiScaleMode(mSpecConfig.lumiScaleMode);
  if (mSpecConfig.outputTracks) {
    mFastTransformHelper->init(ic);
  }
  if (mConfParam->dEdxDisableTopologyPol) {
    LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);
  }

  if (mConfParam->dEdxDisableThresholdMap) {
    LOGP(info, "Disabling loading of threshold map from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalThresholdMap);
  }

  if (mConfParam->dEdxDisableGainMap) {
    LOGP(info, "Disabling loading of gain map from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalGainMap);
  }

  if (mConfParam->dEdxDisableResidualGainMap) {
    LOGP(info, "Disabling loading of residual gain map from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalResidualGainMap);
  }

  if (mConfParam->dEdxDisableResidualGain) {
    LOGP(info, "Disabling loading of residual gain calibration from CCDB");
    mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTimeGain);
  }

  if (mConfParam->dEdxUseFullGainMap) {
    LOGP(info, "Using the full gain map for correcting the cluster charge during calculation of the dE/dx");
    mdEdxCalibContainer->setUsageOfFullGainMap(true);
  }

  if (mConfParam->gainCalibDisableCCDB) {
    LOGP(info, "Disabling loading the TPC pad gain calibration from the CCDB");
    mUpdateGainMapCCDB = false;
  }

  // load from file
  if (!mConfParam->dEdxPolTopologyCorrFile.empty() || !mConfParam->dEdxCorrFile.empty() || !mConfParam->dEdxSplineTopologyCorrFile.empty()) {
    if (!mConfParam->dEdxPolTopologyCorrFile.empty()) {
      LOGP(info, "Loading dE/dx polynomial track topology correction from file: {}", mConfParam->dEdxPolTopologyCorrFile);
      mdEdxCalibContainer->loadPolTopologyCorrectionFromFile(mConfParam->dEdxPolTopologyCorrFile);

      LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB as it was already loaded from input file");
      mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);

      if (std::filesystem::exists(mConfParam->thresholdCalibFile)) {
        LOG(info) << "Loading tpc zero supression map from file " << mConfParam->thresholdCalibFile;
        const auto* thresholdMap = o2::tpc::utils::readCalPads(mConfParam->thresholdCalibFile, "ThresholdMap")[0];
        mdEdxCalibContainer->setZeroSupresssionThreshold(*thresholdMap);

        LOGP(info, "Disabling loading of threshold map from CCDB as it was already loaded from input file");
        mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalThresholdMap);
      } else {
        if (not mConfParam->thresholdCalibFile.empty()) {
          LOG(warn) << "Couldn't find tpc zero supression file " << mConfParam->thresholdCalibFile << ". Not setting any zero supression.";
        }
        LOG(info) << "Setting default zero supression map";
        mdEdxCalibContainer->setDefaultZeroSupresssionThreshold();
      }
    } else if (!mConfParam->dEdxSplineTopologyCorrFile.empty()) {
      LOGP(info, "Loading dE/dx spline track topology correction from file: {}", mConfParam->dEdxSplineTopologyCorrFile);
      mdEdxCalibContainer->loadSplineTopologyCorrectionFromFile(mConfParam->dEdxSplineTopologyCorrFile);

      LOGP(info, "Disabling loading of track topology correction using polynomials from CCDB as splines were loaded from input file");
      mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTopologyPol);
    }
    if (!mConfParam->dEdxCorrFile.empty()) {
      LOGP(info, "Loading dEdx correction from file: {}", mConfParam->dEdxCorrFile);
      mdEdxCalibContainer->loadResidualCorrectionFromFile(mConfParam->dEdxCorrFile);

      LOGP(info, "Disabling loading of residual gain calibration from CCDB as it was already loaded from input file");
      mdEdxCalibContainer->disableCorrectionCCDB(o2::tpc::CalibsdEdx::CalTimeGain);
    }
  }

  if (mConfParam->dEdxPolTopologyCorrFile.empty() && mConfParam->dEdxSplineTopologyCorrFile.empty()) {
    // setting default topology correction to allocate enough memory
    LOG(info) << "Setting default dE/dx polynomial track topology correction to allocate enough memory";
    mdEdxCalibContainer->setDefaultPolTopologyCorrection();
  }

  GPUO2InterfaceConfiguration& config = *mConfig.get();
  mConfig->configCalib.dEdxCalibContainer = mdEdxCalibContainer.get();

  if (std::filesystem::exists(mConfParam->gainCalibFile)) {
    LOG(info) << "Loading tpc gain correction from file " << mConfParam->gainCalibFile;
    const auto* gainMap = o2::tpc::utils::readCalPads(mConfParam->gainCalibFile, "GainMap")[0];
    mTPCPadGainCalib = GPUO2Interface::getPadGainCalib(*gainMap);

    LOGP(info, "Disabling loading the TPC gain correction map from the CCDB as it was already loaded from input file");
    mUpdateGainMapCCDB = false;
  } else {
    if (not mConfParam->gainCalibFile.empty()) {
      LOG(warn) << "Couldn't find tpc gain correction file " << mConfParam->gainCalibFile << ". Not applying any gain correction.";
    }
    mTPCPadGainCalib = GPUO2Interface::getPadGainCalibDefault();
    mTPCPadGainCalib->getGainCorrection(30, 5, 5);
  }
  mConfig->configCalib.tpcPadGain = mTPCPadGainCalib.get();

  mTPCZSLinkMapping.reset(new TPCZSLinkMapping{tpc::Mapper::instance()});
  mConfig->configCalib.tpcZSLinkMapping = mTPCZSLinkMapping.get();
}

void GPURecoWorkflowSpec::finaliseCCDBTPC(ConcreteDataMatcher& matcher, void* obj)
{
  const o2::tpc::CalibdEdxContainer* dEdxCalibContainer = mdEdxCalibContainer.get();

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
  } else if (mTPCVDriftHelper->accountCCDBInputs(matcher, obj)) {
  } else if (mFastTransformHelper->accountCCDBInputs(matcher, obj)) {
  }
}

template <>
bool GPURecoWorkflowSpec::fetchCalibsCCDBTPC<GPUCalibObjectsConst>(ProcessingContext& pc, GPUCalibObjectsConst& newCalibObjects)
{
  // update calibrations for clustering and tracking
  mMustUpdateFastTransform = false;
  if ((mSpecConfig.outputTracks || mSpecConfig.caClusterer) && !mConfParam->disableCalibUpdates) {
    const o2::tpc::CalibdEdxContainer* dEdxCalibContainer = mdEdxCalibContainer.get();

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
        mFastTransformHelper->extractCCDBInputs(pc);
      }
      if (mTPCVDriftHelper->isUpdated() || mFastTransformHelper->isUpdated()) {
        const auto& vd = mTPCVDriftHelper->getVDriftObject();
        LOGP(info, "Updating{}TPC fast transform map and/or VDrift factor of {} wrt reference {} and TDrift offset {} wrt reference {} from source {}",
             mFastTransformHelper->isUpdated() ? " new " : " old ",
             vd.corrFact, vd.refVDrift, vd.timeOffsetCorr, vd.refTimeOffset, mTPCVDriftHelper->getSourceName());

        if (mTPCVDriftHelper->isUpdated() || mFastTransformHelper->isUpdatedMap()) {
          mFastTransformNew.reset(new TPCFastTransform);
          mFastTransformNew->cloneFromObject(*mFastTransformHelper->getCorrMap(), nullptr);
          o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mFastTransformNew, 0, vd.corrFact, vd.refVDrift, vd.getTimeOffset());
          newCalibObjects.fastTransform = mFastTransformNew.get();
        }
        if (mTPCVDriftHelper->isUpdated() || mFastTransformHelper->isUpdatedMapRef()) {
          mFastTransformRefNew.reset(new TPCFastTransform);
          mFastTransformRefNew->cloneFromObject(*mFastTransformHelper->getCorrMapRef(), nullptr);
          o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mFastTransformRefNew, 0, vd.corrFact, vd.refVDrift, vd.getTimeOffset());
          newCalibObjects.fastTransformRef = mFastTransformRefNew.get();
        }
        if (mFastTransformNew || mFastTransformRefNew || mFastTransformHelper->isUpdatedLumi()) {
          mFastTransformHelperNew.reset(new o2::tpc::CorrectionMapsLoader);
          mFastTransformHelperNew->setInstLumi(mFastTransformHelper->getInstLumi(), false);
          mFastTransformHelperNew->setMeanLumi(mFastTransformHelper->getMeanLumi(), false);
          mFastTransformHelperNew->setUseCTPLumi(mFastTransformHelper->getUseCTPLumi());
          mFastTransformHelperNew->setMeanLumiOverride(mFastTransformHelper->getMeanLumiOverride());
          mFastTransformHelperNew->setInstLumiOverride(mFastTransformHelper->getInstLumiOverride());
          mFastTransformHelperNew->setLumiScaleMode(mFastTransformHelper->getLumiScaleMode());
          mFastTransformHelperNew->setCorrMap(mFastTransformNew ? mFastTransformNew.get() : mFastTransform.get());
          mFastTransformHelperNew->setCorrMapRef(mFastTransformRefNew ? mFastTransformRefNew.get() : mFastTransformRef.get());
          mFastTransformHelperNew->acknowledgeUpdate();
          newCalibObjects.fastTransformHelper = mFastTransformHelperNew.get();
        }
        mMustUpdateFastTransform = true;
        mTPCVDriftHelper->acknowledgeUpdate();
        mFastTransformHelper->acknowledgeUpdate();
      }
    }

    if (mdEdxCalibContainerBufferNew) {
      newCalibObjects.dEdxCalibContainer = mdEdxCalibContainerBufferNew.get();
    }

    if (mTPCPadGainCalibBufferNew) {
      newCalibObjects.tpcPadGain = mTPCPadGainCalibBufferNew.get();
    }

    return mdEdxCalibContainerBufferNew || mTPCPadGainCalibBufferNew || mMustUpdateFastTransform;
  }
  return false;
}

void GPURecoWorkflowSpec::cleanOldCalibsTPCPtrs()
{
  if (mdEdxCalibContainerBufferNew) {
    mdEdxCalibContainer = std::move(mdEdxCalibContainerBufferNew);
  }
  if (mTPCPadGainCalibBufferNew) {
    mTPCPadGainCalib = std::move(mTPCPadGainCalibBufferNew);
  }
  if (mFastTransformNew) {
    mFastTransform = std::move(mFastTransformNew);
  }
  if (mFastTransformRefNew) {
    mFastTransformRef = std::move(mFastTransformRefNew);
  }
  if (mFastTransformHelperNew) {
    mFastTransformHelper = std::move(mFastTransformHelperNew);
  }
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
    for (unsigned int itr = 0; itr < ptrs.nOutputTracksTPCO2; itr++) {
      auto& trc = tpcTracks[itr];
      if (trackTune.useTPCInnerCorr) {
        trc.updateParams(trackTune.tpcParInner);
      }
      if (trackTune.tpcCovInnerType != TrackTunePar::AddCovType::Disable) {
        trc.updateCov(trackTune.tpcCovInner, trackTune.tpcCovInnerType == TrackTunePar::AddCovType::WithCorrelations);
      }
      if (trackTune.useTPCOuterCorr) {
        trc.getParamOut().updateParams(trackTune.tpcParOuter);
      }
      if (trackTune.tpcCovOuterType != TrackTunePar::AddCovType::Disable) {
        trc.getParamOut().updateCov(trackTune.tpcCovOuter, trackTune.tpcCovOuterType == TrackTunePar::AddCovType::WithCorrelations);
      }
    }
  }
}

} // namespace o2::gpu
