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

#include "GPUWorkflow/O2GPUDPLDisplay.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TRDBase/GeometryFlat.h"
#include "TRDBase/Geometry.h"
#include "TOFBase/Geo.h"
#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/Propagator.h"
#include "GPUO2InterfaceDisplay.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCFastTransform.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "CorrectionMapsHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "GPUWorkflowHelper/GPUWorkflowHelper.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::globaltracking;
using namespace o2::base;
using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2::trd;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"enable-mc", o2::framework::VariantType::Bool, false, {"enable visualization of MC data"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable visualization of MC data"}}, // for compatibility, overrides enable-mc
    {"display-clusters", VariantType::String, "ITS,TPC,TRD,TOF", {"comma-separated list of clusters to display"}},
    {"display-tracks", VariantType::String, "TPC,ITS,ITS-TPC,TPC-TRD,ITS-TPC-TRD,TPC-TOF,ITS-TPC-TOF", {"comma-separated list of tracks to display"}},
    {"read-from-files", o2::framework::VariantType::Bool, false, {"comma-separated list of tracks to display"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"Disable root input overriding read-from-files"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

void O2GPUDPLDisplaySpec::init(InitContext& ic)
{
  GRPGeomHelper::instance().setRequest(mGGR);
  mConfig.reset(new GPUO2InterfaceConfiguration);
  mConfig->configGRP.solenoidBz = 0;
  mConfParam.reset(new GPUSettingsO2(mConfig->ReadConfigurableParam()));

  mFastTransformHelper.reset(new o2::tpc::CorrectionMapsLoader());
  mFastTransform = std::move(TPCFastTransformHelperO2::instance()->create(0));
  mFastTransformRef = std::move(TPCFastTransformHelperO2::instance()->create(0));
  mFastTransformHelper->setCorrMap(mFastTransform.get());
  mFastTransformHelper->setCorrMapRef(mFastTransformRef.get());
  mConfig->configCalib.fastTransform = mFastTransformHelper->getCorrMap();
  mConfig->configCalib.fastTransformRef = mFastTransformHelper->getCorrMapRef();
  mConfig->configCalib.fastTransformHelper = mFastTransformHelper.get();

  mTrdGeo.reset(new o2::trd::GeometryFlat());
  mConfig->configCalib.trdGeometry = mTrdGeo.get();

  mITSDict = std::make_unique<o2::itsmft::TopologyDictionary>();
  mConfig->configCalib.itsPatternDict = mITSDict.get();

  mConfig->configProcessing.runMC = mUseMC;

  mTFSettings.reset(new o2::gpu::GPUSettingsTF);
  mTFSettings->hasSimStartOrbit = 1;
  auto& hbfu = o2::raw::HBFUtils::Instance();
  mTFSettings->simStartOrbit = hbfu.getFirstIRofTF(o2::InteractionRecord(0, hbfu.orbitFirstSampled)).orbit;
  mAutoContinuousMaxTimeBin = mConfig->configGRP.continuousMaxTimeBin == -1;

  mDisplay.reset(new GPUO2InterfaceDisplay(mConfig.get()));
}

void O2GPUDPLDisplaySpec::run(ProcessingContext& pc)
{
  GRPGeomHelper::instance().checkUpdates(pc);
  const auto grp = o2::parameters::GRPObject::loadFrom();
  if (mConfParam->tpcTriggeredMode ^ !grp->isDetContinuousReadOut(o2::detectors::DetID::TPC)) {
    LOG(fatal) << "configKeyValue tpcTriggeredMode does not match GRP isDetContinuousReadOut(TPC) setting";
  }
  if (mDisplayShutDown) {
    return;
  }

  mTFSettings->tfStartOrbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
  mTFSettings->hasTfStartOrbit = 1;
  mTFSettings->hasNHBFPerTF = 1;
  mTFSettings->nHBFPerTF = GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF();
  mTFSettings->hasRunStartOrbit = 0;

  if (mGRPGeomUpdated) {
    mGRPGeomUpdated = false;
    mConfig->configGRP.solenoidBz = 5.00668f * grp->getL3Current() / 30000.;
    if (mAutoContinuousMaxTimeBin) {
      mConfig->configGRP.continuousMaxTimeBin = (mTFSettings->nHBFPerTF * o2::constants::lhc::LHCMaxBunches + 2 * o2::tpc::constants::LHCBCPERTIMEBIN - 2) / o2::tpc::constants::LHCBCPERTIMEBIN;
    }
    mDisplay->UpdateGRP(&mConfig->configGRP);
    if (mGeometryCreated == 0) {
      auto gm = o2::trd::Geometry::instance();
      gm->createPadPlaneArray();
      gm->createClusterMatrixArray();
      mTrdGeo.reset(new o2::trd::GeometryFlat(*gm));
      mConfig->configCalib.trdGeometry = mTrdGeo.get();
      mGeometryCreated = true;
      mUpdateCalib = true;

      o2::tof::Geo::Init();
      o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G, o2::math_utils::TransformType::L2G, o2::math_utils::TransformType::T2L));
    }
  }
  if (mUpdateCalib) {
    mDisplay->UpdateCalib(&mConfig->configCalib);
  }

  if (mDisplayStarted == false) {
    if (mDisplay->startDisplay()) {
      throw std::runtime_error("Error starting event display");
    }
    mDisplayStarted = true;
  }

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  GPUTrackingInOutPointers ptrs;
  auto tmpContainer = GPUWorkflowHelper::fillIOPtr(ptrs, recoData, mUseMC, &(mConfig->configCalib), mClMask, mTrkMask, mTrkMask);

  ptrs.settingsTF = mTFSettings.get();

  if (mDisplay->show(&ptrs)) {
    mDisplay->endDisplay();
    mDisplayShutDown = true;
  }
}

void O2GPUDPLDisplaySpec::endOfStream(EndOfStreamContext& ec)
{
  if (mDisplayShutDown) {
    return;
  }
  mDisplay->endDisplay();
  mDisplayShutDown = true;
}

void O2GPUDPLDisplaySpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == o2::framework::ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    mConfig->configCalib.itsPatternDict = (const o2::itsmft::TopologyDictionary*)obj;
    mUpdateCalib = true;
    return;
  }
  if (GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    mGRPGeomUpdated = true;
    return;
  }
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;

  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  bool useMC = cfgc.options().get<bool>("enable-mc") && !cfgc.options().get<bool>("disable-mc");
  GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-tracks"));
  GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-clusters"));
  if (!srcTrk.any() && !srcCl.any()) {
    throw std::runtime_error("No input configured");
  }
  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTrk, useMC);
  dataRequest->requestClusters(srcCl, useMC);

  if (cfgc.options().get<bool>("read-from-files")) {
    InputHelper::addInputSpecs(cfgc, specs, srcCl, srcTrk, srcTrk, useMC);
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false, true, false, true, false, o2::base::GRPGeomRequest::Aligned, dataRequest->inputs, true);

  specs.emplace_back(DataProcessorSpec{
    "o2-gpu-display",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<O2GPUDPLDisplaySpec>(useMC, srcTrk, srcCl, dataRequest, ggRequest)}});

  return specs;
}
