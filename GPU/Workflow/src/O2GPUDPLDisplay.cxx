// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "GPUWorkflow/O2GPUDPLDisplay.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/GeometryFlat.h"
#include "TRDBase/Geometry.h"
#include "DetectorsBase/Propagator.h"
#include "GPUO2InterfaceDisplay.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCFastTransform.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/RecoInputContainer.h"

using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::globaltracking;
using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2::trd;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"enable-mc", o2::framework::VariantType::Bool, false, {"enable visualization of MC data"}},
    {"display-clusters", VariantType::String, "TPC,TRD", {"comma-separated list of clusters to display"}},
    {"display-tracks", VariantType::String, "TPC", {"comma-separated list of tracks to display"}},
    {"read-from-files", o2::framework::VariantType::Bool, false, {"comma-separated list of tracks to display"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"Disable root input overriding read-from-files"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

void O2GPUDPLDisplaySpec::init(InitContext& ic)
{
  const auto grp = o2::parameters::GRPObject::loadFrom();
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  mConfig.reset(new GPUO2InterfaceConfiguration);
  mConfig->configGRP.solenoidBz = 5.00668f * grp->getL3Current() / 30000.;
  mConfig->configGRP.continuousMaxTimeBin = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC) ? -1 : 0; // Number of timebins in timeframe if continuous, 0 otherwise
  mConfig->ReadConfigurableParam();

  mFastTransform = std::move(TPCFastTransformHelperO2::instance()->create(0));
  mConfig->configCalib.fastTransform = mFastTransform.get();

  auto gm = o2::trd::Geometry::instance();
  gm->createPadPlaneArray();
  gm->createClusterMatrixArray();
  mTrdGeo.reset(new o2::trd::GeometryFlat(*gm));
  mConfig->configCalib.trdGeometry = mTrdGeo.get();

  mDisplay.reset(new GPUO2InterfaceDisplay(mConfig.get()));
}

void O2GPUDPLDisplaySpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  static bool first = false;
  if (first == false) {
    if (mDisplay->startDisplay()) {
      throw std::runtime_error("Error starting event display");
    }
  }

  GPUTrackingInOutPointers ptrs;
  if (mClMask[GlobalTrackID::TPC]) {
    recoData.addTPCClusters(pc, false);
  }
  if (mTrkMask[GlobalTrackID::TPC]) {
    recoData.addTPCTracks(pc, mUseMC);
  }
  if (mClMask[GlobalTrackID::TRD]) {
    recoData.addTRDTracklets(pc);
  }
  if (mClMask[GlobalTrackID::TPC]) {
    ptrs.clustersNative = &recoData.inputsTPCclusters->clusterIndex;
  }
  if (mTrkMask[GlobalTrackID::TPC]) {
    const auto& tpcTracks = recoData.getTPCTracks<o2::tpc::TrackTPC>();
    const auto& tpcClusRefs = recoData.getTPCTracksClusterRefs();
    ptrs.outputTracksTPCO2 = tpcTracks.data();
    ptrs.nOutputTracksTPCO2 = tpcTracks.size();
    ptrs.outputClusRefsTPCO2 = tpcClusRefs.data();
    ptrs.nOutputClusRefsTPCO2 = tpcClusRefs.size();
  }
  if (mClMask[GlobalTrackID::TRD]) {
    o2::trd::getRecoInputContainer(pc, &ptrs, &recoData);
  }
  if (mUseMC) {
    const auto& tpcTracksMC = recoData.getTPCTracksMCLabels();
    ptrs.outputTracksTPCO2MC = tpcTracksMC.data();
  }

  mDisplay->show(&ptrs);
}

void O2GPUDPLDisplaySpec::endOfStream(EndOfStreamContext& ec)
{
  mDisplay->endDisplay();
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;

  bool useMC = cfgc.options().get<bool>("enable-mc");
  GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-tracks"));
  GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask(cfgc.options().get<std::string>("display-clusters"));
  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTrk, useMC);
  dataRequest->requestClusters(srcCl, useMC);

  if (cfgc.options().get<bool>("read-from-files")) {
    InputHelper::addInputSpecs(cfgc, specs, srcCl, srcTrk, srcTrk, useMC);
  }

  specs.emplace_back(DataProcessorSpec{
    "o2-gpu-display",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<O2GPUDPLDisplaySpec>(useMC, srcTrk, srcCl, dataRequest)}});

  return std::move(specs);
}
