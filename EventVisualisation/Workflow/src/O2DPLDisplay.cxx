// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EveWorkflow/O2DPLDisplay.h"
#include "EveWorkflow/FileProducer.h"
#include "EventVisualisationDataConverter/VisualisationTrack.h"
#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/GeometryFlat.h"
#include "TRDBase/Geometry.h"
#include "TOFBase/Geo.h"
#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/Propagator.h"
#include "GPUO2InterfaceDisplay.h"
#include "GPUO2InterfaceConfiguration.h"
#include "TPCFastTransform.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "GPUWorkflowHelper/GPUWorkflowHelper.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

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

  mITSDict = std::make_unique<o2::itsmft::TopologyDictionary>();
  mConfig->configCalib.itsPatternDict = mITSDict.get();

  mConfig->configProcessing.runMC = mUseMC;

  o2::tof::Geo::Init();

  o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G, o2::math_utils::TransformType::L2G, o2::math_utils::TransformType::T2L));

  //mDisplay.reset(new GPUO2InterfaceDisplay(mConfig.get()));
}

struct GPUWorkflowHelper::tmpDataContainer {
  std::vector<o2::BaseCluster<float>> ITSClustersArray;
  std::vector<int> tpcLinkITS, tpcLinkTRD, tpcLinkTOF;
  std::vector<const o2::track::TrackParCov*> globalTracks;
  std::vector<float> globalTrackTimes;
};

using GID = o2::dataformats::GlobalTrackID;
using PNT = std::array<float, 3>;

std::vector<PNT> getTrackPoints(const o2::track::TrackPar& trc, float minR, float maxR, float maxStep)

{
  // prepare space points from the track param
  std::vector<PNT> pnts;
  int nSteps = std::max(2, int((maxR - minR) / maxStep));
  const auto prop = o2::base::Propagator::Instance();
  float xMin = trc.getX(), xMax = maxR * maxR - trc.getY() * trc.getY();
  if (xMax > 0) {
    xMax = std::sqrt(xMax);
  }
  LOG(INFO) << "R: " << minR << " " << maxR << " || X: " << xMin << " " << xMax;
  float dx = (xMax - xMin) / nSteps;
  auto tp = trc;
  float dxmin = std::abs(xMin - tp.getX()), dxmax = std::abs(xMax - tp.getX());
  bool res = false;
  if (dxmin > dxmax) { //start from closest end
    std::swap(xMin, xMax);
    dx = -dx;
  }
  if (!prop->propagateTo(tp, xMin, false, 0.99, maxStep, o2::base::PropagatorF::MatCorrType::USEMatCorrNONE)) {
    return pnts;
  }
  auto xyz = tp.getXYZGlo();
  pnts.emplace_back(PNT{xyz.X(), xyz.Y(), xyz.Z()});
  for (int is = 0; is < nSteps; is++) {
    if (!prop->propagateTo(tp, tp.getX() + dx, false, 0.99, 999., o2::base::PropagatorF::MatCorrType::USEMatCorrNONE)) {
      return pnts;
    }
    xyz = tp.getXYZGlo();
    pnts.emplace_back(PNT{xyz.X(), xyz.Y(), xyz.Z()});
  }
  return pnts;
}

void O2GPUDPLDisplaySpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  GPUTrackingInOutPointers ptrs;
  auto cnt = GPUWorkflowHelper::fillIOPtr(ptrs, recoData, mUseMC, &(mConfig->configCalib), mClMask, mTrkMask, mTrkMask);

  std::cout << cnt->globalTracks.size() << std::endl;
  o2::event_visualisation::VisualisationEvent vEvent;

  for (int i = 0; i < cnt->globalTracks.size(); i++) {
    auto tr = cnt->globalTracks[i];
    std::cout << tr->getX() << std::endl;
    float rMax = 385;
    float rMin = std::sqrt(tr->getX() * tr->getX() + tr->getY() * tr->getY());
    auto pnts = getTrackPoints(*tr, rMin, rMax, 4.0);

    o2::event_visualisation::VisualisationTrack* vTrack = vEvent.addTrack({.charge = 0,
                                                                           .energy = 0.0,
                                                                           .ID = 0,
                                                                           .PID = 0,
                                                                           .mass = 0.0,
                                                                           .signedPT = 0.0,
                                                                           .startXYZ = {0, 0, 0},
                                                                           .endXYZ = {0, 0, 0},
                                                                           .pxpypz = {0, 0, 0},
                                                                           .parentID = 0,
                                                                           .phi = 0.0,
                                                                           .theta = 0.0,
                                                                           .helixCurvature = 0.0,
                                                                           .type = 0,
                                                                           .source = CosmicSource});
    float dz = 0.0;
    for (size_t ip = 0; ip < pnts.size(); ip++) {
      vTrack->addPolyPoint(pnts[ip][0], pnts[ip][1], pnts[ip][2] + dz);
    }
  }

  FileProducer producer("./jsons");
  vEvent.toFile(producer.newFileName());
}

void O2GPUDPLDisplaySpec::endOfStream(EndOfStreamContext& ec)
{
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

  specs.emplace_back(DataProcessorSpec{
    "o2-gpu-display",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<O2GPUDPLDisplaySpec>(useMC, srcTrk, srcCl, dataRequest)}});

  return std::move(specs);
}
