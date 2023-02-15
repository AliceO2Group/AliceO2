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

#include <vector>
#include <TStopwatch.h>
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCUtils.h"
#include "CommonUtils/NameConf.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "GlobalTrackingStudy/TPCTrackStudy.h"
#include "GPUO2InterfaceRefit.h"
#include "TPCBase/ParameterElectronics.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "Steer/MCKinematicsReader.h"

namespace o2::trackstudy
{

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;
using TBracket = o2::math_utils::Bracketf_t;

using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class TPCTrackStudySpec : public Task
{
 public:
  TPCTrackStudySpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC)
  {
  }
  ~TPCTrackStudySpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  o2::tpc::CorrectionMapsLoader mTPCCorrMapsLoader{};
  bool mUseMC{false}; ///< MC flag
  float mRRef = 0.;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  float mITSROFrameLengthMUS = 0.;
  GTrackID::mask_t mTracksSrc{};
  o2::steer::MCKinematicsReader mcReader; // reader of MC information
  //
  // TPC data
  gsl::span<const o2::tpc::TPCClRefElem> mTPCTrackClusIdx;            ///< input TPC track cluster indices span
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArray;                 ///< input TPC tracks span
  gsl::span<const unsigned char> mTPCRefitterShMap;                   ///< externally set TPC clusters sharing map
  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices
  gsl::span<const o2::MCCompLabel> mTPCTrkLabels;                     ///< input TPC Track MC labels
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter;         ///< TPC refitter used for TPC tracks refit during the reconstruction
};

void TPCTrackStudySpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mRRef = ic.options().get<float>("target-radius");
  if (mRRef < 0.) {
    mRRef = 0.;
  }
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>("tpc-trackStudy.root", "recreate");
}

void TPCTrackStudySpec::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void TPCTrackStudySpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  o2::tpc::CorrectionMapsLoader::extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // none at the moment
  }
  // we may have other params which need to be queried regularly
  bool updateMaps = false;
  if (mTPCCorrMapsLoader.isUpdated()) {
    mTPCCorrMapsLoader.acknowledgeUpdate();
    updateMaps = true;
  }
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
         mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
         mTPCVDriftHelper.getSourceName());
    mTPCVDriftHelper.acknowledgeUpdate();
    updateMaps = true;
  }
  if (updateMaps) {
    mTPCCorrMapsLoader.updateVDrift(mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getVDriftObject().getTimeOffset());
  }
}

void TPCTrackStudySpec::process(o2::globaltracking::RecoContainer& recoData)
{
  static long counter = -1;
  counter++;
  auto prop = o2::base::Propagator::Instance();

  if (mUseMC) { // extract MC tracks
    const o2::steer::DigitizationContext* digCont = nullptr;
    if (!mcReader.initFromDigitContext("collisioncontext.root")) {
      throw std::invalid_argument("initialization of MCKinematicsReader failed");
    }
    digCont = mcReader.getDigitizationContext();
    const auto& intRecs = digCont->getEventRecords();

    mTPCTracksArray = recoData.getTPCTracks();
    mTPCTrackClusIdx = recoData.getTPCTracksClusterRefs();
    mTPCClusterIdxStruct = &recoData.inputsTPCclusters->clusterIndex;
    mTPCRefitterShMap = recoData.clusterShMapTPC;
    mTPCTrkLabels = recoData.getTPCTracksMCLabels();

    mTPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mTPCClusterIdxStruct, &mTPCCorrMapsLoader, prop->getNominalBz(), mTPCTrackClusIdx.data(), mTPCRefitterShMap.data(), nullptr, o2::base::Propagator::Instance());

    float vdriftTB = mTPCVDriftHelper.getVDriftObject().getVDrift() * o2::tpc::ParameterElectronics::Instance().ZbinWidth; // VDrift expressed in cm/TimeBin
    float tpcTBBias = mTPCVDriftHelper.getVDriftObject().getTimeOffset() / (8 * o2::constants::lhc::LHCBunchSpacingMUS);
    float RRef2 = mRRef * mRRef;
    const o2::MCTrack* mcTrack = nullptr;
    std::vector<short> clSector, clRow;
    std::vector<float> clIniX, clIniY, clIniZ, clMovX, clMovY, clMovZ;
    for (size_t itr = 0; itr < mTPCTracksArray.size(); itr++) {
      auto tr = mTPCTracksArray[itr]; // create track copy

      // create refitted copy
      auto trf = tr.getOuterParam(); // we refit inward
      float chi2Out = 0;
      // impose MC time in TPC timebin and refit inward after resetted covariance
      int retVal = mTPCRefitter->RefitTrackAsTrackParCov(trf, mTPCTracksArray[itr].getClusterRef(), tr.getTime0(), &chi2Out, false, true);
      if (retVal < 0) {
        LOGP(warn, "Refit failed ({}) with originaltime0: {} : track#{}[{}]", retVal, tr.getTime0(), counter, ((const o2::track::TrackPar&)tr.getOuterParam()).asString());
        continue;
      }
      // propagate original track
      if (!tr.rotate(tr.getPhi())) {
        continue;
      }
      float curR2 = tr.getX() * tr.getX() + tr.getY() * tr.getY();
      if (curR2 > RRef2) { // try to propagate as close as possible to target radius
        float xtgt = 0;
        if (!tr.getXatLabR(mRRef, xtgt, prop->getNominalBz(), o2::track::DirInward)) {
          xtgt = 0;
        }
        prop->PropagateToXBxByBz(tr, xtgt); // propagation will not necessarilly converge, but we don't care
        if (!tr.rotate(tr.getPhi())) {
          continue;
        }
      }
      // propagate original/refitted track
      if (!trf.rotate(tr.getPhi())) {
        continue;
      }
      curR2 = trf.getX() * trf.getX() + trf.getY() * trf.getY();
      if (curR2 > RRef2) { // try to propagate as close as possible to target radius
        float xtgt = 0;
        if (!trf.getXatLabR(mRRef, xtgt, prop->getNominalBz(), o2::track::DirInward)) {
          xtgt = 0;
        }
        prop->PropagateToXBxByBz(trf, xtgt); // propagation will not necessarilly converge, but we don't care
        // propagate to the same alpha/X as the original track
        if (!trf.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(trf, tr.getX())) {
          continue;
        }
      }

      // extract MC truth
      auto lbl = mTPCTrkLabels[itr];
      if (!lbl.isValid() || !(mcTrack = mcReader.getTrack(lbl))) {
        continue;
      }
      long bc = intRecs[lbl.getEventID()].toLong(); // bunch crossing of the interaction
      float bcTB = bc / 8. + tpcTBBias;             // the same in TPC timebins, accounting for the TPC time bias
      // create MC truth track in O2 format
      std::array<float, 3> xyz{(float)mcTrack->GetStartVertexCoordinatesX(), (float)mcTrack->GetStartVertexCoordinatesY(), (float)mcTrack->GetStartVertexCoordinatesZ()},
        pxyz{(float)mcTrack->GetStartVertexMomentumX(), (float)mcTrack->GetStartVertexMomentumY(), (float)mcTrack->GetStartVertexMomentumZ()};
      TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(mcTrack->GetPdgCode());
      if (!pPDG) {
        continue;
      }
      o2::track::TrackPar mctrO2(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);
      //
      // propagate it to the alpha/X of the reconstructed track
      if (!mctrO2.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(mctrO2, tr.getX())) {
        continue;
      }
      //
      // now create a properly refitted track with correct time and distortions correction
      auto trackRefit = tr.getOuterParam(); // we refit inward
      chi2Out = 0;
      // impose MC time in TPC timebin and refit inward after resetted covariance
      retVal = mTPCRefitter->RefitTrackAsTrackParCov(trackRefit, mTPCTracksArray[itr].getClusterRef(), bcTB, &chi2Out, false, true);
      if (retVal < 0) {
        LOGP(warn, "Refit failed for #{} ({}), imposed time0: {} conventional time0: {} [{}]", counter, retVal, bcTB, tr.getTime0(), ((o2::track::TrackPar&)trackRefit).asString());
        continue;
      }
      // propagate the refitted track to the same X/alpha as original track
      if (!trackRefit.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(trackRefit, tr.getX())) {
        LOGP(warn, "Failed to propagate refitted track#{} [{}] to X/alpha of original track [{}]", counter, trackRefit.asString(), tr.asString());
        continue;
      }
      // estimate Z shift in case of no-distortions
      float dz = (tr.getTime0() - bcTB) * vdriftTB;
      if (tr.hasCSideClustersOnly()) {
        dz = -dz;
      } else if (tr.hasBothSidesClusters()) {
        dz = 0; // CE crossing tracks should not be shifted
      }
      // extract cluster info
      clSector.clear();
      clRow.clear();
      clIniX.clear();
      clIniY.clear();
      clIniZ.clear();
      clMovX.clear();
      clMovY.clear();
      clMovZ.clear();
      int count = tr.getNClusters();
      const auto* corrMap = mTPCCorrMapsLoader.getCorrMap();
      const o2::tpc::ClusterNative* cl = nullptr;
      for (int ic = count; ic--;) {
        uint8_t sector, row;
        cl = &tr.getCluster(mTPCTrackClusIdx, ic, *mTPCClusterIdxStruct, sector, row);
        clSector.push_back(sector);
        clRow.push_back(row);
        float x, y, z;
        corrMap->Transform(sector, row, cl->getPad(), cl->getTime(), x, y, z, tr.getTime0()); // nominal time of the track
        clIniX.push_back(x);
        clIniY.push_back(y);
        clIniZ.push_back(z);
        corrMap->Transform(sector, row, cl->getPad(), cl->getTime(), x, y, z, bcTB); // shifted time of the track
        clMovX.push_back(x);
        clMovY.push_back(y);
        clMovZ.push_back(z);
      }

      // store results
      (*mDBGOut) << "tpc"
                 << "iniTrack=" << tr
                 << "iniTrackRef=" << trf
                 << "movTrackRef=" << trackRefit
                 << "mcTrack=" << mctrO2
                 << "imposedTB=" << bcTB << "dz=" << dz
                 << "clSector=" << clSector
                 << "clRow=" << clRow
                 << "clIniX=" << clIniX
                 << "clIniY=" << clIniY
                 << "clIniZ=" << clIniZ
                 << "clMovX=" << clMovX
                 << "clMovY=" << clMovY
                 << "clMovZ=" << clMovZ
                 << "\n";
    }
  }
}

void TPCTrackStudySpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
}

void TPCTrackStudySpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (mTPCCorrMapsLoader.accountCCDBInputs(matcher, obj)) {
    return;
  }
}

DataProcessorSpec getTPCTrackStudySpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(srcTracks, useMC);
  dataRequest->requestClusters(srcClusters, useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs);

  return DataProcessorSpec{
    "tpc-track-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCTrackStudySpec>(dataRequest, ggRequest, srcTracks, useMC)},
    Options{{"target-radius", VariantType::Float, 70.f, {"Try to propagate to this radius"}}}};
}

} // namespace o2::trackstudy
