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
  mTPCCorrMapsLoader.init(ic);
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
  mTPCCorrMapsLoader.extractCCDBInputs(pc);
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
  auto prop = o2::base::Propagator::Instance();

  mTPCTracksArray = recoData.getTPCTracks();
  mTPCTrackClusIdx = recoData.getTPCTracksClusterRefs();
  mTPCClusterIdxStruct = &recoData.inputsTPCclusters->clusterIndex;
  mTPCRefitterShMap = recoData.clusterShMapTPC;

  std::vector<o2::InteractionTimeRecord> intRecs;
  if (mUseMC) { // extract MC tracks
    const o2::steer::DigitizationContext* digCont = nullptr;
    if (!mcReader.initFromDigitContext("collisioncontext.root")) {
      throw std::invalid_argument("initialization of MCKinematicsReader failed");
    }
    digCont = mcReader.getDigitizationContext();
    intRecs = digCont->getEventRecords();
    mTPCTrkLabels = recoData.getTPCTracksMCLabels();
  }

  mTPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mTPCClusterIdxStruct, &mTPCCorrMapsLoader, prop->getNominalBz(), mTPCTrackClusIdx.data(), mTPCRefitterShMap.data(), nullptr, o2::base::Propagator::Instance());

  float vdriftTB = mTPCVDriftHelper.getVDriftObject().getVDrift() * o2::tpc::ParameterElectronics::Instance().ZbinWidth; // VDrift expressed in cm/TimeBin
  float tpcTBBias = mTPCVDriftHelper.getVDriftObject().getTimeOffset() / (8 * o2::constants::lhc::LHCBunchSpacingMUS);
  float RRef2 = mRRef * mRRef;
  std::vector<short> clSector, clRow;
  std::vector<float> clX, clY, clZ;

  for (size_t itr = 0; itr < mTPCTracksArray.size(); itr++) {
    auto tr = mTPCTracksArray[itr]; // create track copy
    if (tr.hasBothSidesClusters()) {
      continue;
    }

    //=========================================================================
    // create refitted copy
    auto trackRefit = [RRef2, itr, this](o2::track::TrackParCov& trc, float t) -> bool {
      float chi2Out = 0;
      int retVal = this->mTPCRefitter->RefitTrackAsTrackParCov(trc, this->mTPCTracksArray[itr].getClusterRef(), t, &chi2Out, false, true);
      if (retVal < 0) {
        LOGP(warn, "Refit failed ({}) with time={}: track#{}[{}]", retVal, t, counter, trc.asString());
        return false;
      }
      return true;
    };

    auto trackProp = [RRef2, &tr, itr, prop, this](o2::track::TrackParCov& trc) -> bool {
      if (!trc.rotate(tr.getAlpha())) {
        LOGP(warn, "Rotation to original track alpha {} failed, track#{}[{}]", tr.getAlpha(), counter, trc.asString());
        return false;
      }
      float curR2 = trc.getX() * trc.getX() + trc.getY() * trc.getY();
      if (curR2 > RRef2) { // try to propagate as close as possible to target radius
        float xtgt = 0;
        if (!trc.getXatLabR(this->mRRef, xtgt, prop->getNominalBz(), o2::track::DirInward)) {
          xtgt = 0;
          return false;
        }
        float phi = o2::math_utils::sector2Angle(o2::math_utils::angle2Sector(tr.getPhiPos()));
        if (!prop->PropagateToXBxByBz(trc, xtgt) || !trc.rotate(phi)) {
          LOGP(warn, "Propagation to X={} or rotation to {} failed, track#{}[{}]", xtgt, phi, counter, trc.asString());
          return false;
        }
      }
      return true;
    };

    auto prepClus = [this, &tr, &clSector, &clRow, &clX, &clY, &clZ](float t) { // extract cluster info
      clSector.clear();
      clRow.clear();
      clX.clear();
      clY.clear();
      clZ.clear();
      int count = tr.getNClusters();
      const auto* corrMap = this->mTPCCorrMapsLoader.getCorrMap();
      const o2::tpc::ClusterNative* cl = nullptr;
      for (int ic = count; ic--;) {
        uint8_t sector, row;
        cl = &tr.getCluster(this->mTPCTrackClusIdx, ic, *this->mTPCClusterIdxStruct, sector, row);
        clSector.push_back(sector);
        clRow.push_back(row);
        float x, y, z;
        corrMap->Transform(sector, row, cl->getPad(), cl->getTime(), x, y, z, t); // nominal time of the track
        clX.push_back(x);
        clY.push_back(y);
        clZ.push_back(z);
      }
    };

    //=========================================================================

    auto trf = tr.getOuterParam(); // we refit inward original track
    if (!trackRefit(trf, tr.getTime0()) || !trackProp(trf)) {
      continue;
    }

    // propagate original track
    if (!trackProp(tr)) {
      continue;
    }

    prepClus(tr.getTime0()); // original clusters
    counter++;
    // store results
    (*mDBGOut) << "tpcIni"
               << "counter=" << counter
               << "iniTrack=" << tr
               << "iniTrackRef=" << trf
               << "time=" << tr.getTime0()
               << "clSector=" << clSector
               << "clRow=" << clRow
               << "clX=" << clX
               << "clY=" << clY
               << "clZ=" << clZ
               << "\n";

    float dz = 0;

    while (mUseMC) { // impose MC time in TPC timebin and refit inward after resetted covariance
      // extract MC truth
      const o2::MCTrack* mcTrack = nullptr;
      auto lbl = mTPCTrkLabels[itr];
      if (!lbl.isValid() || !(mcTrack = mcReader.getTrack(lbl))) {
        break;
      }
      long bc = intRecs[lbl.getEventID()].toLong(); // bunch crossing of the interaction
      float bcTB = bc / 8. + tpcTBBias;             // the same in TPC timebins, accounting for the TPC time bias
      // create MC truth track in O2 format
      std::array<float, 3> xyz{(float)mcTrack->GetStartVertexCoordinatesX(), (float)mcTrack->GetStartVertexCoordinatesY(), (float)mcTrack->GetStartVertexCoordinatesZ()},
        pxyz{(float)mcTrack->GetStartVertexMomentumX(), (float)mcTrack->GetStartVertexMomentumY(), (float)mcTrack->GetStartVertexMomentumZ()};
      TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(mcTrack->GetPdgCode());
      if (!pPDG) {
        break;
      }
      o2::track::TrackPar mctrO2(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);
      //
      // propagate it to the alpha/X of the reconstructed track
      if (!mctrO2.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(mctrO2, tr.getX())) {
        break;
      }
      // now create a properly refitted track with correct time and distortions correction
      {
        auto trfm = tr.getOuterParam(); // we refit inward
        // impose MC time in TPC timebin and refit inward after resetted covariance
        if (!trackRefit(trfm, bcTB) || !trfm.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(trfm, tr.getX())) {
          LOGP(warn, "Failed to propagate MC-time refitted track#{} [{}] to X/alpha of original track [{}]", counter, trfm.asString(), tr.asString());
          break;
        }
        // estimate Z shift in case of no-distortions
        dz = (tr.getTime0() - bcTB) * vdriftTB;
        if (tr.hasCSideClustersOnly()) {
          dz = -dz;
        }
        //
        prepClus(bcTB); // clusters for MC time
        (*mDBGOut) << "tpcMC"
                   << "counter=" << counter
                   << "movTrackRef=" << trfm
                   << "mcTrack=" << mctrO2
                   << "imposedTB=" << bcTB
                   << "dz=" << dz
                   << "clX=" << clX
                   << "clY=" << clY
                   << "clZ=" << clZ
                   << "\n";
      }
      break;
    }
    // refit and store the same track for a few compatible times
    float tmin = tr.getTime0() - tr.getDeltaTBwd();
    float tmax = tr.getTime0() + tr.getDeltaTFwd();
    int n = 5;
    for (int it = 0; it <= n; it++) {
      float tb = tmin + it * (tmax - tmin) / n;
      auto trfm = tr.getOuterParam(); // we refit inward
      // impose time in TPC timebin and refit inward after resetted covariance
      if (!trackRefit(trfm, tb) || !trfm.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(trfm, tr.getX())) {
        LOGP(warn, "Failed to propagate time={} refitted track#{} [{}] to X/alpha of original track [{}]", tb, counter, trfm.asString(), tr.asString());
        continue;
      }
      // estimate Z shift in case of no-distortions
      dz = (tr.getTime0() - tb) * vdriftTB;
      if (tr.hasCSideClustersOnly()) {
        dz = -dz;
      }
      //
      prepClus(tb); // clusters for MC time
      (*mDBGOut) << "tpcMov"
                 << "counter=" << counter
                 << "copy=" << it
                 << "maxCopy=" << n
                 << "movTrackRef=" << trfm
                 << "imposedTB=" << tb
                 << "dz=" << dz
                 << "clX=" << clX
                 << "clY=" << clY
                 << "clZ=" << clZ
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
  Options opts{{"target-radius", VariantType::Float, 70.f, {"Try to propagate to this radius"}}};
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
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs, opts, srcTracks[GTrackID::CTP] || srcClusters[GTrackID::CTP]);

  return DataProcessorSpec{
    "tpc-track-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCTrackStudySpec>(dataRequest, ggRequest, srcTracks, useMC)},
    opts};
}

} // namespace o2::trackstudy
