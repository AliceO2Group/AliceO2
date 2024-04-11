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
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCWorkflow/TPCRefitter.h"
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

class TPCRefitterSpec final : public Task
{
 public:
  TPCRefitterSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts, GTrackID::mask_t src, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC)
  {
    mTPCCorrMapsLoader.setLumiScaleType(sclOpts.lumiType);
    mTPCCorrMapsLoader.setLumiScaleMode(sclOpts.lumiMode);
  }
  ~TPCRefitterSpec() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process(o2::globaltracking::RecoContainer& recoData);
  bool getDCAs(const o2::track::TrackPar& track, float& dcar, float& dcaz);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  o2::tpc::CorrectionMapsLoader mTPCCorrMapsLoader{};
  bool mUseMC{false}; ///< MC flag
  bool mUseGPUModel{false};
  float mXRef = 83.;
  float mDCAMinPt = 1.;
  int mTFStart = 0;
  int mTFEnd = 999999999;
  int mTFCount = -1;
  int mDCAMinNCl = 80;
  bool mUseR = false;
  bool mEnableDCA = false;
  bool mWriteTrackClusters = false;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOutCl;
  float mITSROFrameLengthMUS = 0.;
  GTrackID::mask_t mTracksSrc{};
  o2::steer::MCKinematicsReader mcReader; // reader of MC information
  //
  // TPC data
  gsl::span<const o2::tpc::TPCClRefElem> mTPCTrackClusIdx;            ///< input TPC track cluster indices span
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArray;                 ///< input TPC tracks span
  gsl::span<const unsigned char> mTPCRefitterShMap;                   ///< externally set TPC clusters sharing map
  gsl::span<const unsigned int> mTPCRefitterOccMap;                   ///< externally set TPC clusters occupancy map
  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices
  gsl::span<const o2::MCCompLabel> mTPCTrkLabels;                     ///< input TPC Track MC labels
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter;         ///< TPC refitter used for TPC tracks refit during the reconstruction
};

void TPCRefitterSpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mXRef = ic.options().get<float>("target-x");
  mUseR = ic.options().get<bool>("use-r-as-x");
  mEnableDCA = ic.options().get<bool>("enable-dcas");
  mUseGPUModel = ic.options().get<bool>("use-gpu-fitter");
  mTFStart = ic.options().get<int>("tf-start");
  mTFEnd = ic.options().get<int>("tf-end");
  mDCAMinPt = ic.options().get<float>("dcaMinPt");
  mDCAMinNCl = ic.options().get<float>("dcaMinNCl");
  if (mXRef < 0.) {
    mXRef = 0.;
  }
  mTPCCorrMapsLoader.init(ic);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>("tpctracks-refitted.root", "recreate");
  mWriteTrackClusters = ic.options().get<bool>("dump-clusters");
  if (ic.options().get<bool>("dump-clusters")) {
    mDBGOutCl = std::make_unique<o2::utils::TreeStreamRedirector>("tpc-trackStudy-cl.root", "recreate");
  }
}

void TPCRefitterSpec::run(ProcessingContext& pc)
{
  mTFCount++;
  if (mTFCount < mTFStart || mTFCount > mTFEnd) {
    LOGP(info, "Skipping TF {}", mTFCount);
    return;
  }

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
  updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);

  if (mTFCount > mTFEnd) {
    LOGP(info, "Stopping processing after TF {}", mTFCount);
    pc.services().get<o2::framework::ControlService>().endOfStream();
    return;
  }
}

void TPCRefitterSpec::updateTimeDependentParams(ProcessingContext& pc)
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

void TPCRefitterSpec::process(o2::globaltracking::RecoContainer& recoData)
{
  static long counter = -1;
  auto prop = o2::base::Propagator::Instance();

  mTPCTracksArray = recoData.getTPCTracks();
  mTPCTrackClusIdx = recoData.getTPCTracksClusterRefs();
  mTPCClusterIdxStruct = &recoData.inputsTPCclusters->clusterIndex;
  mTPCRefitterShMap = recoData.clusterShMapTPC;
  mTPCRefitterOccMap = recoData.occupancyMapTPC;

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

  mTPCRefitter = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mTPCClusterIdxStruct, &mTPCCorrMapsLoader, prop->getNominalBz(), mTPCTrackClusIdx.data(), 0, mTPCRefitterShMap.data(), mTPCRefitterOccMap.data(), mTPCRefitterOccMap.size(), nullptr, prop);

  float vdriftTB = mTPCVDriftHelper.getVDriftObject().getVDrift() * o2::tpc::ParameterElectronics::Instance().ZbinWidth; // VDrift expressed in cm/TimeBin
  float tpcTBBias = mTPCVDriftHelper.getVDriftObject().getTimeOffset() / (8 * o2::constants::lhc::LHCBunchSpacingMUS);
  std::vector<short> clSector, clRow;
  std::vector<float> clX, clY, clZ, clXI, clYI, clZI; // *I are the uncorrected cluster positions
  float dcar, dcaz, dcarRef, dcazRef;

  auto dumpClusters = [this] {
    static int tf = 0;
    const auto* corrMap = this->mTPCCorrMapsLoader.getCorrMap();
    for (int sector = 0; sector < 36; sector++) {
      float alp = ((sector % 18) * 20 + 10) * TMath::DegToRad();
      float sn = TMath::Sin(alp), cs = TMath::Cos(alp);
      for (int row = 0; row < 152; row++) {
        for (int ic = 0; ic < this->mTPCClusterIdxStruct->nClusters[sector][row]; ic++) {
          const auto cl = this->mTPCClusterIdxStruct->clusters[sector][row][ic];
          float x, y, z, xG, yG;
          corrMap->TransformIdeal(sector, row, cl.getPad(), cl.getTime(), x, y, z, 0);
          o2::math_utils::detail::rotateZ(x, y, xG, yG, sn, cs);
          LOGP(debug, "tf:{} s:{} r:{} p:{} t:{} qm:{} qt:{} f:{} x:{} y:{} z:{}", tf, sector, row, cl.getPad(), cl.getTime(), cl.getQmax(), cl.getQtot(), cl.getFlags(), x, y, z);
          (*mDBGOutCl) << "tpccl"
                       << "tf=" << tf << "sect=" << sector << "row=" << row << "pad=" << cl.getPad() << "time=" << cl.getTime() << "qmax=" << cl.getQmax() << "qtot=" << cl.getQtot()
                       << "sigT=" << cl.getSigmaTime() << "sigP=" << cl.getSigmaPad()
                       << "flags=" << cl.getFlags()
                       << "x=" << x << "y=" << y << "z=" << z << "xg=" << xG << "yg=" << yG
                       << "\n";
        }
      }
    }
    tf++;
  };

  if (mDBGOutCl) {
    dumpClusters();
  }

  for (size_t itr = 0; itr < mTPCTracksArray.size(); itr++) {
    auto tr = mTPCTracksArray[itr]; // create track copy
    if (tr.hasBothSidesClusters()) {
      continue;
    }

    //=========================================================================
    // create refitted copy
    auto trackRefit = [itr, this](o2::track::TrackParCov& trc, float t, float chi2refit) -> bool {
      int retVal = mUseGPUModel ? this->mTPCRefitter->RefitTrackAsGPU(trc, this->mTPCTracksArray[itr].getClusterRef(), t, &chi2refit, false, true)
                                : this->mTPCRefitter->RefitTrackAsTrackParCov(trc, this->mTPCTracksArray[itr].getClusterRef(), t, &chi2refit, false, true);
      if (retVal < 0) {
        LOGP(warn, "Refit failed ({}) with time={}: track#{}[{}]", retVal, t, counter, trc.asString());
        return false;
      }
      return true;
    };

    auto trackProp = [&tr, itr, prop, this](o2::track::TrackParCov& trc) -> bool {
      if (!trc.rotate(tr.getAlpha())) {
        LOGP(warn, "Rotation to original track alpha {} failed, track#{}[{}]", tr.getAlpha(), counter, trc.asString());
        return false;
      }
      float xtgt = this->mXRef;
      if (mUseR && !trc.getXatLabR(this->mXRef, xtgt, prop->getNominalBz(), o2::track::DirInward)) {
        xtgt = 0;
        return false;
      }
      if (!prop->PropagateToXBxByBz(trc, xtgt)) {
        LOGP(warn, "Propagation to X={} failed, track#{}[{}]", xtgt, counter, trc.asString());
        return false;
      }
      return true;
    };

    auto prepClus = [this, &tr, &clSector, &clRow, &clX, &clY, &clZ, &clXI, &clYI, &clZI](float t) { // extract cluster info
      clSector.clear();
      clRow.clear();
      clXI.clear();
      clYI.clear();
      clZI.clear();
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
        // ideal transformation without distortions
        corrMap->TransformIdeal(sector, row, cl->getPad(), cl->getTime(), x, y, z, t); // nominal time of the track
        clXI.push_back(x);
        clYI.push_back(y);
        clZI.push_back(z);

        // transformation without distortions
        mTPCCorrMapsLoader.Transform(sector, row, cl->getPad(), cl->getTime(), x, y, z, t); // nominal time of the track
        clX.push_back(x);
        clY.push_back(y);
        clZ.push_back(z);
      }
    };

    //=========================================================================

    auto trf = tr.getOuterParam(); // we refit inward original track
    float chi2refit = 0;
    if (!trackRefit(trf, tr.getTime0(), chi2refit) || !trackProp(trf)) {
      continue;
    }

    // propagate original track
    if (!trackProp(tr)) {
      continue;
    }

    if (mWriteTrackClusters) {
      prepClus(tr.getTime0()); // original clusters
    }

    if (mEnableDCA) {
      dcar = dcaz = dcarRef = dcazRef = 9999.f;
      if ((trf.getPt() > mDCAMinPt) && (tr.getNClusters() > mDCAMinNCl)) {
        getDCAs(trf, dcarRef, dcazRef);
        getDCAs(tr, dcar, dcaz);
      }
    }

    counter++;
    // store results
    (*mDBGOut) << "tpcIni"
               << "counter=" << counter
               << "iniTrack=" << tr
               << "iniTrackRef=" << trf
               << "time=" << tr.getTime0()
               << "chi2refit=" << chi2refit;

    if (mWriteTrackClusters) {
      (*mDBGOut) << "tpcIni"
                 << "clSector=" << clSector
                 << "clRow=" << clRow
                 << "clX=" << clX
                 << "clY=" << clY
                 << "clZ=" << clZ
                 << "clXI=" << clXI  // ideal (uncorrected) cluster positions
                 << "clYI=" << clYI  // ideal (uncorrected) cluster positions
                 << "clZI=" << clZI; // ideal (uncorrected) cluster positions
    }

    if (mEnableDCA) {
      (*mDBGOut) << "tpcIni"
                 << "dcar=" << dcar
                 << "dcaz=" << dcaz
                 << "dcarRef=" << dcarRef
                 << "dcazRef=" << dcazRef;
    }

    (*mDBGOut) << "tpcIni"
               << "\n";

    float dz = 0;

    if (mUseMC) { // impose MC time in TPC timebin and refit inward after resetted covariance
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
        float chi2refit = 0;
        if (!trackRefit(trfm, bcTB, chi2refit) || !trfm.rotate(tr.getAlpha()) || !prop->PropagateToXBxByBz(trfm, tr.getX())) {
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
                   << "chi2refit=" << chi2refit
                   << "dz=" << dz
                   << "clX=" << clX
                   << "clY=" << clY
                   << "clZ=" << clZ
                   << "\n";
      }
      break;
    }
  }
}

void TPCRefitterSpec::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
  mDBGOutCl.reset();
}

void TPCRefitterSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
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

bool TPCRefitterSpec::getDCAs(const o2::track::TrackPar& track, float& dcar, float& dcaz)
{
  auto propagator = o2::base::Propagator::Instance();
  o2::gpu::gpustd::array<float, 2> dca;
  const o2::math_utils::Point3D<float> refPoint{0, 0, 0};
  o2::track::TrackPar propTrack(track);
  const auto ok = propagator->propagateToDCABxByBz(refPoint, propTrack, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT, &dca);
  dcar = dca[0];
  dcaz = dca[1];
  if (!ok) {
    dcar = 9998.;
    dcaz = 9998.;
  }
  return ok;
}

DataProcessorSpec getTPCRefitterSpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts)
{
  std::vector<OutputSpec> outputs;
  Options opts{
    {"target-x", VariantType::Float, 83.f, {"Try to propagate to this radius"}},
    {"dump-clusters", VariantType::Bool, false, {"dump all clusters"}},
    {"write-track-clusters", VariantType::Bool, false, {"write clusters associated to the track, uncorrected and corrected positions"}},
    {"tf-start", VariantType::Int, 0, {"1st TF to process"}},
    {"tf-end", VariantType::Int, 999999999, {"last TF to process"}},
    {"use-gpu-fitter", VariantType::Bool, false, {"use GPU track model for refit instead of TrackParCov"}},
    {"use-r-as-x", VariantType::Bool, false, {"Use radius instead of target sector X"}},
    {"enable-dcas", VariantType::Bool, false, {"Propagate to DCA and add it to the tree"}},
    {"dcaMinPt", VariantType::Float, 1.f, {"Min pT of tracks propagated to DCA"}},
    {"dcaMinNCl", VariantType::Int, 80, {"Min number of clusters for tracks propagated to DCA"}},
  };
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
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs, opts, sclOpts);

  return DataProcessorSpec{
    "tpc-refitter",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCRefitterSpec>(dataRequest, ggRequest, sclOpts, srcTracks, useMC)},
    opts};
}

} // namespace o2::trackstudy
