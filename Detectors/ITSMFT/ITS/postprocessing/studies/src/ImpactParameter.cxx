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

// Skeleton derived from RS's code in ITSOffStudy

#include "ITSStudies/ImpactParameter.h"
#include "ITSStudies/Helpers.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/Task.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsVertexing/PVertexer.h"
#include "DetectorsBase/Propagator.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Framework/DeviceSpec.h"
#include <TH1F.h>
#include <TH2F.h>
#include <TMath.h>
#include "TROOT.h"
#include "TGeoGlobalMagField.h"

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using DetID = o2::detectors::DetID;
using PVertex = o2::dataformats::PrimaryVertex;
using GTrackID = o2::dataformats::GlobalTrackID;

class ImpactParameterStudy : public Task
{
 public:
  ImpactParameterStudy(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, mask_t src) : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src){};
  ~ImpactParameterStudy() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void process(o2::globaltracking::RecoContainer&);
  void endOfStream(EndOfStreamContext&) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  GTrackID::mask_t mTracksSrc{};
  o2::vertexing::PVertexer mVertexer;
  float mITSROFrameLengthMUS = 0.;
  float mITSROFBiasMUS = 0.;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  // output histograms
  std::unique_ptr<TH1F> mHisto_contributorsPV{};
  std::unique_ptr<TH1F> mHisto_trackType;
  std::unique_ptr<TH1F> mHisto_trackTypeRej;
  std::unique_ptr<TH1F> mHisto_trackTypeAcc;
  std::unique_ptr<TH2F> mHisto_X_PVrefitChi2minus1{};
  std::unique_ptr<TH2F> mHisto_Y_PVrefitChi2minus1{};
  std::unique_ptr<TH2F> mHisto_Z_PVrefitChi2minus1{};
  std::unique_ptr<TH1F> mHisto_X_DeltaPVrefitChi2minus1{};
  std::unique_ptr<TH1F> mHisto_Y_DeltaPVrefitChi2minus1{};
  std::unique_ptr<TH1F> mHisto_Z_DeltaPVrefitChi2minus1{};
  std::unique_ptr<TH2F> mHisto_ImpParXY{};
  std::unique_ptr<TH2F> mHisto_ImpParZ{};
  std::unique_ptr<TH1F> mHisto_ImpParXY_2{};
  std::unique_ptr<TH1F> mHisto_ImpParZ_2{};
  std::unique_ptr<TH2F> mHisto_Phi{};

  // output file
  const std::string mOutName{"its_ImpParameter.root"};

  // Data
  std::shared_ptr<DataRequest> mDataRequest;
  gsl::span<const PVertex> mPVertices;
};

void ImpactParameterStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);

  const auto logPtBinning = o2::its::studies::helpers::makeLogBinning(100, 0.1, 10);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mOutName.c_str(), "recreate");
  mHisto_trackType = std::make_unique<TH1F>("trackType", "# Track Type", 32, -0.5, 31.5);
  mHisto_trackType->SetDirectory(nullptr);
  mHisto_trackTypeRej = std::make_unique<TH1F>("trackTypeRej", "# Rejected Track Type", 32, -0.5, 31.5);
  mHisto_trackTypeRej->SetDirectory(nullptr);
  mHisto_trackTypeAcc = std::make_unique<TH1F>("trackTypeAcc", "# Filtered Track Type", 32, -0.5, 31.5);
  mHisto_trackTypeAcc->SetDirectory(nullptr);
  mHisto_contributorsPV = std::make_unique<TH1F>("nContrib_PVrefitNotDoable", "# Contributors per PV", 100, 0, 100);
  mHisto_contributorsPV->SetDirectory(nullptr);
  mHisto_X_PVrefitChi2minus1 = std::make_unique<TH2F>("h2_X_PvVsPVrefit", "#X PV vs PV_{-1}, #mum", 100, -10, 10, 100, -10, 10);
  mHisto_X_PVrefitChi2minus1->SetDirectory(nullptr);
  mHisto_Y_PVrefitChi2minus1 = std::make_unique<TH2F>("h2_Y_PvVsPVrefit", "#Y  PV vs PV_{-1}, #mum", 100, -10, 10, 100, -10, 10);
  mHisto_Y_PVrefitChi2minus1->SetDirectory(nullptr);
  mHisto_Z_PVrefitChi2minus1 = std::make_unique<TH2F>("h2_Z_PvVsPVrefit", "#Z PV vs PV_{-1}, #mum", 100, -10, 10, 100, -10, 10);
  mHisto_Z_PVrefitChi2minus1->SetDirectory(nullptr);
  mHisto_X_DeltaPVrefitChi2minus1 = std::make_unique<TH1F>("h_DeltaXPVrefit", "#DeltaX (PV-PV_{-1}), #mum", 300, -15, 15);
  mHisto_X_DeltaPVrefitChi2minus1->SetDirectory(nullptr);
  mHisto_Y_DeltaPVrefitChi2minus1 = std::make_unique<TH1F>("h_DeltaYPVrefit", "#DeltaY (PV-PV_{-1}), #mum", 300, -15, 15);
  mHisto_Y_DeltaPVrefitChi2minus1->SetDirectory(nullptr);
  mHisto_Z_DeltaPVrefitChi2minus1 = std::make_unique<TH1F>("h_DeltaZPVrefit", "#DeltaZ (PV-PV_{-1}), #mum", 300, -15, 15);
  mHisto_Z_DeltaPVrefitChi2minus1->SetDirectory(nullptr);
  mHisto_ImpParZ = std::make_unique<TH2F>("h1_ImpParZ", "Impact Parameter Z, #mum; p_{T};",  logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHisto_ImpParZ->SetDirectory(nullptr);
  mHisto_ImpParXY = std::make_unique<TH2F>("h1_ImpParXY", "Impact Parameter XY, #mum; p_{T};",   logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHisto_ImpParXY->SetDirectory(nullptr);
  mHisto_ImpParXY_2 = std::make_unique<TH1F>("h1_ImpParXY_2", "Impact Parameter XY, #mum; p_{T};", logPtBinning.size() - 1, logPtBinning.data());
  mHisto_ImpParXY_2->SetDirectory(nullptr);
  mHisto_ImpParZ_2 = std::make_unique<TH1F>("h1_ImpParZ_2", "Impact Parameter Z, #mum; p_{T};",  logPtBinning.size() - 1, logPtBinning.data());
  mHisto_ImpParZ_2->SetDirectory(nullptr);
  mHisto_Phi = std::make_unique<TH2F>("hdphi", "#Phi; #phi; ImpactParam d#phi(#mum);", 100, 0., 6.28, 100, -1000, 1000);
  mHisto_Phi->SetDirectory(nullptr);

}

void ImpactParameterStudy::run(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  //o2::base::Propagator::Instance()->setBz(5);
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void ImpactParameterStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  std::vector<o2::track::TrackParCov> vecPvContributorTrackParCov;
  std::vector<int64_t> vec_globID_contr = {};
  std::vector<o2::track::TrackParCov> trueVecPvContributorTrackParCov;
  std::vector<int64_t> trueVec_globID_contr = {};
  // o2::vertexing::PVertexer vertexer;
  float impParRPhi, impParZ;
  constexpr float toMicrometers = 10000.f; // Conversion from [cm] to [mum]
  bool keepAllTracksPVrefit = false;
  bool removeDiamondConstraint = false;
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto pvertices = recoData.getPrimaryVertices();
 

  int nv = vtxRefs.size() - 1;      // The last entry is for unassigned tracks, ignore them
  for (int iv = 0; iv < nv; iv++) { // Loop over PVs
    LOGP(info, "*** NEW VERTEX ***");
    const auto& vtref = vtxRefs[iv];
    const o2::dataformats::VertexBase& pv = pvertices[iv];
    int it = vtref.getFirstEntry(), itLim = it + vtref.getEntries();
    int i=0;

    for (; it < itLim; it++) {
      auto tvid = trackIndex[it];
      //LOGP(info,"ORIGIN: {}", tvid.getSourceName(tvid.getSource()));
      mHisto_trackType->Fill(tvid.getSource());
      if (!recoData.isTrackSourceLoaded(tvid.getSource())) {
        //LOGP(info,"SOURCE Rej: {}", tvid.getSourceName(tvid.getSource()));
        mHisto_trackTypeRej->Fill(tvid.getSource());
        continue;
      }
      //LOGP(info,"ORIGIN: {}  INDEX: {}", tvid.getSourceName(tvid.getSource()), trackIndex[it]);
      mHisto_trackTypeAcc->Fill(tvid.getSource());
      const o2::track::TrackParCov& trc = recoData.getTrackParam(tvid); // The actual track

      auto refs = recoData.getSingleDetectorRefs(tvid);
      if (!refs[GTrackID::ITS].isIndexSet()) { // might be an afterburner track
          LOGP(info, " ** AFTERBURN **");
          continue;
      }
      // Vectors for reconstructing the vertex
      vec_globID_contr.push_back(trackIndex[it]);
      vecPvContributorTrackParCov.push_back(trc);
      //Store vector with index and tracks ITS only   -- same order as the GLOBAL vectors
      int itsTrackID = refs[GTrackID::ITS].getIndex();
      const o2::track::TrackParCov& trcITS = recoData.getTrackParam(itsTrackID); // The actual ITS track
      trueVec_globID_contr.push_back(itsTrackID);
      trueVecPvContributorTrackParCov.push_back(trcITS);
      LOGP(info,"SOURCE: {}  indexGLOBAL:  {}  indexITS:  {}", tvid.getSourceName(tvid.getSource()), trackIndex[it], trueVec_globID_contr[i]);
      i++;
    } // end loop tracks
    // LOGP(info,"************  SIZE CONTR:   {}   ", vecPvContributorTrackParCov.size());
    LOGP(info,"************  SIZE INDEX GLOBAL:   {}   ", vec_globID_contr.size());
    LOGP(info,"************  SIZE INDEX ITS:   {}   ", trueVec_globID_contr.size());
    
    it = vtref.getFirstEntry();
    // Preparation PVertexer refit
    o2::conf::ConfigurableParam::updateFromString("pvertexer.useMeanVertexConstraint=false");
    bool PVrefit_doable = mVertexer.prepareVertexRefit(vecPvContributorTrackParCov, pv);
    if (!PVrefit_doable) {
      LOG(info) << "Not enough tracks accepted for the refit --> Skipping vertex";
    }
    else{
      mHisto_contributorsPV->Fill(vecPvContributorTrackParCov.size());
      if(vecPvContributorTrackParCov.size() < 6 ) continue;
      for (it=0; it < vec_globID_contr.size(); it++) {
        //vector of booleans to keep track of the track to be skipped
        std::vector<bool> vec_useTrk_PVrefit(vec_globID_contr.size(), true);
        auto tvid = vec_globID_contr[it];
        auto trackIterator = std::find(vec_globID_contr.begin(), vec_globID_contr.end(),vec_globID_contr[it]);
        if (trackIterator != vec_globID_contr.end()) {
          //LOGP(info,"************  Trackiterator:   {}  : {} ", it+1, vec_globID_contr[it]);
          o2::dataformats::VertexBase PVbase_recalculated;
          /// this track contributed to the PV fit: let's do the refit without it
          const int entry = std::distance(vec_globID_contr.begin(), trackIterator);
          if (!keepAllTracksPVrefit) {
              vec_useTrk_PVrefit[entry] = false; /// remove the track from the PV refitting
          }
          auto Pvtx_refitted = mVertexer.refitVertex(vec_useTrk_PVrefit, pv); // vertex refit
          // enable the dca recalculation for the current PV contributor, after removing it from the PV refit
          bool recalc_imppar = true;
          if (Pvtx_refitted.getChi2() < 0) {
            LOG(info) << "---> Refitted vertex has bad chi2 = " << Pvtx_refitted.getChi2();
            recalc_imppar = false;
          }
          vec_useTrk_PVrefit[entry] = true; /// restore the track for the next PV refitting
          if (recalc_imppar) {
            const double DeltaX = pv.getX() - Pvtx_refitted.getX();
            const double DeltaY = pv.getY() - Pvtx_refitted.getY();
            const double DeltaZ = pv.getZ() - Pvtx_refitted.getZ();
            mHisto_X_PVrefitChi2minus1->Fill(pv.getX(), Pvtx_refitted.getX());
            mHisto_Y_PVrefitChi2minus1->Fill(pv.getY(), Pvtx_refitted.getY());
            mHisto_Z_PVrefitChi2minus1->Fill(pv.getZ(), Pvtx_refitted.getZ());
            mHisto_X_DeltaPVrefitChi2minus1->Fill(DeltaX);
            mHisto_Y_DeltaPVrefitChi2minus1->Fill(DeltaY);
            mHisto_Z_DeltaPVrefitChi2minus1->Fill(DeltaZ);

            // fill the newly calculated PV
            PVbase_recalculated.setX(Pvtx_refitted.getX());
            PVbase_recalculated.setY(Pvtx_refitted.getY());
            PVbase_recalculated.setZ(Pvtx_refitted.getZ());
            PVbase_recalculated.setCov(Pvtx_refitted.getSigmaX2(), Pvtx_refitted.getSigmaXY(), Pvtx_refitted.getSigmaY2(), Pvtx_refitted.getSigmaXZ(), Pvtx_refitted.getSigmaYZ(), Pvtx_refitted.getSigmaZ2());

            auto trueID = trueVec_globID_contr[it];
            const o2::track::TrackParCov& trc = recoData.getTrackParam(trueID);
            //auto trackPar = getTrackPar(trc);
            //LOGP(info, "BEFORE");
            auto pt = trc.getPt();
            //LOGP(info, "HERE");
            o2::gpu::gpustd::array<float, 2> dcaInfo{-999., -999.};
            LOGP(info, " ---> Bz={}", o2::base::Propagator::Instance()->getNominalBz());
            if (o2::base::Propagator::Instance()->propagateToDCABxByBz({Pvtx_refitted.getX(), Pvtx_refitted.getY(), Pvtx_refitted.getZ()}, const_cast<o2::track::TrackParCov&>(trc), 2.f, matCorr, &dcaInfo)) {
              impParRPhi = dcaInfo[0] * toMicrometers;
              impParZ = dcaInfo[1] * toMicrometers;
              mHisto_ImpParZ->Fill(pt, impParZ);
              mHisto_ImpParXY->Fill(pt, impParRPhi);
              double phi = trc.getPhi();
              if (phi < 0) phi += 6.28318;
                  mHisto_Phi->Fill(phi, impParRPhi);
            }
          } //end recalc impact param
        }
      } // end loop tracks in pv
    } // else pv refit duable */
    vec_globID_contr.clear();
    vecPvContributorTrackParCov.clear();
    trueVec_globID_contr.clear();
    trueVecPvContributorTrackParCov.clear();
/* 
    mHisto_ImpParXY->FitSlicesY();
    mHisto_ImpParZ->FitSlicesY();
    mHisto_ImpParZ_2 = (std::make_unique<TH1F*>)gROOT->FindObject("h1_ImpParXY_2");
    mHisto_ImpParXY_2 = (std::make_unique<TH1F*>)gROOT->FindObject("h1_ImpParZ_2"); */
    
  } // end loop pv

} // end process

void ImpactParameterStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  o2::base::Propagator::Instance();
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // Note: reading of the ITS AlpideParam needed for ITS timing is done by the RecoContainer
    auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();

    // const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    // if (!grp->isDetContinuousReadOut(DetID::ITS)) {
    //   mITSROFrameLengthMUS = alpParams.roFrameLengthTrig / 1.e3;                                         // ITS ROFrame duration in \mus
    // } else {
    //   mITSROFrameLengthMUS = alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS ROFrame duration in \mus
    // }
    // mITSROFBiasMUS = alpParams.roFrameBiasInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
    // if (o2::base::GRPGeomHelper::instance().getGRPECS()->getRunType() != o2::parameters::GRPECSObject::RunType::COSMICS) {
    //   mVertexer.setBunchFilling(o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling());
    // }
    // mVertexer.setITSROFrameLength(mITSROFrameLengthMUS);
    //LOGP(info, "prima");
    mVertexer.init();
    //LOGP(info, "dopo");
    if (pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId == 0) {
      // PVertexerParams::Instance().printKeyValues();
    }
  }
}

void ImpactParameterStudy::endOfStream(EndOfStreamContext& ec)
{
  mDBGOut.reset();
  TFile fout(mOutName.c_str(), "update");
  fout.WriteTObject(mHisto_contributorsPV.get()); 
  fout.WriteTObject(mHisto_trackType.get()); 
  fout.WriteTObject(mHisto_trackTypeRej.get());   
  fout.WriteTObject(mHisto_trackTypeAcc.get());   
  fout.WriteTObject(mHisto_X_PVrefitChi2minus1.get());
  fout.WriteTObject(mHisto_Y_PVrefitChi2minus1.get());
  fout.WriteTObject(mHisto_Z_PVrefitChi2minus1.get());
  fout.WriteTObject(mHisto_X_DeltaPVrefitChi2minus1.get());
  fout.WriteTObject(mHisto_Y_DeltaPVrefitChi2minus1.get());
  fout.WriteTObject(mHisto_Z_DeltaPVrefitChi2minus1.get());
  fout.WriteTObject(mHisto_ImpParZ.get());
  fout.WriteTObject(mHisto_ImpParXY.get());
  fout.WriteTObject(mHisto_ImpParZ_2.get());
  fout.WriteTObject(mHisto_ImpParXY_2.get());
  fout.WriteTObject(mHisto_Phi.get());
  LOGP(info, "Stored Impact Parameters histograms {} and {} into {}", mHisto_ImpParZ->GetName(), mHisto_ImpParXY->GetName(), mOutName.c_str());
  fout.Close();
}

void ImpactParameterStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
}

DataProcessorSpec getImpactParameterStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestPrimaryVertertices(useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  return DataProcessorSpec{
    "its-study-impactparameter",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ImpactParameterStudy>(dataRequest, ggRequest, srcTracksMask)},
    Options{}};
}
} // namespace study
} // namespace its
} // namespace o2