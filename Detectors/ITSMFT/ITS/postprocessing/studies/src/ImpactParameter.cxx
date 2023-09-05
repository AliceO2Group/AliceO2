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

#include <set>
#include <vector>
#include <utility>
#include <string>
#include "ITSStudies/ImpactParameter.h"
#include "ITSStudies/TrackCuts.h"
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
#include "DataFormatsParameters/GRPECSObject.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Framework/DeviceSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigParamRegistry.h"
#include <TH1F.h>
#include <TH2F.h>
#include <TMath.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TString.h>
#include "TGeoGlobalMagField.h"

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;
using namespace o2::its::study;

using DetID = o2::detectors::DetID;
using PVertex = o2::dataformats::PrimaryVertex;
using GTrackID = o2::dataformats::GlobalTrackID;

class ImpactParameterStudy : public Task
{
 public:
  ImpactParameterStudy(std::shared_ptr<DataRequest> dr,
                       std::shared_ptr<o2::base::GRPGeomRequest> gr,
                       mask_t src,
                       bool useMC) : mDataRequest(dr),
                                     mGGCCDBRequest(gr),
                                     mTracksSrc(src),
                                     mUseMC(useMC) {}
  ~ImpactParameterStudy() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void process(o2::globaltracking::RecoContainer&);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  void saveHistograms();
  void plotHistograms();
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  GTrackID::mask_t mTracksSrc{};
  bool mUseMC{false}; ///< MC flag
  o2::vertexing::PVertexer mVertexer;
  float mITSROFrameLengthMUS = 0.;
  float mITSROFBiasMUS = 0.;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  // output histograms
  std::unique_ptr<TH1F> mHistoContributorsPV{};
  std::unique_ptr<TH1F> mHistoTrackType;
  std::unique_ptr<TH1F> mHistoTrackTypeRej;
  std::unique_ptr<TH1F> mHistoTrackTypeAcc;
  std::unique_ptr<TH2F> mHistoXPvVsRefitted{};
  std::unique_ptr<TH2F> mHistoYPvVsRefitted{};
  std::unique_ptr<TH2F> mHistoZPvVsRefitted{};
  std::unique_ptr<TH1F> mHistoXDeltaPVrefit{};
  std::unique_ptr<TH1F> mHistoYDeltaPVrefit{};
  std::unique_ptr<TH1F> mHistoZDeltaPVrefit{};
  std::unique_ptr<TH2F> mHistoImpParXy{};
  std::unique_ptr<TH2F> mHistoImpParZ{};
  std::unique_ptr<TH2F> mHistoImpParXyPhi{};
  std::unique_ptr<TH2F> mHistoImpParZPhi{};
  std::unique_ptr<TH2F> mHistoImpParXyTop{};
  std::unique_ptr<TH2F> mHistoImpParZTop{};
  std::unique_ptr<TH2F> mHistoImpParXyBottom{};
  std::unique_ptr<TH2F> mHistoImpParZBottom{};
  std::unique_ptr<TH2F> mHistoImpParXyPositiveCharge{};
  std::unique_ptr<TH2F> mHistoImpParZPositiveCharge{};
  std::unique_ptr<TH2F> mHistoImpParXyNegativeCharge{};
  std::unique_ptr<TH2F> mHistoImpParZNegativeCharge{};
  std::unique_ptr<TH1F> mHistoImpParXyMeanPhi{};
  std::unique_ptr<TH1F> mHistoImpParZMeanPhi{};
  std::unique_ptr<TH1F> mHistoImpParXySigmaPhi{};
  std::unique_ptr<TH1F> mHistoImpParZSigmaPhi{};
  std::unique_ptr<TH1F> mHistoImpParXySigma{};
  std::unique_ptr<TH1F> mHistoImpParZSigma{};
  std::unique_ptr<TH1F> mHistoImpParXySigmaTop{};
  std::unique_ptr<TH1F> mHistoImpParZSigmaTop{};
  std::unique_ptr<TH1F> mHistoImpParXySigmaBottom{};
  std::unique_ptr<TH1F> mHistoImpParZSigmaBottom{};
  std::unique_ptr<TH1F> mHistoImpParXyMeanTop{};
  std::unique_ptr<TH1F> mHistoImpParZMeanTop{};
  std::unique_ptr<TH1F> mHistoImpParXyMeanBottom{};
  std::unique_ptr<TH1F> mHistoImpParZMeanBottom{};
  std::unique_ptr<TH1F> mHistoImpParXySigmaPositiveCharge{};
  std::unique_ptr<TH1F> mHistoImpParZSigmaPositiveCharge{};
  std::unique_ptr<TH1F> mHistoImpParXySigmaNegativeCharge{};
  std::unique_ptr<TH1F> mHistoImpParZSigmaNegativeCharge{};

  // output file
  TString mOutName{};

  // Data
  std::shared_ptr<DataRequest> mDataRequest;
  gsl::span<const PVertex> mPVertices;
};

void ImpactParameterStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  auto& params = ITSImpactParameterParamConfig::Instance();
  mOutName = params.outFileName;
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mOutName.Data(), "recreate");

  std::vector<double> logPtBinning = helpers::makeLogBinning(100, 0.1, 10);
  mHistoTrackType = std::make_unique<TH1F>("trackType", "# Track Type", 32, -0.5, 31.5);
  mHistoTrackTypeRej = std::make_unique<TH1F>("trackTypeRej", "# Rejected Track Type", 32, -0.5, 31.5);
  mHistoTrackTypeAcc = std::make_unique<TH1F>("trackTypeAcc", "# Filtered Track Type", 32, -0.5, 31.5);
  mHistoContributorsPV = std::make_unique<TH1F>("nContribPVrefit", "# Contributors per PV", 100, 0, 100);
  mHistoXPvVsRefitted = std::make_unique<TH2F>("histo2dXPvVsPVrefit", "#X PV vs PV_{-1}, #mum", 100, -10, 10, 100, -10, 10);
  mHistoYPvVsRefitted = std::make_unique<TH2F>("histo2dYPvVsPVrefit", "#Y PV vs PV_{-1}, #mum", 100, -10, 10, 100, -10, 10);
  mHistoZPvVsRefitted = std::make_unique<TH2F>("histo2dZPvVsPVrefit", "#Z PV vs PV_{-1}, #mum", 100, -10, 10, 100, -10, 10);
  mHistoXDeltaPVrefit = std::make_unique<TH1F>("histoDeltaXPVrefit", "#DeltaX (PV-PV_{-1}), #mum", 300, -15, 15);
  mHistoYDeltaPVrefit = std::make_unique<TH1F>("histoDeltaYPVrefit", "#DeltaY (PV-PV_{-1}), #mum", 300, -15, 15);
  mHistoZDeltaPVrefit = std::make_unique<TH1F>("histoDeltaZPVrefit", "#DeltaZ (PV-PV_{-1}), #mum", 300, -15, 15);
  mHistoImpParXyPhi = std::make_unique<TH2F>("histoImpParXyPhi", "#Phi; #phi; Impact Parameter XY (#mum);", 100, 0., 6.28, 100, -1000, 1000);
  mHistoImpParZPhi = std::make_unique<TH2F>("histoImpParZPhi", "#Phi; #phi; Impact Parameter Z (#mum);", 100, 0., 6.28, 100, -1000, 1000);
  mHistoImpParZ = std::make_unique<TH2F>("histoImpParZ", "Impact Parameter Z; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParXy = std::make_unique<TH2F>("histoImpParXy", "Impact Parameter XY; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParXyTop = std::make_unique<TH2F>("histoImpParXyTop", "Impact Parameter XY, #phi(track)<#pi; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParXyBottom = std::make_unique<TH2F>("histoImpParXyBottom", "Impact Parameter XY, #phi(track)>#pi; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParZTop = std::make_unique<TH2F>("histoImpParZTop", "Impact Parameter Z, #phi(track)<#pi; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParZBottom = std::make_unique<TH2F>("histoImpParZBottom", "Impact Parameter Z, #phi(track)>#pi; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParXyNegativeCharge = std::make_unique<TH2F>("histoImpParXyNegativeCharge", "Impact Parameter XY, sign<0; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParXyPositiveCharge = std::make_unique<TH2F>("histoImpParXyPositiveCharge", "Impact Parameter XY, sign>0; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParZNegativeCharge = std::make_unique<TH2F>("histoImpParZNegativeCharge", "Impact Parameter Z, sign<0; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);
  mHistoImpParZPositiveCharge = std::make_unique<TH2F>("histoImpParZPositiveCharge", "Impact Parameter Z, sign>0; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data(), 100, -1000, 1000);

  mHistoImpParXySigma = std::make_unique<TH1F>("histoImpParXySigma", "Pointing Resolution XY; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParZSigma = std::make_unique<TH1F>("histoImpParZSigma", "Pointing Resolution Z; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParXyMeanPhi = std::make_unique<TH1F>("histoImpParXyMean", "Pointing Resolution XY; #phi; Mean #mum", 100, 0., 6.28);
  mHistoImpParZMeanPhi = std::make_unique<TH1F>("histoImpParZMean", "Pointing Resolution Z; #phi; Mean #mum", 100, 0., 6.28);
  mHistoImpParXySigmaPhi = std::make_unique<TH1F>("histoImpParXySigmaPhi", "Pointing Resolution XY; #phi; #sigma #mum", 100, 0., 6.28);
  mHistoImpParZSigmaPhi = std::make_unique<TH1F>("histoImpParZSigmaPhi", "Pointing Resolution Z; #phi; #sigma #mum", 100, 0., 6.28);
  mHistoImpParXySigmaTop = std::make_unique<TH1F>("histoImpParXySigmaTop", "Pointing Resolution XY, Top; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParZSigmaTop = std::make_unique<TH1F>("histoImpParZSigmaTop", "Pointing Resolution Z, Top; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParXySigmaBottom = std::make_unique<TH1F>("histoImpParXySigmaBottom", "Pointing Resolution XY, Bottom; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParZSigmaBottom = std::make_unique<TH1F>("histoImpParZSigmaBottom", "Pointing Resolution Z, Bottom; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParXyMeanTop = std::make_unique<TH1F>("histoImpParXyMeanTop", "Pointing Resolution XY, Top; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParZMeanTop = std::make_unique<TH1F>("histoImpParZMeanTop", "Pointing Resolution Z, Top; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParXyMeanBottom = std::make_unique<TH1F>("histoImpParXyMeanBottom", "Mean Pointing Resolution XY, Bottom; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParZMeanBottom = std::make_unique<TH1F>("histoImpParZMeanBottom", "Mean Pointing Resolution Z, Bottom; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParXySigmaPositiveCharge = std::make_unique<TH1F>("histoImpParXySigmaPositiveCharge", "Pointing Resolution XY, sign>0; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParZSigmaPositiveCharge = std::make_unique<TH1F>("histoImpParZSigmaPositiveCharge", "Pointing Resolution Z, sign>0; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParXySigmaNegativeCharge = std::make_unique<TH1F>("histoImpParXySigmaNegativeCharge", "Pointing Resolution XY, sign<0; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());
  mHistoImpParZSigmaNegativeCharge = std::make_unique<TH1F>("histoImpParZSigmaNegativeCharge", "Pointing Resolution Z, sign<0; #it{p}_{T} (GeV/#it{c}); #mum", logPtBinning.size() - 1, logPtBinning.data());

  mHistoTrackType->SetDirectory(nullptr);
  mHistoTrackTypeRej->SetDirectory(nullptr);
  mHistoTrackTypeAcc->SetDirectory(nullptr);
  mHistoContributorsPV->SetDirectory(nullptr);
  mHistoXPvVsRefitted->SetDirectory(nullptr);
  mHistoYPvVsRefitted->SetDirectory(nullptr);
  mHistoZPvVsRefitted->SetDirectory(nullptr);
  mHistoXDeltaPVrefit->SetDirectory(nullptr);
  mHistoYDeltaPVrefit->SetDirectory(nullptr);
  mHistoZDeltaPVrefit->SetDirectory(nullptr);
  mHistoImpParZ->SetDirectory(nullptr);
  mHistoImpParXy->SetDirectory(nullptr);
  mHistoImpParXyPhi->SetDirectory(nullptr);
  mHistoImpParZPhi->SetDirectory(nullptr);
  mHistoImpParXyTop->SetDirectory(nullptr);
  mHistoImpParXyBottom->SetDirectory(nullptr);
  mHistoImpParZTop->SetDirectory(nullptr);
  mHistoImpParZBottom->SetDirectory(nullptr);
  mHistoImpParXyNegativeCharge->SetDirectory(nullptr);
  mHistoImpParXyPositiveCharge->SetDirectory(nullptr);
  mHistoImpParZNegativeCharge->SetDirectory(nullptr);
  mHistoImpParZPositiveCharge->SetDirectory(nullptr);
  mHistoImpParXyMeanPhi->SetDirectory(nullptr);
  mHistoImpParZMeanPhi->SetDirectory(nullptr);
  mHistoImpParXySigmaPhi->SetDirectory(nullptr);
  mHistoImpParZSigmaPhi->SetDirectory(nullptr);
  mHistoImpParXySigma->SetDirectory(nullptr);
  mHistoImpParZSigma->SetDirectory(nullptr);
  mHistoImpParXySigmaTop->SetDirectory(nullptr);
  mHistoImpParZSigmaTop->SetDirectory(nullptr);
  mHistoImpParXySigmaBottom->SetDirectory(nullptr);
  mHistoImpParZSigmaBottom->SetDirectory(nullptr);
  mHistoImpParXyMeanTop->SetDirectory(nullptr);
  mHistoImpParZMeanTop->SetDirectory(nullptr);
  mHistoImpParXyMeanBottom->SetDirectory(nullptr);
  mHistoImpParZMeanBottom->SetDirectory(nullptr);
  mHistoImpParXySigmaPositiveCharge->SetDirectory(nullptr);
  mHistoImpParZSigmaPositiveCharge->SetDirectory(nullptr);
  mHistoImpParXySigmaNegativeCharge->SetDirectory(nullptr);
  mHistoImpParZSigmaNegativeCharge->SetDirectory(nullptr);
}

void ImpactParameterStudy::run(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);
  process(recoData);
}

void ImpactParameterStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  auto& params = ITSImpactParameterParamConfig::Instance();

  o2::base::GRPGeomHelper::instance().getGRPMagField()->print();
  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;
  std::vector<o2::track::TrackParCov> vecPvContributorTrackParCov;
  std::vector<int64_t> vec_globID_contr = {};
  std::vector<o2::track::TrackParCov> trueVecPvContributorTrackParCov;
  std::vector<int64_t> trueVec_globID_contr = {};
  float impParRPhi, impParZ;
  constexpr float toMicrometers = 10000.f;                    // Conversion from [cm] to [mum]
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto pvertices = recoData.getPrimaryVertices();
  auto trmTPCTracks = recoData.getTPCTracks();

  TrackCuts cuts; // ITS and TPC(commented) cut implementation

  int nv = vtxRefs.size() - 1;      // The last entry is for unassigned tracks, ignore them
  for (int iv = 0; iv < nv; iv++) { // Loop over PVs
    LOGP(info, "*** NEW VERTEX  {}***", iv);
    const auto& vtref = vtxRefs[iv];
    const o2::dataformats::VertexBase& pv = pvertices[iv];
    int it = vtref.getFirstEntry(), itLim = it + vtref.getEntries();
    int i = 0;

    for (; it < itLim; it++) {
      auto tvid = trackIndex[it];
      // LOGP(info,"ORIGIN: {}", tvid.getSourceName(tvid.getSource()));
      mHistoTrackType->Fill(tvid.getSource());
      if (!recoData.isTrackSourceLoaded(tvid.getSource())) {
        // LOGP(info,"SOURCE Rej: {}", tvid.getSourceName(tvid.getSource()));
        mHistoTrackTypeRej->Fill(tvid.getSource());
        continue;
      }
      // LOGP(info,"ORIGIN: {}  INDEX: {}", tvid.getSourceName(tvid.getSource()), trackIndex[it]);
      mHistoTrackTypeAcc->Fill(tvid.getSource());
      const o2::track::TrackParCov& trc = recoData.getTrackParam(tvid); // The actual track

      auto refs = recoData.getSingleDetectorRefs(tvid);
      if (!refs[GTrackID::ITS].isIndexSet()) { // might be an afterburner track
        // LOGP(info, " ** AFTERBURN **");
        continue;
      }
      // Apply track selections
      if (params.applyTrackCuts) {
        if (!cuts.isSelected(tvid, recoData)) {
          // LOGP(info, "Fail");
          continue;
        }
      }
      // Vectors for reconstructing the vertex
      vec_globID_contr.push_back(trackIndex[it]);
      vecPvContributorTrackParCov.push_back(trc);
      // Store vector with index and tracks ITS only   -- same order as the GLOBAL vectors
      int itsTrackID = refs[GTrackID::ITS].getIndex();
      const o2::track::TrackParCov& trcITS = recoData.getTrackParam(itsTrackID); // The actual ITS track
      trueVec_globID_contr.push_back(itsTrackID);
      trueVecPvContributorTrackParCov.push_back(trcITS);
      // LOGP(info, "SOURCE: {}  indexGLOBAL:  {}  indexITS:  {}", tvid.getSourceName(tvid.getSource()), trackIndex[it], trueVec_globID_contr[i]);
      i++;
    } // end loop tracks
    LOGP(info, "************  SIZE INDEX GLOBAL:   {}   ", vec_globID_contr.size());
    LOGP(info, "************  SIZE INDEX ITS:   {}   ", trueVec_globID_contr.size());

    it = vtref.getFirstEntry();
    // Preparation PVertexer refit
    o2::conf::ConfigurableParam::updateFromString("pvertexer.useMeanVertexConstraint=false");
    bool PVrefit_doable = mVertexer.prepareVertexRefit(vecPvContributorTrackParCov, pv);
    if (!PVrefit_doable) {
      LOG(info) << "Not enough tracks accepted for the refit --> Skipping vertex";
    } else {
      mHistoContributorsPV->Fill(vecPvContributorTrackParCov.size());
      // Neglect vertices with small number of contributors (configurable)
      if (vecPvContributorTrackParCov.size() < params.minNumberOfContributors) {
        continue;
      }
      for (it = 0; it < vec_globID_contr.size(); it++) {
        // vector of booleans to keep track of the track to be skipped
        std::vector<bool> vec_useTrk_PVrefit(vec_globID_contr.size(), true);
        auto tvid = vec_globID_contr[it];
        auto trackIterator = std::find(vec_globID_contr.begin(), vec_globID_contr.end(), vec_globID_contr[it]);
        if (trackIterator != vec_globID_contr.end()) {
          // LOGP(info,"************  Trackiterator:   {}  : {} ", it+1, vec_globID_contr[it]);
          o2::dataformats::VertexBase PVbase_recalculated;
          /// this track contributed to the PV fit: let's do the refit without it
          const int entry = std::distance(vec_globID_contr.begin(), trackIterator);
          if (!params.useAllTracks) {
            vec_useTrk_PVrefit[entry] = false; /// remove the track from the PV refitting
          }
          auto Pvtx_refitted = mVertexer.refitVertex(vec_useTrk_PVrefit, pv); // vertex refit
          // enable the dca recalculation for the current PV contributor, after removing it from the PV refit
          bool recalc_imppar = true;
          if (Pvtx_refitted.getChi2() < 0) {
            // LOG(info) << "---> Refitted vertex has bad chi2 = " << Pvtx_refitted.getChi2();
            recalc_imppar = false;
          }
          vec_useTrk_PVrefit[entry] = true; /// restore the track for the next PV refitting
          if (recalc_imppar) {
            const double DeltaX = pv.getX() - Pvtx_refitted.getX();
            const double DeltaY = pv.getY() - Pvtx_refitted.getY();
            const double DeltaZ = pv.getZ() - Pvtx_refitted.getZ();
            mHistoXPvVsRefitted->Fill(pv.getX(), Pvtx_refitted.getX());
            mHistoYPvVsRefitted->Fill(pv.getY(), Pvtx_refitted.getY());
            mHistoZPvVsRefitted->Fill(pv.getZ(), Pvtx_refitted.getZ());
            mHistoXDeltaPVrefit->Fill(DeltaX);
            mHistoYDeltaPVrefit->Fill(DeltaY);
            mHistoZDeltaPVrefit->Fill(DeltaZ);

            // fill the newly calculated PV
            PVbase_recalculated.setX(Pvtx_refitted.getX());
            PVbase_recalculated.setY(Pvtx_refitted.getY());
            PVbase_recalculated.setZ(Pvtx_refitted.getZ());
            PVbase_recalculated.setCov(Pvtx_refitted.getSigmaX2(), Pvtx_refitted.getSigmaXY(), Pvtx_refitted.getSigmaY2(), Pvtx_refitted.getSigmaXZ(), Pvtx_refitted.getSigmaYZ(), Pvtx_refitted.getSigmaZ2());

            auto trueID = trueVec_globID_contr[it];
            const o2::track::TrackParCov& trc = recoData.getTrackParam(trueID);
            auto pt = trc.getPt();
            o2::gpu::gpustd::array<float, 2> dcaInfo{-999., -999.};
            // LOGP(info, " ---> Bz={}", o2::base::Propagator::Instance()->getNominalBz());
            if (o2::base::Propagator::Instance()->propagateToDCABxByBz({Pvtx_refitted.getX(), Pvtx_refitted.getY(), Pvtx_refitted.getZ()}, const_cast<o2::track::TrackParCov&>(trc), 2.f, matCorr, &dcaInfo)) {
              impParRPhi = dcaInfo[0] * toMicrometers;
              impParZ = dcaInfo[1] * toMicrometers;
              mHistoImpParXy->Fill(pt, impParRPhi);
              mHistoImpParZ->Fill(pt, impParZ);
              double phi = trc.getPhi();
              mHistoImpParXyPhi->Fill(phi, impParRPhi);
              mHistoImpParZPhi->Fill(phi, impParZ);
              if (phi < TMath::Pi()) {
                mHistoImpParXyTop->Fill(pt, impParRPhi);
                mHistoImpParZTop->Fill(pt, impParZ);
              }
              if (phi > TMath::Pi()) {
                mHistoImpParXyBottom->Fill(pt, impParRPhi);
                mHistoImpParZBottom->Fill(pt, impParZ);
              }
              double sign = trc.getSign();
              if (sign < 0) {
                mHistoImpParXyNegativeCharge->Fill(pt, impParRPhi);
                mHistoImpParZNegativeCharge->Fill(pt, impParZ);
              } else {
                mHistoImpParXyPositiveCharge->Fill(pt, impParRPhi);
                mHistoImpParZPositiveCharge->Fill(pt, impParZ);
              }
            }
          } // end recalc impact param
        }
      } // end loop tracks in pv refitted
    }   // else pv refit duable */
    vec_globID_contr.clear();
    vecPvContributorTrackParCov.clear();
    trueVec_globID_contr.clear();
    trueVecPvContributorTrackParCov.clear();

  } // end loop pv

  mHistoImpParXy->FitSlicesY();
  mHistoImpParZ->FitSlicesY();
  mHistoImpParXyPhi->FitSlicesY();
  mHistoImpParZPhi->FitSlicesY();
  mHistoImpParXyTop->FitSlicesY();
  mHistoImpParZTop->FitSlicesY();
  mHistoImpParXyBottom->FitSlicesY();
  mHistoImpParZBottom->FitSlicesY();
  mHistoImpParXyNegativeCharge->FitSlicesY();
  mHistoImpParZNegativeCharge->FitSlicesY();
  mHistoImpParXyPositiveCharge->FitSlicesY();
  mHistoImpParZPositiveCharge->FitSlicesY();
  mHistoImpParXySigma = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXy_2"))->Clone()));
  mHistoImpParZSigma = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZ_2"))->Clone()));
  mHistoImpParXyMeanPhi = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXyPhi_1"))->Clone()));
  mHistoImpParZMeanPhi = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZPhi_1"))->Clone()));
  mHistoImpParXySigmaPhi = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXyPhi_2"))->Clone()));
  mHistoImpParZSigmaPhi = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZPhi_2"))->Clone()));
  mHistoImpParXyMeanTop = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXyTop_1"))->Clone()));
  mHistoImpParZMeanTop = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZTop_1"))->Clone()));
  mHistoImpParXyMeanBottom = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXyBottom_1"))->Clone()));
  mHistoImpParZMeanBottom = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZBottom_1"))->Clone()));
  mHistoImpParXySigmaTop = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXyTop_2"))->Clone()));
  mHistoImpParZSigmaTop = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZTop_2"))->Clone()));
  mHistoImpParXySigmaBottom = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXyBottom_2"))->Clone()));
  mHistoImpParZSigmaBottom = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZBottom_2"))->Clone()));
  mHistoImpParXySigmaNegativeCharge = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXyNegativeCharge_2"))->Clone()));
  mHistoImpParZSigmaNegativeCharge = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZNegativeCharge_2"))->Clone()));
  mHistoImpParXySigmaPositiveCharge = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParXyPositiveCharge_2"))->Clone()));
  mHistoImpParZSigmaPositiveCharge = std::unique_ptr<TH1F>(static_cast<TH1F*>((gROOT->FindObject("histoImpParZPositiveCharge_2"))->Clone()));
}
// end process

void ImpactParameterStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();
    mVertexer.init();
    if (pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId == 0) {
    }
  }
}

void ImpactParameterStudy::saveHistograms()
{
  mDBGOut.reset();
  TFile fout(mOutName.Data(), "update");
  fout.WriteTObject(mHistoContributorsPV.get());
  fout.WriteTObject(mHistoTrackType.get());
  fout.WriteTObject(mHistoTrackTypeRej.get());
  fout.WriteTObject(mHistoTrackTypeAcc.get());
  fout.WriteTObject(mHistoXPvVsRefitted.get());
  fout.WriteTObject(mHistoYPvVsRefitted.get());
  fout.WriteTObject(mHistoZPvVsRefitted.get());
  fout.WriteTObject(mHistoXDeltaPVrefit.get());
  fout.WriteTObject(mHistoYDeltaPVrefit.get());
  fout.WriteTObject(mHistoZDeltaPVrefit.get());
  fout.WriteTObject(mHistoImpParZ.get());
  fout.WriteTObject(mHistoImpParXy.get());
  fout.WriteTObject(mHistoImpParXyPhi.get());
  fout.WriteTObject(mHistoImpParZPhi.get());
  fout.WriteTObject(mHistoImpParXyTop.get());
  fout.WriteTObject(mHistoImpParZTop.get());
  fout.WriteTObject(mHistoImpParXyBottom.get());
  fout.WriteTObject(mHistoImpParZBottom.get());
  fout.WriteTObject(mHistoImpParXyPositiveCharge.get());
  fout.WriteTObject(mHistoImpParZPositiveCharge.get());
  fout.WriteTObject(mHistoImpParXyNegativeCharge.get());
  fout.WriteTObject(mHistoImpParZNegativeCharge.get());
  fout.WriteTObject(mHistoImpParZMeanPhi.get());
  fout.WriteTObject(mHistoImpParXyMeanPhi.get());
  fout.WriteTObject(mHistoImpParZSigmaPhi.get());
  fout.WriteTObject(mHistoImpParXySigmaPhi.get());
  fout.WriteTObject(mHistoImpParZSigma.get());
  fout.WriteTObject(mHistoImpParXySigma.get());
  fout.WriteTObject(mHistoImpParZMeanTop.get());
  fout.WriteTObject(mHistoImpParXyMeanTop.get());
  fout.WriteTObject(mHistoImpParZMeanBottom.get());
  fout.WriteTObject(mHistoImpParXyMeanBottom.get());
  fout.WriteTObject(mHistoImpParZSigmaTop.get());
  fout.WriteTObject(mHistoImpParXySigmaTop.get());
  fout.WriteTObject(mHistoImpParZSigmaBottom.get());
  fout.WriteTObject(mHistoImpParXySigmaBottom.get());
  fout.WriteTObject(mHistoImpParZSigmaPositiveCharge.get());
  fout.WriteTObject(mHistoImpParXySigmaPositiveCharge.get());
  fout.WriteTObject(mHistoImpParZSigmaNegativeCharge.get());
  fout.WriteTObject(mHistoImpParXySigmaNegativeCharge.get());
  LOGP(info, "Impact Parameters histograms stored in {}", mOutName.Data());
  fout.Close();
}

void ImpactParameterStudy::plotHistograms()
{
  TString directoryName = "./plotsImpactParameter";
  gSystem->mkdir(directoryName);
  TCanvas* dcaXyVsdcaZ = helpers::prepareSimpleCanvas2Histograms(*mHistoImpParXySigma, kRed, "#sigma_{XY}", *mHistoImpParZSigma, kBlue, "#sigma_{Z}");
  TCanvas* dcaXyTopVsBottom = helpers::prepareSimpleCanvas2Histograms(*mHistoImpParXySigmaTop, kRed, "#sigma_{XY}^{TOP}", *mHistoImpParXySigmaBottom, kBlue, "#sigma_{XY}^{BOTTOM}");
  TCanvas* dcaZTopVsBottom = helpers::prepareSimpleCanvas2Histograms(*mHistoImpParZSigmaTop, kRed, "#sigma_{Z}^{TOP}", *mHistoImpParZSigmaBottom, kBlue, "#sigma_{Z}^{BOTTOM}");
  TCanvas* dcaXyPosVsNeg = helpers::prepareSimpleCanvas2Histograms(*mHistoImpParXySigmaPositiveCharge, kRed, "#sigma_{XY}^{positive}", *mHistoImpParXySigmaNegativeCharge, kBlue, "#sigma_{XY}^{negative}");
  TCanvas* dcaZPosVsNeg = helpers::prepareSimpleCanvas2Histograms(*mHistoImpParZSigmaPositiveCharge, kRed, "#sigma_{Z}^{positive}", *mHistoImpParZSigmaNegativeCharge, kBlue, "#sigma_{Z}^{negative}");
  TCanvas* dcaXYvsPhi = helpers::plot2DwithMeanAndSigma(*mHistoImpParXyPhi, *mHistoImpParXyMeanPhi, *mHistoImpParXySigmaPhi, kRed);
  TCanvas* dcaZvsPhi = helpers::plot2DwithMeanAndSigma(*mHistoImpParZPhi, *mHistoImpParZMeanPhi, *mHistoImpParZSigmaPhi, kRed);
  TCanvas* dcaXyPhiMeanTopVsBottom = helpers::prepareSimpleCanvas2DcaMeanValues(*mHistoImpParXyMeanTop, kRed, "mean_{XY}^{TOP}", *mHistoImpParXyMeanBottom, kBlue, "mean_{XY}^{BOTTOM}");
  TCanvas* dcaZPhiMeanTopVsBottom = helpers::prepareSimpleCanvas2DcaMeanValues(*mHistoImpParZMeanTop, kRed, "mean_{Z}^{TOP}", *mHistoImpParZMeanBottom, kBlue, "mean_{Z}^{BOTTOM}");

  dcaXyVsdcaZ->SaveAs(Form("%s/ComparisonDCAXYvsZ.png", directoryName.Data()));
  dcaXyTopVsBottom->SaveAs(Form("%s/ComparisonDCAXyTopVsBottom.png", directoryName.Data()));
  dcaXyPosVsNeg->SaveAs(Form("%s/ComparisonDCAXyPosVsNeg.png", directoryName.Data()));
  dcaZTopVsBottom->SaveAs(Form("%s/ComparisonDCAZTopVsBottom.png", directoryName.Data()));
  dcaZPosVsNeg->SaveAs(Form("%s/ComparisonDCAZPosVsNeg.png", directoryName.Data()));
  dcaXYvsPhi->SaveAs(Form("%s/dcaXYvsPhi.png", directoryName.Data()));
  dcaZvsPhi->SaveAs(Form("%s/dcaZvsPhi.png", directoryName.Data()));
  dcaXyPhiMeanTopVsBottom->SaveAs(Form("%s/dcaXyPhiMeanTopVsBottom.png", directoryName.Data()));
  dcaZPhiMeanTopVsBottom->SaveAs(Form("%s/dcaZPhiMeanTopVsBottom.png", directoryName.Data()));
}

void ImpactParameterStudy::endOfStream(EndOfStreamContext& ec)
{
  auto& params = ITSImpactParameterParamConfig::Instance();
  if (params.generatePlots) {
    saveHistograms();
    plotHistograms();
  }
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
    AlgorithmSpec{adaptFromTask<ImpactParameterStudy>(dataRequest, ggRequest, srcTracksMask, useMC)},
    Options{}};
}

} // namespace study
} // namespace its
} // namespace o2