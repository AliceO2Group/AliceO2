// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Analysis/Multiplicity.h"
#include "Analysis/EventSelection.h"
#include "Analysis/Centrality.h"
#include "Analysis/TriggerAliases.h"
#include "Analysis/ReducedInfoTables.h"
#include "Analysis/VarManager.h"
#include "Analysis/HistogramManager.h"
#include "PID/PIDResponse.h"
#include <iostream>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
//using namespace o2::framework::expressions;
using namespace o2::aod;

struct TableMaker_pp {

  Produces<ReducedEvents> event;
  Produces<ReducedEventsExtended> eventExtended;
  Produces<ReducedEventsVtxCov> eventVtxCov;
  Produces<ReducedTracks> trackBasic;
  Produces<ReducedTracksBarrel> trackBarrel;
  Produces<ReducedTracksBarrelCov> trackBarrelCov;
  Produces<ReducedTracksBarrelPID> trackBarrelPID;
  Produces<ReducedMuons> muonBasic;
  Produces<ReducedMuonsExtended> muonExtended;

  OutputObj<HistogramManager> fHistMan{"output"};

  // HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
  //         a constexpr static bit map must be defined and sent as template argument
  //        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
  //        Additionally, one should make sure that the requested tables are actually provided in the process() function,
  //       otherwise a compile time error will be thrown.
  //        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
  //           to automatically detect the object types transmitted to the VarManager
  constexpr static uint32_t fgEventFillMap = VarManager::ObjTypes::BC | VarManager::ObjTypes::Collision;
  constexpr static uint32_t fgTrackFillMap = VarManager::ObjTypes::Track | VarManager::ObjTypes::TrackExtra | VarManager::ObjTypes::TrackCov;

  void init(o2::framework::InitContext&)
  {
    VarManager::SetDefaultVarNames();
    fHistMan.setObject(new HistogramManager("analysisHistos", "aa", VarManager::kNVars));

    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms("Event;");                      // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator collision, aod::MuonClusters const& clustersMuon, aod::Muons const& tracksMuon, aod::BCs const& bcs, soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov /*, aod::pidRespTPC, aod::pidRespTOF, aod::pidRespTOFbeta*/> const& tracksBarrel)
  {
    uint64_t tag = 0;
    uint32_t triggerAliases = 0;
    for (int i = 0; i < kNaliases; i++) {
      if (collision.alias()[i] > 0) {
        triggerAliases |= (uint32_t(1) << i);
      }
    }

    VarManager::ResetValues();
    VarManager::FillEvent<fgEventFillMap>(collision);       // extract event information and place it in the fgValues array
    fHistMan->FillHistClass("Event", VarManager::fgValues); // automatically fill all the histograms in the class Event

    event(tag, collision.bc().runNumber(), collision.posX(), collision.posY(), collision.posZ(), collision.numContrib());
    eventExtended(collision.bc().globalBC(), collision.bc().triggerMask(), triggerAliases, 0.0f);
    eventVtxCov(collision.covXX(), collision.covXY(), collision.covXZ(), collision.covYY(), collision.covYZ(), collision.covZZ(), collision.chi2());

    uint64_t trackFilteringTag = 0;
    float sinAlpha = 0.f;
    float cosAlpha = 0.f;
    float globalX = 0.f;
    float globalY = 0.f;
    float dcaXY = 0.f;
    float dcaZ = 0.f;
    for (auto& track : tracksBarrel) {

      if (track.pt() < 0.15) {
        continue;
      }
      if (TMath::Abs(track.eta()) > 0.9) {
        continue;
      }

      sinAlpha = sin(track.alpha());
      cosAlpha = cos(track.alpha());
      globalX = track.x() * cosAlpha - track.y() * sinAlpha;
      globalY = track.x() * sinAlpha + track.y() * cosAlpha;

      dcaXY = sqrt(pow((globalX - collision.posX()), 2) +
                   pow((globalY - collision.posY()), 2));
      dcaZ = sqrt(pow(track.z() - collision.posZ(), 2));

      trackBasic(collision, track.globalIndex(), trackFilteringTag, track.pt(), track.eta(), track.phi(), track.charge());
      trackBarrel(track.tpcInnerParam(), track.flags(), track.itsClusterMap(), track.itsChi2NCl(),
                  track.tpcNClsFindable(), track.tpcNClsFindableMinusFound(), track.tpcNClsFindableMinusCrossedRows(),
                  track.tpcNClsShared(), track.tpcChi2NCl(),
                  track.trdChi2(), track.tofChi2(),
                  track.length(), dcaXY, dcaZ);
      trackBarrelCov(track.cYY(), track.cZZ(), track.cSnpSnp(), track.cTglTgl(), track.c1Pt21Pt2());
      trackBarrelPID(track.tpcSignal(),
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      //track.tpcNSigmaEl(), track.tpcNSigmaMu(),
      //track.tpcNSigmaPi(), track.tpcNSigmaKa(), track.tpcNSigmaPr(),
      //track.tpcNSigmaDe(), track.tpcNSigmaTr(), track.tpcNSigmaHe(), track.tpcNSigmaAl(),
      //track.tofSignal(), track.beta(),
      //track.tofNSigmaEl(), track.tofNSigmaMu(),
      //track.tofNSigmaPi(), track.tofNSigmaKa(), track.tofNSigmaPr(),
      //track.tofNSigmaDe(), track.tofNSigmaTr(), track.tofNSigmaHe(), track.tofNSigmaAl(),
      //track.trdSignal());
    }

    for (auto& muon : tracksMuon) {
      // TODO: add proper information for muon tracks
      if (muon.bc() != collision.bc()) {
        continue;
      }
      trackFilteringTag |= (uint64_t(1) << 0); // this is a MUON arm track
      muonBasic(collision, trackFilteringTag, muon.pt(), muon.eta(), muon.phi(), muon.charge());
      muonExtended(muon.inverseBendingMomentum(), muon.thetaX(), muon.thetaY(), muon.zMu(), muon.bendingCoor(), muon.nonBendingCoor(), muon.chi2(), muon.chi2MatchTrigger());
    }
  }

  void DefineHistograms(TString histClasses)
  {
    const int kNRuns = 2;
    int runs[kNRuns] = {244918, 244919};
    TString runsStr;
    for (int i = 0; i < kNRuns; i++) {
      runsStr += Form("%d;", runs[i]);
    }
    VarManager::SetRunNumbers(kNRuns, runs);

    TObjArray* arr = histClasses.Tokenize(";");
    for (Int_t iclass = 0; iclass < arr->GetEntries(); ++iclass) {
      TString classStr = arr->At(iclass)->GetName();

      if (classStr.Contains("Event")) {
        fHistMan->AddHistClass(classStr.Data());
        fHistMan->AddHistogram(classStr.Data(), "VtxZ", "Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ); // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxZ_Run", "Vtx Z", true,
                               kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId, 60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, runsStr.Data());                                        // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxX_VtxY", "Vtx X vs Vtx Y", false, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY);                                             // TH2F histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxX_VtxY_VtxZ", "vtx x - y - z", false, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 60, -15.0, 15.0, VarManager::kVtxZ);     // TH3F histogram
        fHistMan->AddHistogram(classStr.Data(), "NContrib_vs_VtxZ_prof", "Vtx Z vs ncontrib", true, 30, -15.0, 15.0, VarManager::kVtxZ, 10, -1., 1., VarManager::kVtxNcontrib);                             // TProfile histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxZ_vs_VtxX_VtxY_prof", "Vtx Z vs (x,y)", true, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 10, -1., 1., VarManager::kVtxZ); // TProfile2D histogram
        fHistMan->AddHistogram(classStr.Data(), "Ncontrib_vs_VtxZ_VtxX_VtxY_prof", "n-contrib vs (x,y,z)", true,
                               100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 30, -15., 15., VarManager::kVtxZ,
                               "", "", "", VarManager::kVtxNcontrib); // TProfile3D
      }
    } // end loop over histogram classes
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableMaker_pp>("table-maker-pp")};
}
