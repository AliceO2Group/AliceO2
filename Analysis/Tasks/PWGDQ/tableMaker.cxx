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
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisCore/TriggerAliases.h"
#include "AnalysisDataModel/ReducedInfoTables.h"
#include "PWGDQCore/VarManager.h"
#include "PWGDQCore/HistogramManager.h"
#include "PWGDQCore/AnalysisCut.h"
#include "PWGDQCore/AnalysisCompositeCut.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include <iostream>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
//using namespace o2::framework::expressions;
using namespace o2::aod;

using MyEvents = soa::Join<aod::Collisions, aod::EvSels, aod::Cents>;
using MyBarrelTracks = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksExtended, aod::TrackSelection, aod::pidRespTPC, aod::pidRespTOF, aod::pidRespTOFbeta>;

// HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
//         a constexpr static bit map must be defined and sent as template argument
//        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
//        Additionally, one should make sure that the requested tables are actually provided in the process() function,
//       otherwise a compile time error will be thrown.
//        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
//           to automatically detect the object types transmitted to the VarManager
constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::BC | VarManager::ObjTypes::Collision | VarManager::ObjTypes::CollisionCent;
constexpr static uint32_t gkTrackFillMap = VarManager::ObjTypes::Track | VarManager::ObjTypes::TrackExtra | VarManager::ObjTypes::TrackDCA | VarManager::ObjTypes::TrackSelection | VarManager::ObjTypes::TrackCov | VarManager::ObjTypes::TrackPID;

struct TableMaker {

  Produces<ReducedEvents> event;
  Produces<ReducedEventsExtended> eventExtended;
  Produces<ReducedEventsVtxCov> eventVtxCov;
  Produces<ReducedTracks> trackBasic;
  Produces<ReducedTracksBarrel> trackBarrel;
  Produces<ReducedTracksBarrelCov> trackBarrelCov;
  Produces<ReducedTracksBarrelPID> trackBarrelPID;
  Produces<ReducedMuons> muonBasic;
  Produces<ReducedMuonsExtended> muonExtended;

  float* fValues;

  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;

  // TODO: Filters should be used to make lowest level selection. The additional more restrictive cuts should be defined via the AnalysisCuts
  // TODO: Multiple event selections can be applied and decisions stored in the reducedevent::tag
  AnalysisCompositeCut* fEventCut;
  // TODO: Multiple track selections can be applied and decisions stored in the reducedtrack::filteringFlags
  //  Cuts should be defined using Configurables (prepare cut libraries, as discussed in O2 DQ meetings)
  AnalysisCompositeCut* fTrackCut;

  // Partition will select fast a group of tracks with basic requirements
  //   If some of the cuts cannot be included in the Partition expression, add them via AnalysisCut(s)
  Partition<MyBarrelTracks> barrelSelectedTracks = o2::aod::track::pt >= 1.0f && nabs(o2::aod::track::eta) <= 0.9f && o2::aod::track::tpcSignal >= 70.0f && o2::aod::track::tpcSignal <= 100.0f && o2::aod::track::tpcChi2NCl < 4.0f && o2::aod::track::itsChi2NCl < 36.0f;

  // TODO a few of the important muon variables in the central data model are dynamic columns so not usable in expressions (e.g. eta, phi)
  //        Update the data model to have them as expression columns
  Partition<aod::Muons> muonSelectedTracks = o2::aod::muon::pt >= 1.0f;

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms("Event_BeforeCuts;Event_AfterCuts;TrackBarrel_BeforeCuts;TrackBarrel_AfterCuts"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
    DefineCuts();
  }

  void DefineCuts()
  {
    fEventCut = new AnalysisCompositeCut(true);
    AnalysisCut* eventVarCut = new AnalysisCut();
    eventVarCut->AddCut(VarManager::kVtxZ, -10.0, 10.0);
    eventVarCut->AddCut(VarManager::kIsINT7, 0.5, 1.5); // require kINT7
    fEventCut->AddCut(eventVarCut);

    fTrackCut = new AnalysisCompositeCut(true);
    AnalysisCut* trackVarCut = new AnalysisCut();
    //trackVarCut->AddCut(VarManager::kPt, 1.0, 1000.0);
    //trackVarCut->AddCut(VarManager::kEta, -0.9, 0.9);
    //trackVarCut->AddCut(VarManager::kTPCsignal, 70.0, 100.0);
    trackVarCut->AddCut(VarManager::kIsITSrefit, 0.5, 1.5);
    trackVarCut->AddCut(VarManager::kIsTPCrefit, 0.5, 1.5);
    //trackVarCut->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    //trackVarCut->AddCut(VarManager::kITSchi2, 0.1, 36.0);
    trackVarCut->AddCut(VarManager::kTPCncls, 100.0, 161.);

    AnalysisCut* pidCut1 = new AnalysisCut();
    TF1* cutLow1 = new TF1("cutLow1", "pol1", 0., 10.);
    cutLow1->SetParameters(130., -40.0);
    pidCut1->AddCut(VarManager::kTPCsignal, cutLow1, 100.0, false, VarManager::kPin, 0.5, 3.0);

    fTrackCut->AddCut(trackVarCut);
    fTrackCut->AddCut(pidCut1);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvents::iterator const& collision, aod::MuonClusters const& clustersMuon, aod::Muons const& tracksMuon, aod::BCs const& bcs, MyBarrelTracks const& tracksBarrel)
  {
    uint64_t tag = 0;
    uint32_t triggerAliases = 0;
    for (int i = 0; i < kNaliases; i++) {
      if (collision.alias()[i] > 0) {
        triggerAliases |= (uint32_t(1) << i);
      }
    }

    VarManager::ResetValues(0, VarManager::kNEventWiseVariables, fValues);
    VarManager::FillEvent<gkEventFillMap>(collision, fValues); // extract event information and place it in the fgValues array
    fHistMan->FillHistClass("Event_BeforeCuts", fValues);      // automatically fill all the histograms in the class Event

    if (!fEventCut->IsSelected(fValues)) {
      return;
    }

    fHistMan->FillHistClass("Event_AfterCuts", fValues);

    event(tag, collision.bc().runNumber(), collision.posX(), collision.posY(), collision.posZ(), collision.numContrib());
    eventExtended(collision.bc().globalBC(), collision.bc().triggerMask(), triggerAliases, collision.centV0M());
    eventVtxCov(collision.covXX(), collision.covXY(), collision.covXZ(), collision.covYY(), collision.covYZ(), collision.covZZ(), collision.chi2());

    uint64_t trackFilteringTag = 0;
    trackBasic.reserve(barrelSelectedTracks.size());
    trackBarrel.reserve(barrelSelectedTracks.size());
    trackBarrelCov.reserve(barrelSelectedTracks.size());
    trackBarrelPID.reserve(barrelSelectedTracks.size());

    for (auto& track : barrelSelectedTracks) {
      VarManager::FillTrack<gkTrackFillMap>(track, fValues);
      fHistMan->FillHistClass("TrackBarrel_BeforeCuts", fValues);
      if (!fTrackCut->IsSelected(fValues)) {
        continue;
      }
      fHistMan->FillHistClass("TrackBarrel_AfterCuts", fValues);

      if (track.isGlobalTrack()) {
        trackFilteringTag |= (uint64_t(1) << 0);
      }
      if (track.isGlobalTrackSDD()) {
        trackFilteringTag |= (uint64_t(1) << 1);
      }
      trackBasic(event.lastIndex(), track.globalIndex(), trackFilteringTag, track.pt(), track.eta(), track.phi(), track.charge());
      trackBarrel(track.tpcInnerParam(), track.flags(), track.itsClusterMap(), track.itsChi2NCl(),
                  track.tpcNClsFindable(), track.tpcNClsFindableMinusFound(), track.tpcNClsFindableMinusCrossedRows(),
                  track.tpcNClsShared(), track.tpcChi2NCl(),
                  track.trdChi2(), track.tofChi2(),
                  track.length(), track.dcaXY(), track.dcaZ());
      trackBarrelCov(track.cYY(), track.cZZ(), track.cSnpSnp(), track.cTglTgl(), track.c1Pt21Pt2());
      trackBarrelPID(track.tpcSignal(),
                     track.tpcNSigmaEl(), track.tpcNSigmaMu(),
                     track.tpcNSigmaPi(), track.tpcNSigmaKa(), track.tpcNSigmaPr(),
                     track.tpcNSigmaDe(), track.tpcNSigmaTr(), track.tpcNSigmaHe(), track.tpcNSigmaAl(),
                     track.tofSignal(), track.beta(),
                     track.tofNSigmaEl(), track.tofNSigmaMu(),
                     track.tofNSigmaPi(), track.tofNSigmaKa(), track.tofNSigmaPr(),
                     track.tofNSigmaDe(), track.tofNSigmaTr(), track.tofNSigmaHe(), track.tofNSigmaAl(),
                     track.trdSignal());
    }

    muonBasic.reserve(muonSelectedTracks.size());
    muonExtended.reserve(muonSelectedTracks.size());
    for (auto& muon : muonSelectedTracks) {
      // TODO: add proper information for muon tracks
      if (muon.bcId() != collision.bcId()) {
        continue;
      }
      // TODO: the trackFilteringTag will not be needed to encode whether the track is a muon since there is a dedicated table for muons
      trackFilteringTag |= (uint64_t(1) << 0); // this is a MUON arm track
      muonBasic(event.lastIndex(), trackFilteringTag, muon.pt(), muon.eta(), muon.phi(), muon.charge());
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

    std::unique_ptr<TObjArray> arr(histClasses.Tokenize(";"));
    for (Int_t iclass = 0; iclass < arr->GetEntries(); ++iclass) {
      TString classStr = arr->At(iclass)->GetName();

      if (classStr.Contains("Event")) {
        fHistMan->AddHistClass(classStr.Data());
        fHistMan->AddHistogram(classStr.Data(), "VtxZ", "Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ); // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "VtxZ_Run", "Vtx Z", true,
                               kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId, 60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, runsStr.Data());
        fHistMan->AddHistogram(classStr.Data(), "CentVZERO", "Centrality VZERO", false, 100, 0.0, 100.0, VarManager::kCentVZERO); // TH1F histogram
      }

      if (classStr.Contains("Track")) {
        fHistMan->AddHistClass(classStr.Data());
        fHistMan->AddHistogram(classStr.Data(), "Pt", "p_{T} distribution", false, 200, 0.0, 20.0, VarManager::kPt);                                                // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "Eta", "#eta distribution", false, 500, -5.0, 5.0, VarManager::kEta);                                               // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "Phi_Eta", "#phi vs #eta distribution", false, 200, -5.0, 5.0, VarManager::kEta, 200, -6.3, 6.3, VarManager::kPhi); // TH2F histogram

        if (classStr.Contains("Barrel")) {
          fHistMan->AddHistogram(classStr.Data(), "TPCncls", "Number of cluster in TPC", false, 160, -0.5, 159.5, VarManager::kTPCncls); // TH1F histogram
          fHistMan->AddHistogram(classStr.Data(), "TPCncls_Run", "Number of cluster in TPC", true, kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId,
                                 10, -0.5, 159.5, VarManager::kTPCncls, 10, 0., 1., VarManager::kNothing, runsStr.Data());           // TH1F histogram
          fHistMan->AddHistogram(classStr.Data(), "ITSncls", "Number of cluster in ITS", false, 8, -0.5, 7.5, VarManager::kITSncls); // TH1F histogram
          //for TPC PID
          fHistMan->AddHistogram(classStr.Data(), "TPCdedx_pIN", "TPC dE/dx vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, 0.0, 200., VarManager::kTPCsignal); // TH2F histogram
          fHistMan->AddHistogram(classStr.Data(), "DCAxy", "DCAxy", false, 100, -3.0, 3.0, VarManager::kTrackDCAxy);                                                   // TH1F histogram
          fHistMan->AddHistogram(classStr.Data(), "DCAz", "DCAz", false, 100, -5.0, 5.0, VarManager::kTrackDCAz);                                                      // TH1F histogram
          fHistMan->AddHistogram(classStr.Data(), "IsGlobalTrack", "IsGlobalTrack", false, 2, -0.5, 1.5, VarManager::kIsGlobalTrack);                                  // TH1F histogram
          fHistMan->AddHistogram(classStr.Data(), "IsGlobalTrackSDD", "IsGlobalTrackSDD", false, 2, -0.5, 1.5, VarManager::kIsGlobalTrackSDD);                         // TH1F histogram
        }
      }
    } // end loop over histogram classes
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableMaker>("table-maker")};
}
