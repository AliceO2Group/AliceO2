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
#include "Analysis/ReducedInfoTables.h"
#include "Analysis/VarManager.h"
#include "Analysis/HistogramManager.h"
#include "Analysis/AnalysisCut.h"
#include "Analysis/AnalysisCompositeCut.h"
#include <TH1F.h>
#include <TMath.h>
#include <THashList.h>
#include <TString.h>
#include <iostream>
#include <vector>
#include <typeinfo>

using std::cout;
using std::endl;
using std::vector;

using namespace o2;
using namespace o2::framework;
//using namespace o2::framework::expressions;
using namespace o2::aod;

struct TableReader {

  OutputObj<HistogramManager> fHistMan{"output"};
  AnalysisCompositeCut* fEventCut;
  AnalysisCompositeCut* fTrackCut;
  AnalysisCompositeCut* fMuonCut;

  // HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
  //         a constexpr static bit map must be defined and sent as template argument
  //        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
  //        Additionally, one should make sure that the requested tables are actually provided in the process() function,
  //       otherwise a compile time error will be thrown.
  //        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
  //           to automatically detect the object types transmitted to the VarManager
  constexpr static uint32_t fgEventFillMap = VarManager::ObjTypes::ReducedEvent | VarManager::ObjTypes::ReducedEventExtended | VarManager::ObjTypes::ReducedEventVtxCov;
  constexpr static uint32_t fgEventMuonFillMap = VarManager::ObjTypes::ReducedEvent | VarManager::ObjTypes::ReducedEventExtended;
  constexpr static uint32_t fgTrackFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackBarrel | VarManager::ObjTypes::ReducedTrackBarrelCov;
  constexpr static uint32_t fgMuonFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackMuon;

  void init(o2::framework::InitContext&)
  {
    VarManager::SetDefaultVarNames();
    fHistMan.setObject(new HistogramManager("analysisHistos", "aa", VarManager::kNVars));

    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms("Event_BeforeCuts;Event_AfterCuts;TrackBarrel_BeforeCuts;TrackBarrel_AfterCuts;TrackMuon_BeforeCuts;TrackMuon_AfterCuts;PairsBarrel;PairsMuon;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill

    DefineCuts();
  }

  void DefineCuts()
  {
    fEventCut = new AnalysisCompositeCut(true);

    AnalysisCut* varCut = new AnalysisCut();
    varCut->AddCut(VarManager::kVtxZ, -10.0, 10.0);

    TF1* cutLow = new TF1("cutLow", "pol1", 0., 0.1);
    cutLow->SetParameters(0.2635, 1.0);
    //varCut->AddCut(VarManager::kVtxY, cutLow, 0.335, false, VarManager::kVtxX, 0.067, 0.070);

    //varCut->AddCut(VarManager::kVtxY, 0.0, 0.335);
    fEventCut->AddCut(varCut);

    fTrackCut = new AnalysisCompositeCut(true); // true: use AND
    AnalysisCut* cut1 = new AnalysisCut();
    cut1->AddCut(VarManager::kPt, 1.0, 20.0);
    cut1->AddCut(VarManager::kEta, -0.9, 0.9);
    cut1->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    cut1->AddCut(VarManager::kITSchi2, 0.0, 36.0);
    AnalysisCut* cut2 = new AnalysisCut();
    cut2->AddCut(VarManager::kPt, 0.5, 3.0);
    fTrackCut->AddCut(cut1);
    //fTrackCut->AddCut(cut2);

    fMuonCut = new AnalysisCompositeCut(true);
    AnalysisCut kineMuonCut;
    kineMuonCut.AddCut(VarManager::kPt, 1.5, 10.0);
    fMuonCut->AddCut(&kineMuonCut);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>::iterator event,
               soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov> const& tracks,
               soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtended> const& muons)
  {
    // Reset the fgValues array
    // TODO: reseting will have to be done selectively, for example run-wise variables don't need to be reset every event, but just updated if the run changes
    //       The reset can be done selectively, using arguments in the ResetValues() function
    VarManager::ResetValues();

    VarManager::FillEvent<fgEventFillMap>(event);
    fHistMan->FillHistClass("Event_BeforeCuts", VarManager::fgValues); // automatically fill all the histograms in the class Event
    if (!fEventCut->IsSelected(VarManager::fgValues))
      return;
    fHistMan->FillHistClass("Event_AfterCuts", VarManager::fgValues);

    // loop over barrel tracks and store positive and negative tracks in separate arrays
    // TODO: use Partition initiaslized by vector of track indices when this will be available
    std::vector<std::decay_t<decltype(tracks)>::iterator> selectedTracksPos;
    std::vector<std::decay_t<decltype(tracks)>::iterator> selectedTracksNeg;
    for (auto& track : tracks) {

      VarManager::ResetValues(VarManager::kNEventWiseVariables, VarManager::kNMuonTrackVariables);

      VarManager::FillTrack<fgTrackFillMap>(track);
      fHistMan->FillHistClass("TrackBarrel_BeforeCuts", VarManager::fgValues);
      if (!fTrackCut->IsSelected(VarManager::fgValues))
        continue;
      fHistMan->FillHistClass("TrackBarrel_AfterCuts", VarManager::fgValues);

      if (track.charge() < 0)
        selectedTracksNeg.push_back(track);
      if (track.charge() > 0)
        selectedTracksPos.push_back(track);
    }

    // run the same event pairing for barrel tracks
    for (auto& tpos : selectedTracksPos) {
      for (auto& tneg : selectedTracksNeg) {
        VarManager::FillPair(tpos, tneg);
        fHistMan->FillHistClass("PairsBarrel", VarManager::fgValues);
      }
    }

    // loop over muon tracks
    std::vector<std::decay_t<decltype(muons)>::iterator> selectedMuonsPos;
    std::vector<std::decay_t<decltype(muons)>::iterator> selectedMuonsNeg;
    for (auto& muon : muons) {
      VarManager::FillTrack<fgMuonFillMap>(muon);
      fHistMan->FillHistClass("TrackMuon_BeforeCuts", VarManager::fgValues);
      if (!fMuonCut->IsSelected(VarManager::fgValues))
        continue;
      fHistMan->FillHistClass("TrackMuon_AfterCuts", VarManager::fgValues);

      if (muon.charge() < 0)
        selectedMuonsNeg.push_back(muon);
      if (muon.charge() > 0)
        selectedMuonsPos.push_back(muon);
    }

    // same event pairing for muons
    for (auto& tpos : selectedMuonsNeg) {
      for (auto& tneg : selectedMuonsPos) {
        VarManager::FillPair(tpos, tneg);
        fHistMan->FillHistClass("PairsMuon", VarManager::fgValues);
      }
    }
  }

  void DefineHistograms(TString histClasses)
  {
    const int kNRuns = 2;
    int runs[kNRuns] = {244918, 244919};
    TString runsStr;
    for (int i = 0; i < kNRuns; i++)
      runsStr += Form("%d;", runs[i]);
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

        double vtxXbinLims[10] = {0.055, 0.06, 0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 0.074, 0.08};
        double vtxYbinLims[7] = {0.31, 0.32, 0.325, 0.33, 0.335, 0.34, 0.35};
        double vtxZbinLims[13] = {-15.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0};
        double nContribbinLims[9] = {0.0, 100.0, 200.0, 400.0, 600.0, 1000.0, 1500.0, 2000.0, 4000.0};

        fHistMan->AddHistogram(classStr.Data(), "VtxX_VtxY_nonEqualBinning", "Vtx X vs Vtx Y", false, 9, vtxXbinLims, VarManager::kVtxX, 6, vtxYbinLims, VarManager::kVtxY); // THnF histogram with custom non-equal binning

        fHistMan->AddHistogram(classStr.Data(), "VtxZ_weights", "Vtx Z", false,
                               60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, 10, 0., 0., VarManager::kNothing,
                               "", "", "", VarManager::kNothing, VarManager::kVtxNcontrib); // TH1F histogram, filled with weights using the vtx n-contributors

        Int_t vars[4] = {VarManager::kVtxX, VarManager::kVtxY, VarManager::kVtxZ, VarManager::kVtxNcontrib};
        TArrayD binLimits[4];
        binLimits[0] = TArrayD(10, vtxXbinLims);
        binLimits[1] = TArrayD(7, vtxYbinLims);
        binLimits[2] = TArrayD(13, vtxZbinLims);
        binLimits[3] = TArrayD(9, nContribbinLims);
        fHistMan->AddHistogram(classStr.Data(), "vtxHisto", "n contrib vs (x,y,z)", 4, vars, binLimits);

        fHistMan->AddHistogram(classStr.Data(), "CentV0M_vtxZ", "CentV0M vs Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ, 20, 0., 100., VarManager::kCentVZERO); // TH2F histogram

        fHistMan->AddHistogram(classStr.Data(), "VtxChi2", "Vtx chi2", false, 100, 0.0, 100.0, VarManager::kVtxChi2); // TH1F histogram

        continue;
      } // end if(Event)

      if (classStr.Contains("Track")) {
        fHistMan->AddHistClass(classStr.Data());
        fHistMan->AddHistogram(classStr.Data(), "Pt", "p_{T} distribution", false, 200, 0.0, 20.0, VarManager::kPt);                                               // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "Eta", "#eta distribution", false, 500, -5.0, 5.0, VarManager::kEta);                                              // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "Phi_Eta", "#phi vs #eta distribution", false, 200, -5.0, 5.0, VarManager::kEta, 200, -6.3, 6.3, VarManager::kPhi); // TH2F histogram
        fHistMan->AddHistogram(classStr.Data(), "P", "p distribution", false, 200, 0.0, 20.0, VarManager::kP);                                                      // TH1F histogram
        fHistMan->AddHistogram(classStr.Data(), "Px", "p_{x} distribution", false, 200, 0.0, 20.0, VarManager::kPx);
        fHistMan->AddHistogram(classStr.Data(), "Py", "p_{y} distribution", false, 200, 0.0, 20.0, VarManager::kPy);
        fHistMan->AddHistogram(classStr.Data(), "Pz", "p_{z} distribution", false, 400, -20.0, 20.0, VarManager::kPz);

        if (classStr.Contains("Barrel")) {
          fHistMan->AddHistogram(classStr.Data(), "TPCdedx_pIN", "TPC dE/dx vs pIN", false, 100, 0.0, 20.0, VarManager::kPin,
                                 200, 0.0, 200., VarManager::kTPCsignal); // TH2F histogram

          fHistMan->AddHistogram(classStr.Data(), "Cov1Pt_Pt", "cov(1/pt,1/pt) vs p_{T} distribution", false, 20, 0.0, 5.0, VarManager::kPt, 100, 0.0, 1.0, VarManager::kTrackC1Pt21Pt2); // TH2F histogram
        }

        if (classStr.Contains("Muon")) {
          fHistMan->AddHistogram(classStr.Data(), "InvBendingMom", "", false, 100, 0.0, 1.0, VarManager::kMuonInvBendingMomentum);
          fHistMan->AddHistogram(classStr.Data(), "ThetaX", "", false, 100, -1.0, 1.0, VarManager::kMuonThetaX);
          fHistMan->AddHistogram(classStr.Data(), "ThetaY", "", false, 100, -2.0, 2.0, VarManager::kMuonThetaY);
          fHistMan->AddHistogram(classStr.Data(), "ZMu", "", false, 100, -30.0, 30.0, VarManager::kMuonZMu);
          fHistMan->AddHistogram(classStr.Data(), "BendingCoor", "", false, 100, 0.32, 0.35, VarManager::kMuonBendingCoor);
          fHistMan->AddHistogram(classStr.Data(), "NonBendingCoor", "", false, 100, 0.065, 0.07, VarManager::kMuonNonBendingCoor);
          fHistMan->AddHistogram(classStr.Data(), "Chi2", "", false, 100, 0.0, 200.0, VarManager::kMuonChi2);
          fHistMan->AddHistogram(classStr.Data(), "Chi2MatchTrigger", "", false, 100, 0.0, 20.0, VarManager::kMuonChi2MatchTrigger);
        }
      }

      if (classStr.Contains("Pairs")) {
        fHistMan->AddHistClass(classStr.Data());
        fHistMan->AddHistogram(classStr.Data(), "Mass", "", false, 100, 0.0, 5.0, VarManager::kMass);
      }
    } // end loop over histogram classes
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableReader>("table-reader")};
}
