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
#include "AnalysisDataModel/ReducedInfoTables.h"
#include "PWGDQCore/VarManager.h"
#include "PWGDQCore/HistogramManager.h"
#include "PWGDQCore/AnalysisCut.h"
#include "PWGDQCore/AnalysisCompositeCut.h"
#include <TH1F.h>
#include <TMath.h>
#include <THashList.h>
#include <TString.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
//using namespace o2::framework::expressions;
using namespace o2::aod;

// Some definitions
namespace o2::aod
{

namespace reducedevent
{
DECLARE_SOA_COLUMN(Category, category, int);
DECLARE_SOA_COLUMN(IsEventSelected, isEventSelected, int);
} // namespace reducedevent

namespace reducedtrack
{
DECLARE_SOA_COLUMN(IsMuonSelected, isMuonSelected, int);
} // namespace reducedtrack
namespace reducedpair
{
DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent);
DECLARE_SOA_COLUMN(Mass, mass, float);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Rap, rap, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Charge, charge, int);
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(Pmom, pmom, [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} // namespace reducedpair

DECLARE_SOA_TABLE(EventCuts, "AOD", "EVENTCUTS", reducedevent::IsEventSelected);
DECLARE_SOA_TABLE(EventCategories, "AOD", "EVENTCATEGORIES", reducedevent::Category);
DECLARE_SOA_TABLE(MuonTrackCuts, "AOD", "MUONTRACKCUTS", reducedtrack::IsMuonSelected);
} // namespace o2::aod

using MyEvents = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended>;
using MyEventsSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::EventCuts>;
using MyEventsVtxCov = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>;
using MyEventsVtxCovSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov, aod::EventCuts>;
using MyMuonTracks = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtended>;
using MyMuonTracksSelected = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtended, aod::MuonTrackCuts>;

void DefineHistograms(HistogramManager* histMan, TString histClasses);

// HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
//         a constexpr static bit map must be defined and sent as template argument
//        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
//        Additionally, one should make sure that the requested tables are actually provided in the process() function,
//       otherwise a compile time error will be thrown.
//        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
//           to automatically detect the object types transmitted to the VarManager
constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::ReducedEvent | VarManager::ObjTypes::ReducedEventExtended;
constexpr static uint32_t gkMuonFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackMuon;

struct EventSelection {
  Produces<aod::EventCuts> eventSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut* fEventCut;

  float* fValues;

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "Event_BeforeCuts;Event_AfterCuts;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                 // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    DefineCuts();
  }

  void DefineCuts()
  {
    fEventCut = new AnalysisCompositeCut(true);

    AnalysisCut* varCut = new AnalysisCut();
    varCut->AddCut(VarManager::kVtxZ, -10.0, 10.0);
    varCut->AddCut(VarManager::kIsMuonUnlikeLowPt7, 0.5, 1.5);

    fEventCut->AddCut(varCut);
    // TODO: Add more cuts, also enable cuts which are not easily possible via the VarManager (e.g. trigger selections)

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvents::iterator const& event)
  {
    // Reset the fValues array
    VarManager::ResetValues(0, VarManager::kNEventWiseVariables, fValues);

    VarManager::FillEvent<gkEventFillMap>(event, fValues);
    fHistMan->FillHistClass("Event_BeforeCuts", fValues); // automatically fill all the histograms in the class Event
    if (fEventCut->IsSelected(fValues)) {
      fHistMan->FillHistClass("Event_AfterCuts", fValues);
      eventSel(1);
    } else {
      eventSel(0);
    }
  }
};

struct MuonTrackSelection {
  Produces<aod::MuonTrackCuts> trackSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut* fMuonCut23;
  AnalysisCompositeCut* fMuonCut310;

  float* fValues; // array to be used by the VarManager

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "TrackMuon_BeforeCuts;TrackMuon_AfterCuts;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                         // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());                 // provide the list of required variables so that VarManager knows what to fill

    DefineCuts();
  }

  void DefineCuts()
  {
    fMuonCut23 = new AnalysisCompositeCut(true);
    AnalysisCut kineMuonCut23;
    kineMuonCut23.AddCut(VarManager::kPt, 1.0, 20);
    kineMuonCut23.AddCut(VarManager::kEta, -4.0, -2.5);
    kineMuonCut23.AddCut(VarManager::kMuonChi2, 0.0, 1e6);
    // The value of sigma_PDCA depends on the postion w.r.t the absorber. For 17.6 < RAbs < 26.5cm, sigma_PDCA = 99 cmxGeV/c. For 26.5 < RAbs < 89.5 cm, sigma_PDCA = 54
    // temporarily not applied
    //    kineMuonCut23.AddCut(VarManager::kMuonRAtAbsorberEnd, 17.6, 26.5);
    //    kineMuonCut23.AddCut(VarManager::kMuonPDca, 0.0, 594); // Cut is pDCA < 6*sigma_PDCA
    fMuonCut23->AddCut(&kineMuonCut23);

    fMuonCut310 = new AnalysisCompositeCut(true);
    AnalysisCut kineMuonCut310;
    kineMuonCut310.AddCut(VarManager::kPt, 1.0, 20);
    kineMuonCut310.AddCut(VarManager::kEta, -4.0, -2.5);
    kineMuonCut310.AddCut(VarManager::kMuonChi2, 0.0, 1e6);
    // The value of sigma_PDCA depends on the postion w.r.t the absorber. For 17.6 < RAbs < 26.5cm, sigma_PDCA = 99 cmxGeV/c. For 26.5 < RAbs < 89.5 cm, sigma_PDCA = 54
    // temporarily not applied
    //    kineMuonCut310.AddCut(VarManager::kMuonRAtAbsorberEnd, 26.5, 89.5);
    //    kineMuonCut310.AddCut(VarManager::kMuonPDca, 0.0, 324); // Cut is pDCA < 6*sigma_PDCA
    fMuonCut310->AddCut(&kineMuonCut310);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEventsSelected::iterator const& event, MyMuonTracks const& muons)
  {
    VarManager::ResetValues(0, VarManager::kNMuonTrackVariables, fValues);
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    for (auto& muon : muons) {
      //VarManager::ResetValues(VarManager::kNBarrelTrackVariables, VarManager::kNMuonTrackVariables, fValues);
      VarManager::FillTrack<gkMuonFillMap>(muon, fValues);
      fHistMan->FillHistClass("TrackMuon_BeforeCuts", fValues);

      if (fMuonCut23->IsSelected(fValues)) {
        trackSel(1);
        fHistMan->FillHistClass("TrackMuon_AfterCuts", fValues);
      } else if (fMuonCut310->IsSelected(fValues)) {
        trackSel(1);
        fHistMan->FillHistClass("TrackMuon_AfterCuts", fValues);
      } else {
        trackSel(0);
      }
    }
  }
};

struct DileptonMuMu {
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut* fDiMuonCut;
  //NOTE: one could define also a dilepton cut, but for now basic selections can be supported using Partition
  // NOTE TO THE NOTE: a dimuon cut is needed on the rapidity of the pair. So I added one. Hopefully this works

  float* fValues;

  Partition<MyMuonTracksSelected> posMuons = aod::reducedtrack::charge > 0 && aod::reducedtrack::isMuonSelected == 1;
  Partition<MyMuonTracksSelected> negMuons = aod::reducedtrack::charge < 0 && aod::reducedtrack::isMuonSelected == 1;

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "PairsMuonULS;PairsMuonLSpp;PairsMuonLSnn;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                         // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    DefineCuts();
  }

  void DefineCuts()
  {
    fDiMuonCut = new AnalysisCompositeCut(true);
    AnalysisCut* diMuonCut = new AnalysisCut();
    diMuonCut->AddCut(VarManager::kRap, 2.5, 4.0);
    fDiMuonCut->AddCut(diMuonCut);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEventsVtxCovSelected::iterator const& event, MyMuonTracksSelected const& tracks)
  {
    if (!event.isEventSelected()) {
      return;
    }
    // Reset the fValues array
    VarManager::ResetValues(0, VarManager::kNVars, fValues);

    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    // same event pairing for muons
    for (auto& tpos : posMuons) {
      for (auto& tneg : negMuons) {
        //dileptonList(event, VarManager::fgValues[VarManager::kMass], VarManager::fgValues[VarManager::kPt], VarManager::fgValues[VarManager::kEta], VarManager::fgValues[VarManager::kPhi], 1);
        VarManager::FillPair(tpos, tneg, nullptr, VarManager::kJpsiToMuMu);
        if (!fDiMuonCut->IsSelected(VarManager::fgValues)) {
          return;
        }
        fHistMan->FillHistClass("PairsMuonULS", VarManager::fgValues);
      }
      for (auto tpos2 = tpos + 1; tpos2 != posMuons.end(); ++tpos2) { // ++ pairs
        VarManager::FillPair(tpos, tpos2, nullptr, VarManager::kJpsiToMuMu);
        if (!fDiMuonCut->IsSelected(VarManager::fgValues)) {
          return;
        }
        fHistMan->FillHistClass("PairsMuonLSpp", VarManager::fgValues);
      }
    }
    for (auto tneg : negMuons) { // -- pairs
      for (auto tneg2 = tneg + 1; tneg2 != negMuons.end(); ++tneg2) {
        VarManager::FillPair(tneg, tneg2, nullptr, VarManager::kJpsiToMuMu);
        if (!fDiMuonCut->IsSelected(VarManager::fgValues)) {
          return;
        }
        fHistMan->FillHistClass("PairsMuonLSnn", VarManager::fgValues);
      }
    }
  }
};

void DefineHistograms(HistogramManager* histMan, TString histClasses)
{
  //
  // Define here the histograms for all the classes required in analysis.
  //  The histogram classes are provided in the histClasses string, separated by semicolon ";"
  //  The histogram classes and their components histograms are defined below depending on the name of the histogram class
  //
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
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "VtxZ", "Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ); // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "VtxZ_Run", "Vtx Z", true,
                            kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId, 60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, runsStr.Data());                                        // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "VtxX_VtxY", "Vtx X vs Vtx Y", false, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY);                                             // TH2F histogram
      histMan->AddHistogram(classStr.Data(), "VtxX_VtxY_VtxZ", "vtx x - y - z", false, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 60, -15.0, 15.0, VarManager::kVtxZ);     // TH3F histogram
      histMan->AddHistogram(classStr.Data(), "NContrib_vs_VtxZ_prof", "Vtx Z vs ncontrib", true, 30, -15.0, 15.0, VarManager::kVtxZ, 10, -1., 1., VarManager::kVtxNcontrib);                             // TProfile histogram
      histMan->AddHistogram(classStr.Data(), "VtxZ_vs_VtxX_VtxY_prof", "Vtx Z vs (x,y)", true, 100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 10, -1., 1., VarManager::kVtxZ); // TProfile2D histogram
      histMan->AddHistogram(classStr.Data(), "Ncontrib_vs_VtxZ_VtxX_VtxY_prof", "n-contrib vs (x,y,z)", true,
                            100, 0.055, 0.08, VarManager::kVtxX, 100, 0.31, 0.35, VarManager::kVtxY, 30, -15., 15., VarManager::kVtxZ,
                            "", "", "", VarManager::kVtxNcontrib); // TProfile3D

      double vtxXbinLims[10] = {0.055, 0.06, 0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 0.074, 0.08};
      double vtxYbinLims[7] = {0.31, 0.32, 0.325, 0.33, 0.335, 0.34, 0.35};
      double vtxZbinLims[13] = {-15.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0};
      double nContribbinLims[9] = {0.0, 100.0, 200.0, 400.0, 600.0, 1000.0, 1500.0, 2000.0, 4000.0};

      histMan->AddHistogram(classStr.Data(), "VtxX_VtxY_nonEqualBinning", "Vtx X vs Vtx Y", false, 9, vtxXbinLims, VarManager::kVtxX, 6, vtxYbinLims, VarManager::kVtxY); // THnF histogram with custom non-equal binning

      histMan->AddHistogram(classStr.Data(), "VtxZ_weights", "Vtx Z", false,
                            60, -15.0, 15.0, VarManager::kVtxZ, 10, 0., 0., VarManager::kNothing, 10, 0., 0., VarManager::kNothing,
                            "", "", "", VarManager::kNothing, VarManager::kVtxNcontrib); // TH1F histogram, filled with weights using the vtx n-contributors

      Int_t vars[4] = {VarManager::kVtxX, VarManager::kVtxY, VarManager::kVtxZ, VarManager::kVtxNcontrib};
      TArrayD binLimits[4];
      binLimits[0] = TArrayD(10, vtxXbinLims);
      binLimits[1] = TArrayD(7, vtxYbinLims);
      binLimits[2] = TArrayD(13, vtxZbinLims);
      binLimits[3] = TArrayD(9, nContribbinLims);
      histMan->AddHistogram(classStr.Data(), "vtxHisto", "n contrib vs (x,y,z)", 4, vars, binLimits);

      histMan->AddHistogram(classStr.Data(), "CentV0M_vtxZ", "CentV0M vs Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ, 20, 0., 100., VarManager::kCentVZERO); // TH2F histogram

      histMan->AddHistogram(classStr.Data(), "VtxChi2", "Vtx chi2", false, 100, 0.0, 100.0, VarManager::kVtxChi2); // TH1F histogram

      continue;
    } // end if(Event)

    if (classStr.Contains("Track")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "Pt", "p_{T} distribution", false, 200, 0.0, 20.0, VarManager::kPt);                                                // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "Eta", "#eta distribution", false, 19, -4.2, -2.3, VarManager::kEta);                                               // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "Phi_Eta", "#phi vs #eta distribution", false, 200, -5.0, 5.0, VarManager::kEta, 200, -6.3, 6.3, VarManager::kPhi); // TH2F histogram
      histMan->AddHistogram(classStr.Data(), "P", "p distribution", false, 200, 0.0, 20.0, VarManager::kP);                                                      // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "Px", "p_{x} distribution", false, 200, 0.0, 20.0, VarManager::kPx);
      histMan->AddHistogram(classStr.Data(), "Py", "p_{y} distribution", false, 200, 0.0, 20.0, VarManager::kPy);
      histMan->AddHistogram(classStr.Data(), "Pz", "p_{z} distribution", false, 400, -20.0, 20.0, VarManager::kPz);

      if (classStr.Contains("Muon")) {
        histMan->AddHistogram(classStr.Data(), "InvBendingMom", "", false, 100, 0.0, 1.0, VarManager::kMuonInvBendingMomentum);
        histMan->AddHistogram(classStr.Data(), "ThetaX", "", false, 100, -1.0, 1.0, VarManager::kMuonThetaX);
        histMan->AddHistogram(classStr.Data(), "ThetaY", "", false, 100, -2.0, 2.0, VarManager::kMuonThetaY);
        histMan->AddHistogram(classStr.Data(), "ZMu", "", false, 100, -30.0, 30.0, VarManager::kMuonZMu);
        histMan->AddHistogram(classStr.Data(), "BendingCoor", "", false, 100, 0.32, 0.35, VarManager::kMuonBendingCoor);
        histMan->AddHistogram(classStr.Data(), "NonBendingCoor", "", false, 100, 0.065, 0.07, VarManager::kMuonNonBendingCoor);
        histMan->AddHistogram(classStr.Data(), "Chi2", "", false, 100, 0.0, 200.0, VarManager::kMuonChi2);
        histMan->AddHistogram(classStr.Data(), "Chi2MatchTrigger", "", false, 100, 0.0, 20.0, VarManager::kMuonChi2MatchTrigger);
        histMan->AddHistogram(classStr.Data(), "RAtAbsorberEnd", "", false, 140, 10, 150, VarManager::kMuonRAtAbsorberEnd);
        histMan->AddHistogram(classStr.Data(), "p x dca", "", false, 700, 0.0, 700, VarManager::kMuonRAtAbsorberEnd);
      }
    }

    if (classStr.Contains("Pairs")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "Mass", "", false, 100, 2.0, 12, VarManager::kMass);
      histMan->AddHistogram(classStr.Data(), "Pt", "", false, 200, 0.0, 20.0, VarManager::kPt);
      histMan->AddHistogram(classStr.Data(), "Rapidity", "", false, 19, 2.0, 4.3, VarManager::kRap);
      histMan->AddHistogram(classStr.Data(), "Mass_Pt", "mass vs p_{T} distribution", false, 100, 0.0, 20.0, VarManager::kMass, 200, 0.0, 20.0, VarManager::kPt); // TH2F histogram
      histMan->AddHistogram(classStr.Data(), "Mass_Y", "mass vs y distribution", false, 100, 0.0, 20.0, VarManager::kMass, 19, 2.0, 4.3, VarManager::kRap);       // TH2F histogram
    }
  } // end loop over histogram classes
}

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelection>("my-event-selection"),
    adaptAnalysisTask<MuonTrackSelection>("muon-track-selection"),
    adaptAnalysisTask<DileptonMuMu>("dilepton-mumu")};
}
