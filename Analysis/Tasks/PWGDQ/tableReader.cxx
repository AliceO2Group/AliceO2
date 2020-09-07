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

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
//using namespace o2::framework::expressions;
using namespace o2::aod;

// Some definitions
namespace o2::aod
{
namespace reducedtrack
{
DECLARE_SOA_COLUMN(IsBarrelSelected, isBarrelSelected, int);
DECLARE_SOA_COLUMN(IsMuonSelected, isMuonSelected, int);
} // namespace reducedtrack
namespace reducedpair
{
DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent);
DECLARE_SOA_COLUMN(Mass, mass, float);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Charge, charge, int);
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(Pmom, pmom, [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} // namespace reducedpair

DECLARE_SOA_TABLE(BarrelTrackCuts, "AOD", "BARRELTRACKCUTS", reducedtrack::IsBarrelSelected);
DECLARE_SOA_TABLE(MuonTrackCuts, "AOD", "MUONTRACKCUTS", reducedtrack::IsMuonSelected);
DECLARE_SOA_TABLE(Dileptons, "AOD", "DILEPTON", reducedtrack::ReducedEventId, reducedpair::Mass, reducedpair::Pt, reducedpair::Eta, reducedpair::Phi, reducedpair::Charge,
                  reducedpair::Px<reducedpair::Pt, reducedpair::Phi>, reducedpair::Py<reducedpair::Pt, reducedpair::Phi>,
                  reducedpair::Pz<reducedpair::Pt, reducedpair::Eta>, reducedpair::Pmom<reducedpair::Pt, reducedpair::Eta>);
using Dilepton = Dileptons::iterator;
} // namespace o2::aod

using MyEvent = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended>::iterator;
using MyEventVtxCov = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>::iterator;
using MyBarrelTracks = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov, aod::ReducedTracksBarrelPID>;
using MyBarrelTracksSelected = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov, aod::ReducedTracksBarrelPID, aod::BarrelTrackCuts>;
using MyMuonTracks = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtended>;
using MyMuonTracksSelected = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtended, aod::MuonTrackCuts>;

void DefineHistograms(o2::framework::OutputObj<HistogramManager> histMan, TString histClasses);

// HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
//         a constexpr static bit map must be defined and sent as template argument
//        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
//        Additionally, one should make sure that the requested tables are actually provided in the process() function,
//       otherwise a compile time error will be thrown.
//        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
//           to automatically detect the object types transmitted to the VarManager
constexpr static uint32_t fgEventFillMap = VarManager::ObjTypes::ReducedEvent | VarManager::ObjTypes::ReducedEventExtended;
constexpr static uint32_t fgTrackFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackBarrel | VarManager::ObjTypes::ReducedTrackBarrelCov | VarManager::ObjTypes::ReducedTrackBarrelPID;
constexpr static uint32_t fgMuonFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackMuon;

struct BarrelTrackSelection {
  Produces<aod::BarrelTrackCuts> trackSel;
  OutputObj<HistogramManager> fHistMan{"output"};
  AnalysisCompositeCut* fTrackCut;

  float* fValues; // array to be used by the VarManager

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan.setObject(new HistogramManager("analysisHistos", "aa", VarManager::kNVars));
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "TrackBarrel_BeforeCuts;TrackBarrel_AfterCuts;");                                                                                       // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                                                                                                                   // provide the list of required variables so that VarManager knows what to fill

    DefineCuts();
  }

  void DefineCuts()
  {
    fTrackCut = new AnalysisCompositeCut(true); // true: use AND
    AnalysisCut* cut1 = new AnalysisCut();
    cut1->AddCut(VarManager::kPt, 1.5, 20.0);
    cut1->AddCut(VarManager::kEta, -0.9, 0.9);
    cut1->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    cut1->AddCut(VarManager::kITSchi2, 0.0, 36.0);
    cut1->AddCut(VarManager::kITSncls, 2.5, 7.5);
    cut1->AddCut(VarManager::kTPCncls, 69.5, 159.5);
    AnalysisCut* cut2 = new AnalysisCut();
    cut2->AddCut(VarManager::kPt, 0.5, 3.0);
    fTrackCut->AddCut(cut1);
    //fTrackCut->AddCut(cut2);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvent event, MyBarrelTracks const& tracks)
  {
    for (int i = 0; i < VarManager::kNVars; ++i)
      fValues[i] = -9999.0f;
    // fill event information which might be needed in histograms that combine track and event properties
    VarManager::FillEvent<fgEventFillMap>(event, fValues);

    for (auto& track : tracks) {
      for (int i = VarManager::kNEventWiseVariables; i < VarManager::kNMuonTrackVariables; ++i)
        fValues[i] = -9999.0f;
      VarManager::FillTrack<fgTrackFillMap>(track, fValues);
      fHistMan->FillHistClass("TrackBarrel_BeforeCuts", fValues);

      if (fTrackCut->IsSelected(fValues)) {
        trackSel(1);
        fHistMan->FillHistClass("TrackBarrel_AfterCuts", fValues);
      } else
        trackSel(0);
    }
  }
};

struct MuonTrackSelection {
  Produces<aod::MuonTrackCuts> trackSel;
  OutputObj<HistogramManager> fHistMan{"output"};
  AnalysisCompositeCut* fTrackCut;

  float* fValues;

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan.setObject(new HistogramManager("analysisHistos", "aa", VarManager::kNVars));
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "TrackMuon_BeforeCuts;TrackMuon_AfterCuts;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                         // provide the list of required variables so that VarManager knows what to fill

    DefineCuts();
  }

  void DefineCuts()
  {
    fTrackCut = new AnalysisCompositeCut(true);
    AnalysisCut kineMuonCut;
    kineMuonCut.AddCut(VarManager::kPt, 1.5, 10.0);
    fTrackCut->AddCut(&kineMuonCut);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvent event, MyMuonTracks const& muons)
  {
    for (int i = 0; i < VarManager::kNVars; ++i)
      fValues[i] = -9999.0f;
    VarManager::FillEvent<fgEventFillMap>(event, fValues);

    for (auto& muon : muons) {
      for (int i = VarManager::kNBarrelTrackVariables; i < VarManager::kNMuonTrackVariables; ++i)
        fValues[i] = -9999.0f;
      VarManager::FillTrack<fgMuonFillMap>(muon, fValues);
      fHistMan->FillHistClass("TrackMuon_BeforeCuts", fValues);

      if (fTrackCut->IsSelected(fValues)) {
        trackSel(1);
        fHistMan->FillHistClass("TrackMuon_AfterCuts", fValues);
      } else
        trackSel(0);
    }
  }
};

struct TableReader {
  Produces<aod::Dileptons> dileptonList;
  OutputObj<HistogramManager> fHistMan{"output"};
  AnalysisCompositeCut* fEventCut;
  //NOTE: one could define also a dilepton cut, but for now basic selections can be supported using Partition

  Partition<MyBarrelTracksSelected> posTracks = aod::reducedtrack::charge > 0 && aod::reducedtrack::isBarrelSelected == 1;
  Partition<MyBarrelTracksSelected> negTracks = aod::reducedtrack::charge < 0 && aod::reducedtrack::isBarrelSelected == 1;
  Partition<MyMuonTracksSelected> posMuons = aod::reducedtrack::charge > 0 && aod::reducedtrack::isMuonSelected == 1;
  Partition<MyMuonTracksSelected> negMuons = aod::reducedtrack::charge < 0 && aod::reducedtrack::isMuonSelected == 1;

  void init(o2::framework::InitContext&)
  {
    VarManager::SetDefaultVarNames();
    fHistMan.setObject(new HistogramManager("analysisHistos", "aa", VarManager::kNVars));
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "Event_BeforeCuts;Event_AfterCuts;PairsBarrelPM;PairsBarrelPP;PairsBarrelMM;PairsMuon;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                                                                     // provide the list of required variables so that VarManager knows what to fill

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

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEventVtxCov event, MyBarrelTracksSelected const& tracks, MyMuonTracksSelected const& muons)
  {
    // Reset the VarManager::fgValues array
    // The reset can be done selectively, using arguments in the ResetValues() function
    VarManager::ResetValues();

    VarManager::FillEvent<fgEventFillMap>(event);
    fHistMan->FillHistClass("Event_BeforeCuts", VarManager::fgValues); // automatically fill all the histograms in the class Event
    if (!fEventCut->IsSelected(VarManager::fgValues))
      return;
    fHistMan->FillHistClass("Event_AfterCuts", VarManager::fgValues);

    // Run the same event pairing for barrel tracks
    // TODO: Use combinations() when this will work for Partitions
    /* e.g.
     * for (auto& [tpos, tneg] : combinations(posTracks, negTracks)) {
      VarManager::FillPair(tpos, tneg);
      fHistMan->FillHistClass("PairsBarrelPM", VarManager::fgValues);
    }
    */

    for (auto tpos : posTracks) {
      for (auto tneg : negTracks) { // +- pairs
        VarManager::FillPair(tpos, tneg);
        dileptonList(event, VarManager::fgValues[VarManager::kMass], VarManager::fgValues[VarManager::kPt], VarManager::fgValues[VarManager::kEta], VarManager::fgValues[VarManager::kPhi], 0);
        fHistMan->FillHistClass("PairsBarrelPM", VarManager::fgValues);
      }
      for (auto tpos2 = tpos + 1; tpos2 != posTracks.end(); ++tpos2) { // ++ pairs
        VarManager::FillPair(tpos, tpos2);
        dileptonList(event, VarManager::fgValues[VarManager::kMass], VarManager::fgValues[VarManager::kPt], VarManager::fgValues[VarManager::kEta], VarManager::fgValues[VarManager::kPhi], 2);
        fHistMan->FillHistClass("PairsBarrelPP", VarManager::fgValues);
      }
    }
    for (auto tneg : negTracks) { // -- pairs
      for (auto tneg2 = tneg + 1; tneg2 != negTracks.end(); ++tneg2) {
        VarManager::FillPair(tneg, tneg2);
        dileptonList(event, VarManager::fgValues[VarManager::kMass], VarManager::fgValues[VarManager::kPt], VarManager::fgValues[VarManager::kEta], VarManager::fgValues[VarManager::kPhi], -2);
        fHistMan->FillHistClass("PairsBarrelMM", VarManager::fgValues);
      }
    }

    // same event pairing for muons
    for (auto& tpos : posMuons) {
      for (auto& tneg : negMuons) {
        //dileptonList(event, VarManager::fgValues[VarManager::kMass], VarManager::fgValues[VarManager::kPt], VarManager::fgValues[VarManager::kEta], VarManager::fgValues[VarManager::kPhi], 1);
        VarManager::FillPair(tpos, tneg);
        fHistMan->FillHistClass("PairsMuon", VarManager::fgValues);
      }
    }
  }
};

struct DileptonHadronAnalysis {
  //
  // This task combines dilepton candidates with a track and could be used for example
  //  in analyses with the dilepton as one of the decay products of a higher mass resonance (e.g. B0 -> Jpsi + K)
  //    or in dilepton + hadron correlations, etc.
  // It requires the TableReader task to be in the workflow and produce the dilepton table
  //
  OutputObj<HistogramManager> fHistMan{"output"};
  AnalysisCompositeCut* fHadronCut;
  //NOTE: no cut has been included for dileptons because that can be controlled via the TableReader task and the partition below

  // use two values array to avoid mixing up the quantities
  float* fValuesDilepton;
  float* fValuesHadron;

  Partition<aod::Dileptons> selDileptons = aod::reducedpair::charge == 0 && aod::reducedpair::mass > 2.92f && aod::reducedpair::mass<3.16f && aod::reducedpair::pt> 5.0f;

  constexpr static uint32_t fgDileptonFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::Pair;

  void init(o2::framework::InitContext&)
  {
    fValuesDilepton = new float[VarManager::kNVars];
    fValuesHadron = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan.setObject(new HistogramManager("analysisHistos", "aa", VarManager::kNVars));
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "DileptonsSelected;HadronsSelected;DileptonHadronInvMass;DileptonHadronCorrelation"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                                                                 // provide the list of required variables so that VarManager knows what to fill

    DefineCuts();
  }

  void DefineCuts()
  {
    fHadronCut = new AnalysisCompositeCut(true); // true: use AND
    AnalysisCut* cut1 = new AnalysisCut();
    cut1->AddCut(VarManager::kPt, 4.0, 20.0);
    cut1->AddCut(VarManager::kEta, -0.9, 0.9);
    cut1->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    cut1->AddCut(VarManager::kITSchi2, 0.0, 36.0);
    cut1->AddCut(VarManager::kITSncls, 2.5, 7.5);
    cut1->AddCut(VarManager::kTPCncls, 69.5, 159.5);
    fHadronCut->AddCut(cut1);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvent event, MyBarrelTracks const& hadrons, aod::Dileptons const& dileptons)
  {
    for (int i = 0; i < VarManager::kNVars; ++i)
      fValuesHadron[i] = -9999.0f;
    // fill event information which might be needed in histograms that combine track/pair and event properties
    VarManager::FillEvent<fgEventFillMap>(event, fValuesHadron);

    // loop once over dileptons for QA purposes
    for (auto dilepton : selDileptons) {
      for (int i = VarManager::kNEventWiseVariables; i < VarManager::kNMuonTrackVariables; ++i)
        fValuesDilepton[i] = -9999.0f;
      VarManager::FillTrack<fgDileptonFillMap>(dilepton, fValuesDilepton);
      fHistMan->FillHistClass("DileptonsSelected", fValuesDilepton);
    }

    // loop over hadrons
    for (auto& hadron : hadrons) {
      for (int i = VarManager::kNEventWiseVariables; i < VarManager::kNMuonTrackVariables; ++i)
        fValuesHadron[i] = -9999.0f;
      VarManager::FillTrack<fgTrackFillMap>(hadron, fValuesHadron);
      if (!fHadronCut->IsSelected(fValuesHadron))
        continue;

      fHistMan->FillHistClass("HadronsSelected", fValuesHadron);

      for (auto dilepton : selDileptons) {
        // TODO: At the moment there is no check on whether this hadron is one of the dilepton daughters
        VarManager::FillDileptonHadron(dilepton, hadron, fValuesHadron);
        fHistMan->FillHistClass("DileptonHadronInvMass", fValuesHadron);
        fHistMan->FillHistClass("DileptonHadronCorrelation", fValuesHadron);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<BarrelTrackSelection>("barrel-track-selection"),
    adaptAnalysisTask<MuonTrackSelection>("muon-track-selection"),
    adaptAnalysisTask<TableReader>("table-reader"),
    adaptAnalysisTask<DileptonHadronAnalysis>("dilepton-hadron")};
}

void DefineHistograms(o2::framework::OutputObj<HistogramManager> histMan, TString histClasses)
{
  //
  // Define here the histograms for all the classes required in analysis.
  //  The histogram classes are provided in the histClasses string, separated by semicolon ";"
  //  The histogram classes and their components histograms are defined below depending on the name of the histogram class
  //
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
      histMan->AddHistogram(classStr.Data(), "Eta", "#eta distribution", false, 500, -5.0, 5.0, VarManager::kEta);                                               // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "Phi_Eta", "#phi vs #eta distribution", false, 200, -5.0, 5.0, VarManager::kEta, 200, -6.3, 6.3, VarManager::kPhi); // TH2F histogram
      histMan->AddHistogram(classStr.Data(), "P", "p distribution", false, 200, 0.0, 20.0, VarManager::kP);                                                      // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "Px", "p_{x} distribution", false, 200, 0.0, 20.0, VarManager::kPx);
      histMan->AddHistogram(classStr.Data(), "Py", "p_{y} distribution", false, 200, 0.0, 20.0, VarManager::kPy);
      histMan->AddHistogram(classStr.Data(), "Pz", "p_{z} distribution", false, 400, -20.0, 20.0, VarManager::kPz);

      if (classStr.Contains("Barrel")) {
        histMan->AddHistogram(classStr.Data(), "TPCncls", "Number of cluster in TPC", false, 160, -0.5, 159.5, VarManager::kTPCncls); // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "ITSncls", "Number of cluster in ITS", false, 8, -0.5, 7.5, VarManager::kITSncls);     // TH1F histogram
        //for TPC PID
        histMan->AddHistogram(classStr.Data(), "TPCdedx_pIN", "TPC dE/dx vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, 0.0, 200., VarManager::kTPCsignal);                    // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCnSigmaEl_pIN", "TPC dE/dx n#sigma_{e} vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, -10, +10, VarManager::kTPCnSigmaEl);   // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCnSigmaEl_Eta", "TPC dE/dx n#sigma_{e} vs #eta", false, 20, -1, +1, VarManager::kEta, 200, -10, +10, VarManager::kTPCnSigmaEl);      // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCnSigmaPi_pIN", "TPC dE/dx n#sigma_{#pi} vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, -10, +10, VarManager::kTPCnSigmaPi); // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCnSigmaPi_Eta", "TPC dE/dx n#sigma_{#pi} vs #eta", false, 20, -1, +1, VarManager::kEta, 200, -10, +10, VarManager::kTPCnSigmaPi);    // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCnSigmaKa_pIN", "TPC dE/dx n#sigma_{K} vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, -10, +10, VarManager::kTPCnSigmaKa);   // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCnSigmaKa_Eta", "TPC dE/dx n#sigma_{K} vs #eta", false, 20, -1, +1, VarManager::kEta, 200, -10, +10, VarManager::kTPCnSigmaKa);      // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCnSigmaPr_pIN", "TPC dE/dx n#sigma_{p} vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, -10, +10, VarManager::kTPCnSigmaPr);   // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCnSigmaPr_Eta", "TPC dE/dx n#sigma_{p} vs #eta", false, 20, -1, +1, VarManager::kEta, 200, -10, +10, VarManager::kTPCnSigmaPr);      // TH2F histogram

        //for TOF PID
        histMan->AddHistogram(classStr.Data(), "TOFbeta_pIN", "TOF #beta vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 120, 0.0, 1.2, VarManager::kTOFbeta);                       // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TOFnSigmaEl_pIN", "TOF #beta n#sigma_{e} vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, -10, +10, VarManager::kTOFnSigmaEl);   // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TOFnSigmaEl_Eta", "TOF #beta n#sigma_{e} vs #eta", false, 20, -1, +1, VarManager::kEta, 200, -10, +10, VarManager::kTOFnSigmaEl);      // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TOFnSigmaPi_pIN", "TOF #beta n#sigma_{#pi} vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, -10, +10, VarManager::kTOFnSigmaPi); // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TOFnSigmaPi_Eta", "TOF #beta n#sigma_{#pi} vs #eta", false, 20, -1, +1, VarManager::kEta, 200, -10, +10, VarManager::kTOFnSigmaPi);    // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TOFnSigmaKa_pIN", "TOF #beta n#sigma_{K} vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, -10, +10, VarManager::kTOFnSigmaKa);   // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TOFnSigmaKa_Eta", "TOF #beta n#sigma_{K} vs #eta", false, 20, -1, +1, VarManager::kEta, 200, -10, +10, VarManager::kTOFnSigmaKa);      // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TOFnSigmaPr_pIN", "TOF #beta n#sigma_{p} vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, -10, +10, VarManager::kTOFnSigmaPr);   // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TOFnSigmaPr_Eta", "TOF #beta n#sigma_{p} vs #eta", false, 20, -1, +1, VarManager::kEta, 200, -10, +10, VarManager::kTOFnSigmaPr);      // TH2F histogram

        histMan->AddHistogram(classStr.Data(), "Cov1Pt_Pt", "cov(1/pt,1/pt) vs p_{T} distribution", false, 20, 0.0, 5.0, VarManager::kPt, 100, 0.0, 1.0, VarManager::kTrackC1Pt21Pt2); // TH2F histogram
      }

      if (classStr.Contains("Muon")) {
        histMan->AddHistogram(classStr.Data(), "InvBendingMom", "", false, 100, 0.0, 1.0, VarManager::kMuonInvBendingMomentum);
        histMan->AddHistogram(classStr.Data(), "ThetaX", "", false, 100, -1.0, 1.0, VarManager::kMuonThetaX);
        histMan->AddHistogram(classStr.Data(), "ThetaY", "", false, 100, -2.0, 2.0, VarManager::kMuonThetaY);
        histMan->AddHistogram(classStr.Data(), "ZMu", "", false, 100, -30.0, 30.0, VarManager::kMuonZMu);
        histMan->AddHistogram(classStr.Data(), "BendingCoor", "", false, 100, 0.32, 0.35, VarManager::kMuonBendingCoor);
        histMan->AddHistogram(classStr.Data(), "NonBendingCoor", "", false, 100, 0.065, 0.07, VarManager::kMuonNonBendingCoor);
        histMan->AddHistogram(classStr.Data(), "Chi2", "", false, 100, 0.0, 200.0, VarManager::kMuonChi2);
        histMan->AddHistogram(classStr.Data(), "Chi2MatchTrigger", "", false, 100, 0.0, 20.0, VarManager::kMuonChi2MatchTrigger);
      }
    }

    if (classStr.Contains("DileptonsSelected")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "Mass_Pt", "", false, 100, 0.0, 5.0, VarManager::kMass, 100, 0.0, 20.0, VarManager::kPt);
    }

    if (classStr.Contains("HadronsSelected")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "Eta_Pt", "", false, 20, -1.0, 1.0, VarManager::kEta, 100, 0.0, 20.0, VarManager::kPt);
      histMan->AddHistogram(classStr.Data(), "Eta_Phi", "", false, 20, -1.0, 1.0, VarManager::kEta, 100, -8.0, 8.0, VarManager::kPhi);
    }

    if (classStr.Contains("Pairs")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "Mass_Pt", "", false, 100, 0.0, 5.0, VarManager::kMass, 100, 0.0, 20.0, VarManager::kPt);
    }

    if (classStr.Contains("DileptonHadronInvMass")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "Mass_Pt", "", false, 40, 0.0, 20.0, VarManager::kPairMass, 40, 0.0, 20.0, VarManager::kPairPt);
    }

    if (classStr.Contains("DileptonHadronCorrelation")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "DeltaEta_DeltaPhi", "", false, 20, -2.0, 2.0, VarManager::kDeltaEta, 50, -8.0, 8.0, VarManager::kDeltaPhi);
      histMan->AddHistogram(classStr.Data(), "DeltaEta_DeltaPhiSym", "", false, 20, -2.0, 2.0, VarManager::kDeltaEta, 50, -8.0, 8.0, VarManager::kDeltaPhiSym);
    }
  } // end loop over histogram classes
}
