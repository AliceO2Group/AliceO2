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
// Contact: daiki.sekihata@cern.ch
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/ReducedInfoTables.h"
#include "AnalysisCore/VarManager.h"
#include "AnalysisCore/HistogramManager.h"
#include "AnalysisCore/AnalysisCut.h"
#include "AnalysisCore/AnalysisCompositeCut.h"
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
using namespace o2::framework::expressions;
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
DECLARE_SOA_COLUMN(IsBarrelSelected, isBarrelSelected, uint8_t);
} // namespace reducedtrack

namespace reducedpair
{
DECLARE_SOA_INDEX_COLUMN(ReducedEvent, reducedevent);
DECLARE_SOA_COLUMN(Mass, mass, float);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Charge, charge, int);
DECLARE_SOA_COLUMN(FilterMap, filterMap, uint8_t);
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(Pmom, pmom, [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} // namespace reducedpair

DECLARE_SOA_TABLE(EventCuts, "AOD", "EVENTCUTS", reducedevent::IsEventSelected);
DECLARE_SOA_TABLE(EventCategories, "AOD", "EVENTCATEGORIES", reducedevent::Category);
DECLARE_SOA_TABLE(BarrelTrackCuts, "AOD", "BARRELTRACKCUTS", reducedtrack::IsBarrelSelected);
DECLARE_SOA_TABLE(Dileptons, "AOD", "DILEPTON", reducedpair::ReducedEventId, reducedpair::Mass, reducedpair::Pt, reducedpair::Eta, reducedpair::Phi, reducedpair::Charge, reducedpair::FilterMap,
                  reducedpair::Px<reducedpair::Pt, reducedpair::Phi>, reducedpair::Py<reducedpair::Pt, reducedpair::Phi>,
                  reducedpair::Pz<reducedpair::Pt, reducedpair::Eta>, reducedpair::Pmom<reducedpair::Pt, reducedpair::Eta>);
using Dilepton = Dileptons::iterator;
} // namespace o2::aod

using MyEvents = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended>;
using MyEventsSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::EventCuts>;
using MyEventsVtxCov = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>;
using MyEventsVtxCovSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov, aod::EventCuts>;
using MyBarrelTracks = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov, aod::ReducedTracksBarrelPID>;
using MyBarrelTracksSelected = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov, aod::ReducedTracksBarrelPID, aod::BarrelTrackCuts>;

void DefineHistograms(HistogramManager* histMan, TString histClasses);

// HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
//         a constexpr static bit map must be defined and sent as template argument
//        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
//        Additionally, one should make sure that the requested tables are actually provided in the process() function,
//       otherwise a compile time error will be thrown.
//        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
//           to automatically detect the object types transmitted to the VarManager
constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::ReducedEvent | VarManager::ObjTypes::ReducedEventExtended;
constexpr static uint32_t gkTrackFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackBarrel | VarManager::ObjTypes::ReducedTrackBarrelCov | VarManager::ObjTypes::ReducedTrackBarrelPID;

int gNTrackCuts = 3;

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
    varCut->AddCut(VarManager::kIsINT7, 0.5, 1.5);
    varCut->AddCut(VarManager::kVtxNcontrib, 0.5, 1e+10);
    varCut->AddCut(VarManager::kCentVZERO, 0.0, 101);

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

struct BarrelTrackSelection {
  Produces<aod::BarrelTrackCuts> trackSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  std::vector<AnalysisCompositeCut> fTrackCuts;

  float* fValues; // array to be used by the VarManager

  void init(o2::framework::InitContext&)
  {
    DefineCuts();

    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString cutNames = "TrackBarrel_BeforeCuts;";
    for (int i = 0; i < gNTrackCuts; i++) {
      cutNames += Form("TrackBarrel_%s;", fTrackCuts[i].GetName());
    }

    DefineHistograms(fHistMan, cutNames.Data());     // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
  }

  void DefineCuts()
  {
    AnalysisCut* commonCuts = new AnalysisCut();
    commonCuts->AddCut(VarManager::kPt, 0.2, 10.0);
    commonCuts->AddCut(VarManager::kEta, -0.8, 0.8);
    commonCuts->AddCut(VarManager::kIsSPDany, 0.5, 1.5);
    commonCuts->AddCut(VarManager::kIsITSrefit, 0.5, 1.5);
    commonCuts->AddCut(VarManager::kIsTPCrefit, 0.5, 1.5);
    commonCuts->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    commonCuts->AddCut(VarManager::kITSchi2, 0.0, 5.0);
    commonCuts->AddCut(VarManager::kTPCncls, 70.0, 160.);
    commonCuts->AddCut(VarManager::kTrackDCAxy, -1.0, 1.0);
    commonCuts->AddCut(VarManager::kTrackDCAz, -3.0, 3.0);
    commonCuts->AddCut(VarManager::kTPCsignal, 70.0, 100.0);
    commonCuts->AddCut(VarManager::kTPCsignal, 75.0, 100.0, false, VarManager::kPin, 2.0, 10.0, false);

    AnalysisCut* pidCut_TPChadrej = new AnalysisCut();
    //pidCut_TPChadrej->AddCut(VarManager::kTPCsignal, 70.0, 100.0, false);
    //pidCut_TPChadrej->AddCut(VarManager::kTPCsignal, 75.0, 100.0, false, VarManager::kPin, 2.0, 10.0, false);
    //TF1* cutLow1 = new TF1("cutLow1", "pol1", 0., 10.);
    //cutLow1->SetParameters(130., -40.0);
    //pidCut1->AddCut(VarManager::kTPCsignal, cutLow1, 100.0, false, VarManager::kPin, 0.5, 3.0);

    //reject pion
    TF1* f1maxPi = new TF1("f1maxPi", "[0]+[1]*x", 0, 10);
    f1maxPi->SetParameters(85, -50);
    pidCut_TPChadrej->AddCut(VarManager::kTPCsignal, 70, f1maxPi, true, VarManager::kPin, 0.0, 0.4, false); //exclude = false

    //reject kaon
    TF1* f1minKa = new TF1("f1minKa", "[0]+[1]*x", 0, 10);
    f1minKa->SetParameters(220, -300);
    TF1* f1maxKa = new TF1("f1maxKa", "[0]+[1]*x", 0, 10);
    f1maxKa->SetParameters(182.5, -150);
    pidCut_TPChadrej->AddCut(VarManager::kTPCsignal, f1minKa, f1maxKa, true, VarManager::kPin, 0.4, 0.8, false); //exclude = false

    //reject protoon
    TF1* f1minPr = new TF1("f1minPr", "[0]+[1]*x", 0, 10);
    f1minPr->SetParameters(170, -100);
    TF1* f1maxPr = new TF1("f1maxPr", "[0]+[1]*x", 0, 10);
    f1maxPr->SetParameters(175, -75);
    pidCut_TPChadrej->AddCut(VarManager::kTPCsignal, f1minPr, f1maxPr, true, VarManager::kPin, 0.8, 1.4, false); //exclude = false

    AnalysisCut* pidCut_TOFrecovery = new AnalysisCut();
    pidCut_TOFrecovery->AddCut(VarManager::kTOFbeta, 0.99, 1.01);

    AnalysisCompositeCut trackCut1("cut1", "cut1", true); // true: use AND
    trackCut1.AddCut(commonCuts);
    trackCut1.AddCut(pidCut_TPChadrej);

    AnalysisCompositeCut trackCut2("cut2", "cut2", true); // true: use AND
    trackCut2.AddCut(commonCuts);
    trackCut2.AddCut(pidCut_TOFrecovery);

    //AnalysisCompositeCut trackCut3and("cut3and", "cut3and", true); // true: use AND
    //trackCut3and.AddCut(pidCut_TPChadrej);
    //trackCut3and.AddCut(pidCut_TOFrecovery);

    //AnalysisCompositeCut trackCut3or("cut3or", "cut3or", false); // false: use OR
    //trackCut3or.AddCut(pidCut_TPChadrej);
    //trackCut3or.AddCut(pidCut_TOFrecovery);

    //AnalysisCompositeCut tmptrackCut3and("tmpcut3and", "tmpcut3and", true); // true: use AND
    //tmptrackCut3and.AddCut(commonCuts);
    //tmptrackCut3and.AddCut(&trackCut3and);

    //AnalysisCompositeCut tmptrackCut3or("tmpcut3or", "tmpcut3or", true); // true: use AND
    //tmptrackCut3or.AddCut(commonCuts);
    //tmptrackCut3or.AddCut(&trackCut3or);

    AnalysisCompositeCut trackCut3("cut3", "cut3", false); // false: use OR
    trackCut3.AddCut(&trackCut1);
    trackCut3.AddCut(&trackCut2);

    fTrackCuts.push_back(trackCut1);
    fTrackCuts.push_back(trackCut2);
    //fTrackCuts.push_back(tmptrackCut3and);
    //fTrackCuts.push_back(tmptrackCut3or);
    fTrackCuts.push_back(trackCut3);

    //gNTrackCuts = fTrackCuts.size();
    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEventsSelected::iterator const& event, MyBarrelTracks const& tracks)
  {
    VarManager::ResetValues(0, VarManager::kNBarrelTrackVariables, fValues);
    // fill event information which might be needed in histograms that combine track and event properties
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    uint8_t filterMap = uint8_t(0);

    trackSel.reserve(tracks.size());

    for (auto& track : tracks) {
      filterMap = uint8_t(0);
      VarManager::FillTrack<gkTrackFillMap>(track, fValues);
      if (event.isEventSelected()) {
        fHistMan->FillHistClass("TrackBarrel_BeforeCuts", fValues);
      }

      int i = 0;
      for (auto cut = fTrackCuts.begin(); cut != fTrackCuts.end(); ++cut, ++i) {
        if ((*cut).IsSelected(fValues)) {
          filterMap |= (uint8_t(1) << i);
          fHistMan->FillHistClass(Form("TrackBarrel_%s", (*cut).GetName()), fValues);
        }
      }
      trackSel(filterMap);
    }
  }
};

struct DileptonEE {
  Produces<aod::Dileptons> dileptonList;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  //NOTE: one could define also a dilepton cut, but for now basic selections can be supported using Partition

  float* fValues;

  Partition<MyBarrelTracksSelected> posTracks = aod::reducedtrack::charge > 0 && aod::reducedtrack::isBarrelSelected > uint8_t(0);
  Partition<MyBarrelTracksSelected> negTracks = aod::reducedtrack::charge < 0 && aod::reducedtrack::isBarrelSelected > uint8_t(0);

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString histNames = "";
    for (int i = 0; i < gNTrackCuts; i++) {
      histNames += Form("PairsBarrelPM_cut%d;PairsBarrelPP_cut%d;PairsBarrelMM_cut%d;", i + 1, i + 1, i + 1);
    }

    DefineHistograms(fHistMan, histNames.Data());    // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
  }

  void process(MyEventsVtxCovSelected::iterator const& event, MyBarrelTracksSelected const& tracks)
  {
    if (!event.isEventSelected()) {
      return;
    }
    // Reset the fValues array
    VarManager::ResetValues(0, VarManager::kNVars, fValues);

    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    // Run the same event pairing for barrel tracks
    // TODO: Use combinations() when this will work for Partitions
    /* e.g.
     * for (auto& [tpos, tneg] : combinations(posTracks, negTracks)) {
      VarManager::FillPair(tpos, tneg);
      fHistMan->FillHistClass("PairsBarrelPM", VarManager::fgValues);
    }
    */
    uint8_t filter = 0;
    for (auto tpos : posTracks) {
      for (auto tneg : negTracks) { // +- pairs
        filter = tpos.isBarrelSelected() & tneg.isBarrelSelected();
        if (!filter) { // the tracks must have at least one filter bit in common to continue
          continue;
        }
        VarManager::FillPair(tpos, tneg, fValues);
        dileptonList(event, fValues[VarManager::kMass], fValues[VarManager::kPt], fValues[VarManager::kEta], fValues[VarManager::kPhi], 0, filter);
        for (int i = 0; i < gNTrackCuts; ++i) {
          if (filter & (uint8_t(1) << i)) {
            fHistMan->FillHistClass(Form("PairsBarrelPM_cut%d", i + 1), fValues);
          }
        }
      }
      for (auto tpos2 = tpos + 1; tpos2 != posTracks.end(); ++tpos2) { // ++ pairs
        filter = tpos.isBarrelSelected() & tpos2.isBarrelSelected();
        if (!filter) { // the tracks must have at least one filter bit in common to continue
          continue;
        }
        VarManager::FillPair(tpos, tpos2, fValues);
        dileptonList(event, fValues[VarManager::kMass], fValues[VarManager::kPt], fValues[VarManager::kEta], fValues[VarManager::kPhi], 2, filter);
        for (int i = 0; i < gNTrackCuts; ++i) {
          if (filter & (uint8_t(1) << i)) {
            fHistMan->FillHistClass(Form("PairsBarrelPP_cut%d", i + 1), fValues);
          }
        }
      }
    }
    for (auto tneg : negTracks) { // -- pairs
      for (auto tneg2 = tneg + 1; tneg2 != negTracks.end(); ++tneg2) {
        filter = tneg.isBarrelSelected() & tneg2.isBarrelSelected();
        if (!filter) { // the tracks must have at least one filter bit in common to continue
          continue;
        }
        VarManager::FillPair(tneg, tneg2, fValues);
        dileptonList(event, fValues[VarManager::kMass], fValues[VarManager::kPt], fValues[VarManager::kEta], fValues[VarManager::kPhi], -2, filter);
        for (int i = 0; i < gNTrackCuts; ++i) {
          if (filter & (uint8_t(1) << i)) {
            fHistMan->FillHistClass(Form("PairsBarrelMM_cut%d", i + 1), fValues);
          }
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelection>("my-event-selection"),
    adaptAnalysisTask<BarrelTrackSelection>("barrel-track-selection"),
    adaptAnalysisTask<DileptonEE>("dilepton-ee")};
}

void DefineHistograms(HistogramManager* histMan, TString histClasses)
{
  //
  // Define here the histograms for all the classes required in analysis.
  //  The histogram classes are provided in the histClasses string, separated by semicolon ";"
  //  The histogram classes and their components histograms are defined below depending on the name of the histogram class
  //
  const int kNRuns = 135;
  int runs[kNRuns] = {
    244917, 244918, 244975, 244980, 244982, 244983, 245064, 245066, 245068, 245145,
    245146, 245151, 245152, 245231, 245232, 245259, 245343, 245345, 245346, 245347,
    245349, 245353, 245396, 245397, 245401, 245407, 245409, 245411, 245439, 245441,
    245446, 245450, 245452, 245454, 245496, 245497, 245501, 245504, 245505, 245507,
    245535, 245540, 245542, 245543, 245544, 245545, 245554, 245683, 245692, 245700,
    245702, 245705, 245829, 245831, 245833, 245923, 245949, 245952, 245954, 245963,
    246001, 246003, 246012, 246036, 246037, 246042, 246048, 246049, 246052, 246053,
    246087, 246089, 246113, 246115, 246148, 246151, 246152, 246153, 246178, 246180,
    246181, 246182, 246185, 246217, 246222, 246225, 246271, 246272, 246275, 246276,
    246391, 246392, 246424, 246428, 246431, 246434, 246487, 246488, 246493, 246495,
    246675, 246676, 246750, 246751, 246757, 246758, 246759, 246760, 246763, 246765,
    246766, 246804, 246805, 246807, 246808, 246809, 246810, 246844, 246845, 246846,
    246847, 246851, 246865, 246867, 246870, 246871, 246928, 246945, 246948, 246980,
    246982, 246984, 246989, 246991, 246994};
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
      histMan->AddHistogram(classStr.Data(), "VtxZ", "Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ);                                                          // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "CentV0M_vtxZ", "CentV0M vs Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ, 20, 0., 100., VarManager::kCentVZERO); // TH2F histogram
      continue;
    } // end if(Event)

    if (classStr.Contains("Track")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "Pt", "p_{T} distribution", false, 100, 0.0, 10.0, VarManager::kPt); // TH1F histogram
      //histMan->AddHistogram(classStr.Data(), "Eta", "#eta distribution", false, 200, -1.0, 1.0, VarManager::kEta);                                               // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "Eta_Phi", "#phi vs #eta distribution", false, 100, 0, 6.3, VarManager::kPhi, 200, -1.0, 1.0, VarManager::kEta); // TH2F histogram

      if (classStr.Contains("Barrel")) {
        histMan->AddHistogram(classStr.Data(), "TPCncls", "Number of cluster in TPC", false, 160, -0.5, 159.5, VarManager::kTPCncls); // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "TPCncls_Run", "Number of cluster in TPC", true, kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId,
                              10, -0.5, 159.5, VarManager::kTPCncls, 10, 0., 1., VarManager::kNothing, runsStr.Data());           // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "ITSncls", "Number of cluster in ITS", false, 8, -0.5, 7.5, VarManager::kITSncls); // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "ITSchi2", "ITS chi2", false, 100, 0.0, 50.0, VarManager::kITSchi2);               // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "IsITSrefit", "", false, 2, -0.5, 1.5, VarManager::kIsITSrefit);                   // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "IsTPCrefit", "", false, 2, -0.5, 1.5, VarManager::kIsTPCrefit);                   // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "IsSPDany", "", false, 2, -0.5, 1.5, VarManager::kIsSPDany);                       // TH1F histogram
        //for TPC PID
        histMan->AddHistogram(classStr.Data(), "TPCdedx_pIN", "TPC dE/dx vs pIN", false, 1000, 0.0, 10.0, VarManager::kPin, 200, 0.0, 200., VarManager::kTPCsignal); // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCchi2", "TPC chi2", false, 100, 0.0, 10.0, VarManager::kTPCchi2);                                                  // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "DCAxy", "DCAxy", false, 100, -5.0, 5.0, VarManager::kTrackDCAxy);                                                    // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "DCAz", "DCAz", false, 100, -5.0, 5.0, VarManager::kTrackDCAz);                                                       // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "TOFbeta_pIN", "TOF #beta vs pIN", false, 1000, 0.0, 10.0, VarManager::kPin, 100, 0.2, 1.2, VarManager::kTOFbeta);    // TH2F histogram
      }
    }

    if (classStr.Contains("Pairs")) {
      histMan->AddHistClass(classStr.Data());
      //histMan->AddHistogram(classStr.Data(), "Mass_Pt_Cent", "", false, 125, 0.0, 5.0, VarManager::kMass, 20, 0.0, 20.0, VarManager::kPt, 10, 0.0, 100.0, VarManager::kCentVZERO);
      histMan->AddHistogram(classStr.Data(), "Mass_Pt", "m_{ee} vs. p_{T,ee};m_{ee} (GeV/c^{2});p_{T,ee} (GeV/c)", false, 500, 0.0, 5.0, VarManager::kMass, 200, 0.0, 20.0, VarManager::kPt);
    }

  } // end loop over histogram classes
}
