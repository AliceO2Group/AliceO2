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
DECLARE_SOA_COLUMN(FilterMap, filterMap, uint8_t);
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float pt, float phi) -> float { return pt * std::cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float pt, float phi) -> float { return pt * std::sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float pt, float eta) -> float { return pt * std::sinh(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(Pmom, pmom, [](float pt, float eta) -> float { return pt * std::cosh(eta); });
} // namespace reducedpair

DECLARE_SOA_TABLE(EventCuts, "AOD", "EVENTCUTS", reducedevent::IsEventSelected);
DECLARE_SOA_TABLE(EventCategories, "AOD", "EVENTCATEGORIES", reducedevent::Category);
DECLARE_SOA_TABLE(BarrelTrackCuts, "AOD", "BARRELTRACKCUTS", reducedtrack::IsBarrelSelected);
DECLARE_SOA_TABLE(MuonTrackCuts, "AOD", "MUONTRACKCUTS", reducedtrack::IsMuonSelected);
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
constexpr static uint32_t gkTrackFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackBarrel | VarManager::ObjTypes::ReducedTrackBarrelCov | VarManager::ObjTypes::ReducedTrackBarrelPID;
constexpr static uint32_t gkMuonFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackMuon;

int gNTrackCuts = 2;

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
    varCut->AddCut(VarManager::kCentVZERO, 0.0, 10.0);

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
    commonCuts->AddCut(VarManager::kPt, 1.0, 20.0);
    commonCuts->AddCut(VarManager::kTPCsignal, 70.0, 100.0);
    commonCuts->AddCut(VarManager::kEta, -0.9, 0.9);
    commonCuts->AddCut(VarManager::kIsSPDany, 0.5, 1.5);
    commonCuts->AddCut(VarManager::kIsITSrefit, 0.5, 1.5);
    commonCuts->AddCut(VarManager::kIsTPCrefit, 0.5, 1.5);
    commonCuts->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    commonCuts->AddCut(VarManager::kITSchi2, 0.1, 36.0);
    commonCuts->AddCut(VarManager::kTPCncls, 100.0, 161.);
    commonCuts->AddCut(VarManager::kTrackDCAxy, -1.0, 1.0);
    commonCuts->AddCut(VarManager::kTrackDCAz, -3.0, 3.0);

    AnalysisCut* pidCut1 = new AnalysisCut();
    TF1* cutLow1 = new TF1("cutLow1", "pol1", 0., 10.);
    cutLow1->SetParameters(130., -40.0);
    pidCut1->AddCut(VarManager::kTPCsignal, cutLow1, 100.0, false, VarManager::kPin, 0.5, 3.0);

    AnalysisCut* pidCut2 = new AnalysisCut();
    pidCut2->AddCut(VarManager::kTPCsignal, 73.0, 100.0);

    AnalysisCompositeCut trackCut1("cut1", "cut1", true); // true: use AND
    trackCut1.AddCut(commonCuts);
    trackCut1.AddCut(pidCut1);

    AnalysisCompositeCut trackCut2("cut2", "cut2", true); // true: use AND
    trackCut2.AddCut(commonCuts);
    trackCut2.AddCut(pidCut1);
    trackCut2.AddCut(pidCut2);

    fTrackCuts.push_back(trackCut1);
    fTrackCuts.push_back(trackCut2);

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

struct MuonTrackSelection {
  Produces<aod::MuonTrackCuts> trackSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut* fTrackCut;

  float* fValues;

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "TrackMuon_BeforeCuts;TrackMuon_AfterCuts;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                         // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());

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

  void process(MyEventsSelected::iterator const& event, MyMuonTracks const& muons)
  {
    VarManager::ResetValues(0, VarManager::kNMuonTrackVariables, fValues);
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    for (auto& muon : muons) {
      //VarManager::ResetValues(VarManager::kNBarrelTrackVariables, VarManager::kNMuonTrackVariables, fValues);
      VarManager::FillTrack<gkMuonFillMap>(muon, fValues);
      //if(event.isEventSelected())
      fHistMan->FillHistClass("TrackMuon_BeforeCuts", fValues);

      if (fTrackCut->IsSelected(fValues)) {
        trackSel(1);
        //if(event.isEventSelected())
        fHistMan->FillHistClass("TrackMuon_AfterCuts", fValues);
      } else {
        trackSel(0);
      }
    }
  }
};

struct TableReader {
  Produces<aod::Dileptons> dileptonList;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  //NOTE: one could define also a dilepton cut, but for now basic selections can be supported using Partition

  float* fValues;

  Partition<MyBarrelTracksSelected> posTracks = aod::reducedtrack::charge > 0 && aod::reducedtrack::isBarrelSelected > uint8_t(0);
  Partition<MyBarrelTracksSelected> negTracks = aod::reducedtrack::charge < 0 && aod::reducedtrack::isBarrelSelected > uint8_t(0);
  Partition<MyMuonTracksSelected> posMuons = aod::reducedtrack::charge > 0 && aod::reducedtrack::isMuonSelected == 1;
  Partition<MyMuonTracksSelected> negMuons = aod::reducedtrack::charge < 0 && aod::reducedtrack::isMuonSelected == 1;

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

  //void process(soa::Filtered<MyEventsVtxCovSelected>::iterator const& event, MyBarrelTracksSelected const& tracks, MyMuonTracksSelected const& muons)
  void process(MyEventsVtxCovSelected::iterator const& event, MyBarrelTracksSelected const& tracks, MyMuonTracksSelected const& muons)
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

    // same event pairing for muons
    for (auto& tpos : posMuons) {
      for (auto& tneg : negMuons) {
        //dileptonList(event, VarManager::fgValues[VarManager::kMass], VarManager::fgValues[VarManager::kPt], VarManager::fgValues[VarManager::kEta], VarManager::fgValues[VarManager::kPhi], 1);
        VarManager::FillPair(tpos, tneg, fValues);
        fHistMan->FillHistClass("PairsMuon", fValues);
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
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut* fHadronCut; // TODO: this cut will be moved into the barrel/muon track selection task
  //NOTE: no cut has been included for dileptons because that can be controlled via the TableReader task and the partition below

  // use two values array to avoid mixing up the quantities
  float* fValuesDilepton;
  float* fValuesHadron;

  Filter eventFilter = aod::reducedevent::isEventSelected == 1;

  Partition<aod::Dileptons> selDileptons = aod::reducedpair::charge == 0 && aod::reducedpair::mass > 2.92f && aod::reducedpair::mass<3.16f && aod::reducedpair::pt> 5.0f;

  constexpr static uint32_t fgDileptonFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::Pair;

  void init(o2::framework::InitContext&)
  {
    fValuesDilepton = new float[VarManager::kNVars];
    fValuesHadron = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "DileptonsSelected;HadronsSelected;DileptonHadronInvMass;DileptonHadronCorrelation"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    DefineCuts();
  }

  void DefineCuts()
  {
    fHadronCut = new AnalysisCompositeCut(true); // true: use AND
    AnalysisCut* cut1 = new AnalysisCut();
    cut1->AddCut(VarManager::kPt, 4.0, 20.0);
    cut1->AddCut(VarManager::kEta, -0.9, 0.9);
    cut1->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    cut1->AddCut(VarManager::kITSchi2, 0.1, 36.0);
    cut1->AddCut(VarManager::kITSncls, 2.5, 7.5);
    cut1->AddCut(VarManager::kTPCncls, 69.5, 159.5);
    fHadronCut->AddCut(cut1);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(soa::Filtered<MyEventsVtxCovSelected>::iterator const& event, MyBarrelTracks const& hadrons, aod::Dileptons const& dileptons)
  {
    VarManager::ResetValues(0, VarManager::kNVars, fValuesHadron);
    VarManager::ResetValues(0, VarManager::kNVars, fValuesDilepton);
    // fill event information which might be needed in histograms that combine track/pair and event properties
    VarManager::FillEvent<gkEventFillMap>(event, fValuesHadron);
    VarManager::FillEvent<gkEventFillMap>(event, fValuesDilepton); // TODO: check if needed (just for dilepton QA which might be depending on event wise variables)

    // loop once over dileptons for QA purposes
    for (auto dilepton : selDileptons) {
      VarManager::FillTrack<fgDileptonFillMap>(dilepton, fValuesDilepton);
      fHistMan->FillHistClass("DileptonsSelected", fValuesDilepton);
    }

    // loop over hadrons
    for (auto& hadron : hadrons) {
      VarManager::FillTrack<gkTrackFillMap>(hadron, fValuesHadron);
      if (!fHadronCut->IsSelected(fValuesHadron)) { // TODO: this will be moved to a partition when the selection will be done in the barrel/muon track selection
        continue;
      }

      fHistMan->FillHistClass("HadronsSelected", fValuesHadron);

      for (auto dilepton : selDileptons) {
        // TODO: At the moment there is no check on whether this hadron is one of the dilepton daughters!
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
    adaptAnalysisTask<EventSelection>("my-event-selection"),
    adaptAnalysisTask<BarrelTrackSelection>("barrel-track-selection"),
    adaptAnalysisTask<MuonTrackSelection>("muon-track-selection"),
    adaptAnalysisTask<TableReader>("table-reader"),
    adaptAnalysisTask<DileptonHadronAnalysis>("dilepton-hadron")

  };
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
      histMan->AddHistogram(classStr.Data(), "VtxZ", "Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ); // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "CentV0M_vtxZ", "CentV0M vs Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ, 20, 0., 100., VarManager::kCentVZERO); // TH2F histogram
      continue;
    } // end if(Event)

    if (classStr.Contains("Track")) {
      histMan->AddHistClass(classStr.Data());
      histMan->AddHistogram(classStr.Data(), "Pt", "p_{T} distribution", false, 200, 0.0, 20.0, VarManager::kPt);                                                // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "Eta", "#eta distribution", false, 500, -5.0, 5.0, VarManager::kEta);                                               // TH1F histogram
      histMan->AddHistogram(classStr.Data(), "Phi_Eta", "#phi vs #eta distribution", false, 200, -5.0, 5.0, VarManager::kEta, 200, -6.3, 6.3, VarManager::kPhi); // TH2F histogram

      if (classStr.Contains("Barrel")) {
        histMan->AddHistogram(classStr.Data(), "TPCncls", "Number of cluster in TPC", false, 160, -0.5, 159.5, VarManager::kTPCncls); // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "TPCncls_Run", "Number of cluster in TPC", true, kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId,
                              10, -0.5, 159.5, VarManager::kTPCncls, 10, 0., 1., VarManager::kNothing, runsStr.Data());               // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "ITSncls", "Number of cluster in ITS", false, 8, -0.5, 7.5, VarManager::kITSncls);     // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "ITSchi2", "ITS chi2", false, 100, 0.0, 50.0, VarManager::kITSchi2);                   // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "IsITSrefit", "", false, 2, -0.5, 1.5, VarManager::kIsITSrefit);                       // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "IsTPCrefit", "", false, 2, -0.5, 1.5, VarManager::kIsTPCrefit);                       // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "IsSPDany", "", false, 2, -0.5, 1.5, VarManager::kIsSPDany);                           // TH1F histogram
        //for TPC PID
        histMan->AddHistogram(classStr.Data(), "TPCdedx_pIN", "TPC dE/dx vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, 0.0, 200., VarManager::kTPCsignal);                    // TH2F histogram
        histMan->AddHistogram(classStr.Data(), "TPCchi2", "TPC chi2", false, 100, 0.0, 10.0, VarManager::kTPCchi2);                                                                    // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "DCAxy", "DCAxy", false, 100, -3.0, 3.0, VarManager::kTrackDCAxy); // TH1F histogram
        histMan->AddHistogram(classStr.Data(), "DCAz", "DCAz", false, 100, -5.0, 5.0, VarManager::kTrackDCAz);    // TH1F histogram
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
      histMan->AddHistogram(classStr.Data(), "Mass_Pt_Cent", "", false, 125, 0.0, 5.0, VarManager::kMass, 20, 0.0, 20.0, VarManager::kPt, 10, 0.0, 100.0, VarManager::kCentVZERO);
      histMan->AddHistogram(classStr.Data(), "Mass_Pt", "", false, 125, 0.0, 5.0, VarManager::kMass, 100, 0.0, 20.0, VarManager::kPt);
      histMan->AddHistogram(classStr.Data(), "Mass", "", false, 125, 0.0, 5.0, VarManager::kMass);
      histMan->AddHistogram(classStr.Data(), "Mass_Run", "", true, kNRuns, 0.5, 0.5 + kNRuns, VarManager::kRunId, 10, 0.0, 5.0, VarManager::kMass,
                            10, 0., 1., VarManager::kNothing, runsStr.Data());
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
