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
#include "PWGDQCore/VarManager.h"
#include "PWGDQCore/HistogramManager.h"
#include "PWGDQCore/AnalysisCut.h"
#include "PWGDQCore/AnalysisCompositeCut.h"
#include "PWGDQCore/CutsLibrary.h"
#include "PWGDQCore/HistogramsLibrary.h"
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

DECLARE_SOA_TABLE(EventCuts, "AOD", "EVENTCUTS", reducedevent::IsEventSelected);
DECLARE_SOA_TABLE(EventCategories, "AOD", "EVENTCATEGORIES", reducedevent::Category);
DECLARE_SOA_TABLE(BarrelTrackCuts, "AOD", "BARRELTRACKCUTS", reducedtrack::IsBarrelSelected);
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

struct EventSelection {
  Produces<aod::EventCuts> eventSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut fEventCut{true};

  float* fValues;
  Configurable<std::string> fConfigCuts{"cfgEventCuts", "eventStandard", "Comma separated list of event cuts; multiple cuts are applied with a logical AND"};

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "analysisHistos", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "Event_BeforeCuts;Event_AfterCuts;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                 // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    DefineCuts();
  }

  void DefineCuts()
  {
    // default cut is "eventStandard" (kINT7 and vtxZ selection)
    TString cutNamesStr = fConfigCuts.value;
    if (!cutNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(cutNamesStr.Tokenize(","));
      for (int icut = 0; icut < objArray->GetEntries(); ++icut) {
        fEventCut.AddCut(dqcuts::GetAnalysisCut(objArray->At(icut)->GetName()));
      }
    }

    // NOTE: Additional cuts to those specified via the Configurable may still be added
    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvents::iterator const& event)
  {
    // Reset the fValues array
    VarManager::ResetValues(0, VarManager::kNEventWiseVariables, fValues);

    VarManager::FillEvent<gkEventFillMap>(event, fValues);
    fHistMan->FillHistClass("Event_BeforeCuts", fValues); // automatically fill all the histograms in the class Event
    if (fEventCut.IsSelected(fValues)) {
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
  Configurable<std::string> fConfigCuts{"cfgBarrelTrackCuts", "lmeePID_TPChadrej,lmeePID_TOFrec,lmeePID_TPChadrejTOFrec", "Comma separated list of barrel track cuts"};

  void init(o2::framework::InitContext&)
  {
    DefineCuts();

    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "analysisHistos", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString cutNames = "TrackBarrel_BeforeCuts;";
    for (int i = 0; i < fTrackCuts.size(); i++) {
      cutNames += Form("TrackBarrel_%s;", fTrackCuts[i].GetName());
    }

    DefineHistograms(fHistMan, cutNames.Data());     // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
  }

  void DefineCuts()
  {
    // available cuts: jpsiKineAndQuality, jpsiPID1, jpsiPID2
    TString cutNamesStr = fConfigCuts.value;
    if (!cutNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(cutNamesStr.Tokenize(","));
      for (int icut = 0; icut < objArray->GetEntries(); ++icut) {
        fTrackCuts.push_back(*dqcuts::GetCompositeCut(objArray->At(icut)->GetName()));
      }
    }

    // NOTE: Additional cuts to those specified via the Configurable may still be added

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
  std::vector<AnalysisCompositeCut> fPairCuts;
  //NOTE: one could define also a dilepton cut, but for now basic selections can be supported using Partition

  float* fValues;

  Partition<MyBarrelTracksSelected> posTracks = aod::reducedtrack::sign > 0 && aod::reducedtrack::isBarrelSelected > uint8_t(0);
  Partition<MyBarrelTracksSelected> negTracks = aod::reducedtrack::sign < 0 && aod::reducedtrack::isBarrelSelected > uint8_t(0);

  Configurable<std::string> fConfigTrackCuts{"cfgBarrelTrackCuts", "lmeePID_TPChadrej,lmeePID_TOFrec,lmeePID_TPChadrejTOFrec", "Comma separated list of barrel track cuts"};
  Configurable<std::string> fConfigPairCuts{"cfgPairCuts", "pairMassLow", "Comma separated list of pair cuts"};

  int fNTrackCuts;
  int fNPairCuts;
  TObjArray* fTrkCutsNameArray;

  void DefineCuts()
  {
    // available pair cuts in CutsLibrary: pairNoCut,pairMassLow,pairJpsi,pairPsi2S,pairUpsilon,pairJpsiPtLow1, pairJpsiPtLow2
    TString pairCutNamesStr = fConfigPairCuts.value;
    std::unique_ptr<TObjArray> objArray(pairCutNamesStr.Tokenize(","));
    fNPairCuts = objArray->GetEntries();
    if (fNPairCuts) {
      for (int icut = 0; icut < fNPairCuts; ++icut) {
        fPairCuts.push_back(*dqcuts::GetCompositeCut(objArray->At(icut)->GetName()));
      }
    }

    // NOTE: Additional cuts to those specified via the Configurable may still be added

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void init(o2::framework::InitContext&)
  {
    DefineCuts();
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "analysisHistos", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    // configure histograms
    TString trackCutNamesStr = fConfigTrackCuts.value;
    fTrkCutsNameArray = trackCutNamesStr.Tokenize(",");
    fNTrackCuts = fTrkCutsNameArray->GetEntries();
    TString histNames = "";
    for (int i = 0; i < fNTrackCuts; i++) {
      histNames += Form("PairsBarrelPM_%s;PairsBarrelPP_%s;PairsBarrelMM_%s;", fTrkCutsNameArray->At(i)->GetName(), fTrkCutsNameArray->At(i)->GetName(), fTrkCutsNameArray->At(i)->GetName());
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
        for (int i = 0; i < fNTrackCuts; ++i) {
          if (filter & (uint8_t(1) << i)) {
            fHistMan->FillHistClass(Form("PairsBarrelPM_%s", fTrkCutsNameArray->At(i)->GetName()), fValues);
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
        for (int i = 0; i < fNTrackCuts; ++i) {
          if (filter & (uint8_t(1) << i)) {
            fHistMan->FillHistClass(Form("PairsBarrelPP_%s", fTrkCutsNameArray->At(i)->GetName()), fValues);
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
        for (int i = 0; i < fNTrackCuts; ++i) {
          if (filter & (uint8_t(1) << i)) {
            fHistMan->FillHistClass(Form("PairsBarrelMM_%s", fTrkCutsNameArray->At(i)->GetName()), fValues);
          }
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelection>(cfgc, TaskName{"my-event-selection"}),
    adaptAnalysisTask<BarrelTrackSelection>(cfgc, TaskName{"barrel-track-selection"}),
    adaptAnalysisTask<DileptonEE>(cfgc, TaskName{"dilepton-ee"}),

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
  VarManager::SetRunNumbers(kNRuns, runs);

  std::unique_ptr<TObjArray> objArray(histClasses.Tokenize(";"));
  for (Int_t iclass = 0; iclass < objArray->GetEntries(); ++iclass) {
    TString classStr = objArray->At(iclass)->GetName();
    histMan->AddHistClass(classStr.Data());

    if (classStr.Contains("Event")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "event", "trigger,vtxPbPb");
    }

    if (classStr.Contains("Track")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "track", "tpc,its,tpcpid,tofpid,dca");
    }

    if (classStr.Contains("Pairs")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "pair_lmee");
    }

  } // end loop over histogram classes
}
