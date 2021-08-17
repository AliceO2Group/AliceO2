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
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
// Analysis task for processing O2::DQ MC skimmed AODs
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
#include "PWGDQCore/HistogramsLibrary.h"
#include "PWGDQCore/CutsLibrary.h"
#include "PWGDQCore/MCSignal.h"
#include "PWGDQCore/MCSignalLibrary.h"
#include <TH1F.h>
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
DECLARE_SOA_COLUMN(IsEventSelected, isEventSelected, int);
} // namespace reducedevent

namespace reducedtrack
{
DECLARE_SOA_COLUMN(IsBarrelSelected, isBarrelSelected, uint8_t);
} // namespace reducedtrack

DECLARE_SOA_TABLE(EventCuts, "AOD", "EVENTCUTS", reducedevent::IsEventSelected);
DECLARE_SOA_TABLE(BarrelTrackCuts, "AOD", "BARRELTRACKCUTS", reducedtrack::IsBarrelSelected);
} // namespace o2::aod

using MyEvents = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsMC>;
using MyEventsSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::EventCuts, aod::ReducedEventsMC>;
using MyEventsVtxCov = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov, aod::ReducedEventsMC>;
using MyEventsVtxCovSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov, aod::EventCuts, aod::ReducedEventsMC>;

using MyBarrelTracks = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov, aod::ReducedTracksBarrelPID, aod::ReducedTracksBarrelLabels>;
using MyBarrelTracksSelected = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov, aod::ReducedTracksBarrelPID, aod::BarrelTrackCuts, aod::ReducedTracksBarrelLabels>;

void DefineHistograms(HistogramManager* histMan, TString histClasses);

// HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
//         a constexpr static bit map must be defined and sent as template argument
//        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
//        Additionally, one should make sure that the requested tables are actually provided in the process() function,
//       otherwise a compile time error will be thrown.
//        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
//           to automatically detect the object types transmitted to the VarManager

constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::ReducedEvent | VarManager::ObjTypes::ReducedEventExtended | VarManager::ObjTypes::ReducedEventMC;
constexpr static uint32_t gkTrackFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackBarrel | VarManager::ObjTypes::ReducedTrackBarrelCov | VarManager::ObjTypes::ReducedTrackBarrelPID;
constexpr static uint32_t gkParticleMCFillMap = VarManager::ObjTypes::ParticleMC;

struct DQEventSelection {
  Produces<aod::EventCuts> eventSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut* fEventCut;

  float* fValues;

  Configurable<std::string> fConfigEventCuts{"cfgEventCuts", "eventStandard", "Event selection"};

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
    TString eventCutStr = fConfigEventCuts.value;
    fEventCut->AddCut(dqcuts::GetAnalysisCut(eventCutStr.Data()));

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

struct DQBarrelTrackSelection {
  Produces<aod::BarrelTrackCuts> trackSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  std::vector<AnalysisCompositeCut> fTrackCuts;

  float* fValues;                   // array to be used by the VarManager
  std::vector<MCSignal> fMCSignals; // list of signals to be checked

  Configurable<std::string> fConfigElectronCuts{"cfgElectronCuts", "jpsiPID1", "Comma separated list of barrel track cuts"};
  Configurable<std::string> fConfigTrackMCSignals{"cfgBarrelTrackMCSignals", "", "Comma separated list of MC signals"};

  void init(o2::framework::InitContext&)
  {
    DefineCuts();

    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString configSigNamesStr = fConfigTrackMCSignals.value;
    std::unique_ptr<TObjArray> sigNamesArray(configSigNamesStr.Tokenize(","));

    // Configure histogram classes for each track cut;
    // Add histogram classes for each track cut and for each requested MC signal (reconstructed tracks with MC truth)
    TString histClasses = "TrackBarrel_BeforeCuts;";
    for (int i = 0; i < fTrackCuts.size(); i++) {
      histClasses += Form("TrackBarrel_%s;", fTrackCuts[i].GetName());
      for (int isig = 0; isig < sigNamesArray->GetEntries(); ++isig) {
        MCSignal* sig = o2::aod::dqmcsignals::GetMCSignal(sigNamesArray->At(isig)->GetName());
        if (sig) {
          if (sig->GetNProngs() != 1) { // NOTE: only 1 prong signals
            continue;
          }
          fMCSignals.push_back(*sig);
          histClasses += Form("TrackBarrel_%s_%s;", fTrackCuts[i].GetName(), sigNamesArray->At(isig)->GetName());
        }
      }
    }

    DefineHistograms(fHistMan, histClasses.Data());  // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
  }

  void DefineCuts()
  {
    // Defines track cuts for both the dielectron candidates and also for the dilepton - hadron task
    TString cutNamesStr = fConfigElectronCuts.value;
    if (!cutNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(cutNamesStr.Tokenize(","));
      for (int icut = 0; icut < objArray->GetEntries(); ++icut) {
        fTrackCuts.push_back(*dqcuts::GetCompositeCut(objArray->At(icut)->GetName()));
      }
    }
    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEventsSelected::iterator const& event, MyBarrelTracks const& tracks, ReducedMCTracks const& tracksMC)
  {
    cout << "Event ######################################" << endl;
    VarManager::ResetValues(0, VarManager::kNMCParticleVariables, fValues);
    // fill event information which might be needed in histograms that combine track and event properties
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    uint8_t filterMap = uint8_t(0);
    trackSel.reserve(tracks.size());

    for (auto& track : tracks) {
      filterMap = uint8_t(0);
      cout << "track ===================================" << endl;
      VarManager::FillTrack<gkTrackFillMap>(track, fValues);
      cout << "after fill ===================================" << endl;
      auto mctrack = track.reducedMCTrack();
      cout << "MC matched track " << mctrack.index() << "/" << mctrack.globalIndex() << ", pdgCode " << mctrack.pdgCode() << ", pt rec/gen " << track.pt() << "/" << mctrack.pt() << ", MCflags " << mctrack.mcReducedFlags() << endl;
      VarManager::FillTrack<gkParticleMCFillMap>(mctrack, fValues);
      cout << "after fill MC ===================================" << endl;
      fHistMan->FillHistClass("TrackBarrel_BeforeCuts", fValues);

      int i = 0;
      for (auto cut = fTrackCuts.begin(); cut != fTrackCuts.end(); cut++, i++) {
        if ((*cut).IsSelected(fValues)) {
          filterMap |= (uint8_t(1) << i);
          fHistMan->FillHistClass(Form("TrackBarrel_%s", (*cut).GetName()), fValues);
        }
      }
      trackSel(filterMap);
      if (!filterMap) {
        continue;
      }

      for (auto& sig : fMCSignals) {
        if (sig.CheckSignal(false, tracksMC, mctrack)) {
          for (int j = 0; j < fTrackCuts.size(); j++) {
            if (filterMap & (uint8_t(1) << j)) {
              fHistMan->FillHistClass(Form("TrackBarrel_%s_%s", fTrackCuts[j].GetName(), sig.GetName()), fValues);
            }
          }
        }
      }
    } // end loop over tracks
  }
};

struct DQQuarkoniumPairing {
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  //NOTE: one could define also a dilepton cut, but for now basic selections can be supported using Partition

  float* fValues;
  std::vector<TString> fCutNames;
  std::vector<MCSignal> fRecMCSignals;
  std::vector<MCSignal> fGenMCSignals;

  Filter filterEventSelected = aod::reducedevent::isEventSelected == 1;
  Filter filterBarrelTrackSelected = aod::reducedtrack::isBarrelSelected > uint8_t(0);

  // TODO: here we specify signals, however signal decisions are precomputed and stored in mcReducedFlags
  // TODO: The tasks based on skimmed MC could/should rely ideally just on these flags
  // TODO:   special AnalysisCuts to be prepared in this direction
  Configurable<std::string> fConfigElectronCuts{"cfgElectronCuts", "jpsiPID1", "Comma separated list of barrel track cuts"};
  Configurable<std::string> fConfigMCRecSignals{"cfgBarrelMCRecSignals", "", "Comma separated list of MC signals (reconstructed)"};
  Configurable<std::string> fConfigMCGenSignals{"cfgBarrelMCGenSignals", "", "Comma separated list of MC signals (generated)"};
  // TODO: cuts on the MC truth information to be added if needed

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString histClasses = "";
    TString sigNamesStr = fConfigMCRecSignals.value;
    std::unique_ptr<TObjArray> objRecSigArray(sigNamesStr.Tokenize(","));

    // Add pair histogram classes separately for each track cut;
    //  Also, add histogram classes for each track cut and each user specified reconstructed MC signal (must be 2-prong MC signals)
    TString cutNamesStr = fConfigElectronCuts.value;
    if (!cutNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(cutNamesStr.Tokenize(","));
      for (int icut = 0; icut < objArray->GetEntries(); ++icut) {
        fCutNames.push_back(objArray->At(icut)->GetName());
        histClasses += Form("PairsBarrelSEPM_%s;PairsBarrelSEPP_%s;PairsBarrelSEMM_%s;",
                            fCutNames[icut].Data(), fCutNames[icut].Data(), fCutNames[icut].Data());
        if (objRecSigArray) {
          for (int isig = 0; isig < objRecSigArray->GetEntries(); ++isig) {
            MCSignal* sig = o2::aod::dqmcsignals::GetMCSignal(objRecSigArray->At(isig)->GetName());
            if (sig) {
              if (sig->GetNProngs() != 2) { // NOTE: 2-prong signals required
                continue;
              }
              fRecMCSignals.push_back(*sig);
              histClasses += Form("PairsBarrelSEPM_%s_%s;", fCutNames[icut].Data(), sig->GetName());
            }
          }
        }
      }
    }

    // Add histogram classes for each specified generated signal
    TString sigGenNamesStr = fConfigMCGenSignals.value;
    std::unique_ptr<TObjArray> objGenSigArray(sigGenNamesStr.Tokenize(","));
    for (int isig = 0; isig < objGenSigArray->GetEntries(); isig++) {
      MCSignal* sig = o2::aod::dqmcsignals::GetMCSignal(objGenSigArray->At(isig)->GetName());
      if (sig) {
        if (sig->GetNProngs() != 1) { // NOTE: 1-prong signals required
          continue;
        }
        fGenMCSignals.push_back(*sig);
        histClasses += Form("MCTruthGen_%s;", sig->GetName());
      }
    }

    DefineHistograms(fHistMan, histClasses.Data());  // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    VarManager::SetupTwoProngDCAFitter(5.0f, true, 200.0f, 4.0f, 1.0e-3f, 0.9f, true); // TODO: get these parameters from Configurables
  }

  void process(soa::Filtered<MyEventsVtxCovSelected>::iterator const& event, soa::Filtered<MyBarrelTracksSelected> const& tracks, ReducedMCTracks const& tracksMC)
  {
    // Reset the fValues array
    VarManager::ResetValues(0, VarManager::kNVars, fValues);
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    // Run the same event pairing for barrel tracks
    uint8_t twoTrackFilter = 0;
    uint16_t dileptonFilterMap = 0;
    // Loop over reconstructed pairs and fill histograms
    for (auto& [t1, t2] : combinations(tracks, tracks)) {
      twoTrackFilter = t1.isBarrelSelected() & t2.isBarrelSelected();
      if (!twoTrackFilter) { // the tracks must have at least one filter bit in common to continue
        continue;
      }
      VarManager::FillPair(t1, t2, fValues);
      VarManager::FillPairVertexing(event, t1, t2, fValues);
      for (int i = 0; i < fCutNames.size(); ++i) {
        if (twoTrackFilter & (uint8_t(1) << i)) {
          if (t1.sign() * t2.sign() < 0) {
            fHistMan->FillHistClass(Form("PairsBarrelSEPM_%s", fCutNames[i].Data()), fValues);
            for (auto& sig : fRecMCSignals) {
              if (sig.CheckSignal(false, tracksMC, t1.reducedMCTrack(), t2.reducedMCTrack())) {
                fHistMan->FillHistClass(Form("PairsBarrelSEPM_%s_%s", fCutNames[i].Data(), sig.GetName()), fValues);
              }
            }
          } else {
            if (t1.sign() > 0) {
              fHistMan->FillHistClass(Form("PairsBarrelSEPP_%s", fCutNames[i].Data()), fValues);
            } else {
              fHistMan->FillHistClass(Form("PairsBarrelSEMM_%s", fCutNames[i].Data()), fValues);
            }
          }
        }
      }
    } // end loop over barrel track pairs

    // loop over mc stack and fill histograms for pure MC truth signals
    for (auto& mctrack : tracksMC) {
      VarManager::FillTrack<gkParticleMCFillMap>(mctrack, fValues);
      // NOTE: Signals are checked here based on the skimmed MC stack, so depending on the requested signal, the stack could be incomplete.
      // NOTE: However, the working model is that the decisions on MC signals are precomputed during skimming and are stored in the mcReducedFlags member.
      // TODO:  Use the mcReducedFlags to select signals
      for (auto& sig : fGenMCSignals) {
        if (sig.CheckSignal(false, tracksMC, mctrack)) {
          fHistMan->FillHistClass(Form("MCTruthGen_%s", sig.GetName()), fValues);
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<DQEventSelection>(cfgc),
    adaptAnalysisTask<DQBarrelTrackSelection>(cfgc),
    adaptAnalysisTask<DQQuarkoniumPairing>(cfgc)};
}

void DefineHistograms(HistogramManager* histMan, TString histClasses)
{
  //
  // Define here the histograms for all the classes required in analysis.
  //  The histogram classes are provided in the histClasses string, separated by semicolon ";"
  //  The histogram classes and their components histograms are defined below depending on the name of the histogram class
  //
  std::unique_ptr<TObjArray> objArray(histClasses.Tokenize(";"));
  for (Int_t iclass = 0; iclass < objArray->GetEntries(); ++iclass) {
    TString classStr = objArray->At(iclass)->GetName();
    histMan->AddHistClass(classStr.Data());

    // NOTE: The level of detail for histogramming can be controlled via configurables
    if (classStr.Contains("Event")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "event", "trigger,cent,mc");
    }

    if (classStr.Contains("Track")) {
      if (classStr.Contains("Barrel")) {
        dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "track", "its,tpcpid,dca,tofpid,mc");
      }
    }

    if (classStr.Contains("Pairs")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "pair_barrel", "vertexing-barrel");
    }

    if (classStr.Contains("MCTruthGen")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "mctruth");
      /*histMan->AddHistogram(objArray->At(iclass)->GetName(), "Pt", "MC generator p_{T} distribution", false, 200, 0.0, 20.0, VarManager::kMCPt);
      histMan->AddHistogram(objArray->At(iclass)->GetName(), "Eta", "MC generator #eta distribution", false, 500, -5.0, 5.0, VarManager::kMCEta);*/
      histMan->AddHistogram(objArray->At(iclass)->GetName(), "Phi", "MC generator #varphi distribution", false, 500, -6.3, 6.3, VarManager::kMCPhi);
    }
  } // end loop over histogram classes
}
