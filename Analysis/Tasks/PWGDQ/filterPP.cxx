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
#include "PWGDQCore/CutsLibrary.h"
#include "PWGDQCore/HistogramsLibrary.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include <TH1F.h>
#include <TH2I.h>
#include <TMath.h>
#include <THashList.h>
#include <TString.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod;

// Some definitions
namespace o2::aod
{
namespace dqppfilter
{
DECLARE_SOA_COLUMN(IsDQEventSelected, isDQEventSelected, int);
DECLARE_SOA_COLUMN(IsBarrelSelected, isBarrelSelected, uint8_t);
} // namespace dqppfilter

DECLARE_SOA_TABLE(DQEventCuts, "AOD", "DQEVENTCUTS", dqppfilter::IsDQEventSelected);
DECLARE_SOA_TABLE(DQBarrelTrackCuts, "AOD", "DQBARRELCUTS", dqppfilter::IsBarrelSelected);
} // namespace o2::aod

using MyEvents = soa::Join<aod::Collisions, aod::EvSels>;
using MyEventsSelected = soa::Join<aod::Collisions, aod::EvSels, aod::DQEventCuts>;
using MyBarrelTracks = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksExtended, aod::TrackSelection, aod::pidRespTPC, aod::pidRespTOF, aod::pidRespTOFbeta>;
using MyBarrelTracksSelected = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksExtended, aod::TrackSelection, aod::pidRespTPC, aod::pidRespTOF, aod::pidRespTOFbeta, aod::DQBarrelTrackCuts>;

constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::BC | VarManager::ObjTypes::Collision;
constexpr static uint32_t gkTrackFillMap = VarManager::ObjTypes::Track | VarManager::ObjTypes::TrackExtra | VarManager::ObjTypes::TrackDCA | VarManager::ObjTypes::TrackSelection | VarManager::ObjTypes::TrackCov | VarManager::ObjTypes::TrackPID;

void DefineHistograms(HistogramManager* histMan, TString histClasses);

struct EventSelectionTask {
  Produces<aod::DQEventCuts> eventSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut fEventCut{true};

  float* fValues;

  Configurable<std::string> fConfigCuts{"cfgEventCuts", "eventStandard", "Comma separated list of event cuts; multiple cuts are applied with a logical AND"};

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

  void process(MyEvents::iterator const& event, aod::BCs const& bcs)
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

struct BarrelTrackSelectionTask {
  Produces<aod::DQBarrelTrackCuts> trackSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  std::vector<AnalysisCompositeCut> fTrackCuts;

  float* fValues; // array to be used by the VarManager

  Configurable<std::string> fConfigCuts{"cfgBarrelTrackCuts", "jpsiPID1", "Comma separated list of barrel track cuts"};

  void init(o2::framework::InitContext&)
  {
    DefineCuts();

    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
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

  void process(MyEvents::iterator const& event, MyBarrelTracks const& tracks, aod::BCs const& bcs)
  {
    uint8_t filterMap = uint8_t(0);
    trackSel.reserve(tracks.size());

    VarManager::ResetValues(0, VarManager::kNBarrelTrackVariables, fValues);
    // fill event information which might be needed in histograms that combine track and event properties
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    for (auto& track : tracks) {
      filterMap = uint8_t(0);
      // TODO: if a condition on the event selection is applied, the histogram output is missing
      VarManager::FillTrack<gkTrackFillMap>(track, fValues);
      fHistMan->FillHistClass("TrackBarrel_BeforeCuts", fValues);
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

struct FilterPPTask {
  Produces<aod::DQEventFilter> eventFilter;
  OutputObj<THashList> fOutputList{"output"};
  OutputObj<TH2I> fStats{"stats"};
  HistogramManager* fHistMan;
  std::vector<AnalysisCompositeCut> fPairCuts;

  float* fValues;

  Partition<MyBarrelTracksSelected> posTracks = aod::track::signed1Pt > 0.0f && aod::dqppfilter::isBarrelSelected > uint8_t(0);
  Partition<MyBarrelTracksSelected> negTracks = aod::track::signed1Pt < 0.0f && aod::dqppfilter::isBarrelSelected > uint8_t(0);

  Configurable<std::string> fConfigTrackCuts{"cfgBarrelTrackCuts", "jpsiPID1", "Comma separated list of barrel track cuts"};
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

    // initialize the variable manager
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    // configure histograms
    TString trackCutNamesStr = fConfigTrackCuts.value;
    fTrkCutsNameArray = trackCutNamesStr.Tokenize(",");
    fNTrackCuts = fTrkCutsNameArray->GetEntries();
    TString histNames = "";
    for (int i = 0; i < fNTrackCuts; i++) {
      histNames += Form("PairsBarrelPM_%s;", fTrkCutsNameArray->At(i)->GetName());
    }
    DefineHistograms(fHistMan, histNames.Data());    // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    // configure the stats histogram
    fStats.setObject(new TH2I("stats", "stats", fNPairCuts, -0.5, -0.5 + fNPairCuts, fNTrackCuts, -0.5, -0.5 + fNTrackCuts));
    for (int i = 0; i < fNPairCuts; ++i) {
      fStats->GetXaxis()->SetBinLabel(i + 1, fPairCuts[i].GetName());
    }
    for (int i = 0; i < fNTrackCuts; ++i) {
      fStats->GetYaxis()->SetBinLabel(i + 1, fTrkCutsNameArray->At(i)->GetName());
    }
  }

  void process(MyEventsSelected::iterator const& event, MyBarrelTracksSelected const& tracks, aod::BCs const& bcs)
  {
    uint64_t filter = 0;

    if (event.isDQEventSelected() == 1) {
      // Reset the fValues array
      VarManager::ResetValues(0, VarManager::kNVars, fValues);
      VarManager::FillEvent<gkEventFillMap>(event, fValues);

      uint8_t* pairsCount = new uint8_t[fNPairCuts * fNTrackCuts];
      for (int i = 0; i < fNPairCuts * fNTrackCuts; i++) {
        pairsCount[i] = 0;
      }

      uint8_t cutFilter = 0;
      for (auto tpos : posTracks) {
        for (auto tneg : negTracks) { // +- pairs
          cutFilter = tpos.isBarrelSelected() & tneg.isBarrelSelected();
          if (!cutFilter) { // the tracks must have at least one filter bit in common to continue
            continue;
          }
          VarManager::FillPair(tpos, tneg, fValues); // compute pair quantities
          for (int i = 0; i < fNTrackCuts; ++i) {
            if (!(cutFilter & (uint8_t(1) << i))) {
              continue;
            }
            fHistMan->FillHistClass(Form("PairsBarrelPM_%s", fTrkCutsNameArray->At(i)->GetName()), fValues);
            int j = 0;
            for (auto cut = fPairCuts.begin(); cut != fPairCuts.end(); cut++, j++) {
              if ((*cut).IsSelected(fValues)) {
                pairsCount[j + i * fNPairCuts] += 1;
              }
            }
          }
        }
      }

      for (int i = 0; i < fNTrackCuts; i++) {
        for (int j = 0; j < fNPairCuts; j++) {
          if (pairsCount[j + i * fNPairCuts] > 0) {
            filter |= (uint64_t(1) << (j + i * fNPairCuts));
            fStats->Fill(j, i);
          }
        }
      }
      delete[] pairsCount;
    }
    eventFilter(filter);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelectionTask>("dq-event-selection"),
    adaptAnalysisTask<BarrelTrackSelectionTask>("dq-barrel-track-selection"),
    adaptAnalysisTask<FilterPPTask>("dq-ppFilter")};
}

void DefineHistograms(HistogramManager* histMan, TString histClasses)
{
  //
  // Define here the histograms for all the classes required in analysis.
  //  The histogram classes are provided in the histClasses string, separated by semicolon ";"
  //  NOTE: Additional histograms to those predefined in the library can be added here !!
  const int kNRuns = 25;
  int runs[kNRuns] = {
    285009, 285011, 285012, 285013, 285014, 285015, 285064, 285065, 285066, 285106,
    285108, 285125, 285127, 285165, 285200, 285202, 285203, 285222, 285224, 285327,
    285328, 285347, 285364, 285365, 285396};
  VarManager::SetRunNumbers(kNRuns, runs);

  std::unique_ptr<TObjArray> objArray(histClasses.Tokenize(";"));
  for (Int_t iclass = 0; iclass < objArray->GetEntries(); ++iclass) {
    TString classStr = objArray->At(iclass)->GetName();
    histMan->AddHistClass(classStr.Data());

    if (classStr.Contains("Event")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "event", "trigger,vtxPbPb");
    }

    if (classStr.Contains("Track")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "track", "its,tpcpid,dca");
    }

    if (classStr.Contains("Pairs")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "pair");
    }
  }
}
