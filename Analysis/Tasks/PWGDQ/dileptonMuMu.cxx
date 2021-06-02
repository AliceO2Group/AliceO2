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
#include "PWGDQCore/HistogramsLibrary.h"
#include "PWGDQCore/CutsLibrary.h"
#include <TH1F.h>
#include <THashList.h>
#include <TString.h>
#include <iostream>
#include <vector>
#include <algorithm>

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
DECLARE_SOA_COLUMN(MixingHash, mixingHash, int);
DECLARE_SOA_COLUMN(IsEventSelected, isEventSelected, int);
DECLARE_SOA_COLUMN(IsEventMixingSelected, isEventMixingSelected, int);
} // namespace reducedevent

namespace reducedtrack
{
DECLARE_SOA_COLUMN(IsMuonSelected, isMuonSelected, uint8_t);
} // namespace reducedtrack

DECLARE_SOA_TABLE(EventCuts, "AOD", "EVENTCUTS", reducedevent::IsEventSelected);
DECLARE_SOA_TABLE(EventMixingCuts, "AOD", "EVENTMIXINGCUTS", reducedevent::IsEventMixingSelected);
DECLARE_SOA_TABLE(MixingHashes, "AOD", "MIXINGHASHES", reducedevent::MixingHash);
DECLARE_SOA_TABLE(MuonTrackCuts, "AOD", "MUONTRACKCUTS", reducedtrack::IsMuonSelected);
} // namespace o2::aod

using MyEvents = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended>;
using MyEventsSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::EventMixingCuts>;
using MyEventsHashSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::EventMixingCuts, aod::MixingHashes>;
using MyEventsVtxCov = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>;
using MyEventsVtxCovSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov, aod::EventCuts>;

using MyMuonTracks = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtra, aod::ReducedMuonsCov>;
using MyMuonTracksSelected = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtra, aod::ReducedMuonsCov, aod::MuonTrackCuts>;

void DefineHistograms(HistogramManager* histMan, TString histClasses);

// HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
//         a constexpr static bit map must be defined and sent as template argument
//        The user has to include in this bit map all the tables needed in analysis, as defined in VarManager::ObjTypes
//        Additionally, one should make sure that the requested tables are actually provided in the process() function,
//       otherwise a compile time error will be thrown.
//        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
//           to automatically detect the object types transmitted to the VarManager
constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::ReducedEvent | VarManager::ObjTypes::ReducedEventExtended;
constexpr static uint32_t gkMuonFillMap = VarManager::ObjTypes::ReducedMuon | VarManager::ObjTypes::ReducedMuonExtra | VarManager::ObjTypes::ReducedMuonCov;

struct DQEventSelection {
  Produces<aod::EventCuts> eventSel;
  Produces<aod::EventMixingCuts> eventMixingSel;
  Produces<aod::MixingHashes> hash;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut* fEventCut;
  AnalysisCompositeCut* fEventMixingCut;
  float* fValues;

  // TODO: make mixing binning configurable
  std::vector<float> fCentLimsHashing{0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f};

  Configurable<std::string> fConfigEventCuts{"cfgEventCuts", "eventDimuonStandard", "Event selection"};
  Configurable<std::string> fConfigEventMixingCuts{"cfgEventMixingCuts", "eventMuonStandard", "Event selection"};

  int getHash(float centrality)
  {
    if ((centrality < *fCentLimsHashing.begin()) || (centrality > *(fCentLimsHashing.end() - 1))) {
      return -1;
    }
    auto cent = std::lower_bound(fCentLimsHashing.begin(), fCentLimsHashing.end(), centrality);
    return (cent - fCentLimsHashing.begin());
  }

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "Event_BeforeCuts;Event_AfterCuts;EventMixing_BeforeCuts;EventMixing_AfterCuts;"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());                                                              // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    DefineCuts();
  }

  void DefineCuts()
  {
    fEventCut = new AnalysisCompositeCut(true);
    TString eventCutStr = fConfigEventCuts.value;
    fEventCut->AddCut(dqcuts::GetAnalysisCut(eventCutStr.Data()));

    fEventMixingCut = new AnalysisCompositeCut(true);
    TString eventMixingCutStr = fConfigEventMixingCuts.value;
    fEventMixingCut->AddCut(dqcuts::GetAnalysisCut(eventMixingCutStr.Data()));

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

    fHistMan->FillHistClass("EventMixing_BeforeCuts", fValues);
    if (fEventMixingCut->IsSelected(fValues)) {
      fHistMan->FillHistClass("EventMixing_AfterCuts", fValues);
      eventMixingSel(1);
    } else {
      eventMixingSel(0);
    }
    int hh = getHash(fValues[VarManager::kCentVZERO]);
    hash(hh);
  }
};

struct DQMuonTrackSelection {
  Produces<aod::MuonTrackCuts> trackSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  // NOTE: One single cut is implemented for muons, but multiple one can be computed
  AnalysisCompositeCut* fTrackCut;

  float* fValues;

  Configurable<float> fConfigMuonPtLow{"cfgMuonLowPt", 1.0f, "Low pt cut for muons"};
  Configurable<std::string> fConfigMuonCuts{"cfgMuonCuts", "muonQualityCuts", "muon cut"};

  Filter muonFilter = aod::reducedmuon::pt >= fConfigMuonPtLow;

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
    TString muonCutStr = fConfigMuonCuts.value;
    fTrackCut->AddCut(dqcuts::GetCompositeCut(muonCutStr.Data()));

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvents::iterator const& event, soa::Filtered<MyMuonTracks> const& muons)
  {
    VarManager::ResetValues(0, VarManager::kNMuonTrackVariables, fValues);
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    for (auto& muon : muons) {
      VarManager::FillTrack<gkMuonFillMap>(muon, fValues);
      fHistMan->FillHistClass("TrackMuon_BeforeCuts", fValues);

      if (fTrackCut->IsSelected(fValues)) {
        trackSel(uint8_t(1));
        fHistMan->FillHistClass("TrackMuon_AfterCuts", fValues);
      } else {
        trackSel(uint8_t(0));
      }
    }
  }
};

struct DQEventMixing {
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  float* fValues;
  // NOTE: The track filter produced by the barrel track selection contain a number of electron cut decisions and one last cut for hadrons used in the
  //           dilepton - hadron task downstream. So the bit mask is required to select pairs just based on the electron cuts
  uint8_t fTwoTrackFilterMask = 0;
  std::vector<TString> fCutNames;

  Filter filterEventMxingSelected = aod::reducedevent::isEventMixingSelected == 1;
  Filter filterMuonTrackSelected = aod::reducedtrack::isMuonSelected > uint8_t(0);

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString histNames = "";
    histNames += "PairsMuonMEPM;PairsMuonMEPP;PairsMuonMEMM;";

    DefineHistograms(fHistMan, histNames.Data());    // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
  }

  void process(soa::Filtered<MyEventsHashSelected>& events, soa::Filtered<MyMuonTracksSelected> const& muons)
  {
    uint8_t twoTrackFilter = 0;

    events.bindExternalIndices(&muons);
    auto muonsTuple = std::make_tuple(muons);
    AnalysisDataProcessorBuilder::GroupSlicer slicerMuons(events, muonsTuple);

    // Strictly upper categorised collisions, for 100 combinations per bin, skipping those in entry -1
    for (auto& [event1, event2] : selfCombinations("fMixingHash", 100, -1, events, events)) {

      // event informaiton is required to fill histograms where both event and pair information is required (e.g. inv.mass vs centrality)
      VarManager::ResetValues(0, VarManager::kNVars, fValues);
      VarManager::FillEvent<gkEventFillMap>(event1, fValues);

      auto im1 = slicerMuons.begin();
      auto im2 = slicerMuons.begin();
      for (auto& slice : slicerMuons) {
        if (slice.groupingElement().index() == event1.index()) {
          im1 = slice;
          break;
        }
      }
      for (auto& slice : slicerMuons) {
        if (slice.groupingElement().index() == event2.index()) {
          im2 = slice;
          break;
        }
      }

      auto muons1 = std::get<soa::Filtered<MyMuonTracksSelected>>(im1.associatedTables());
      muons1.bindExternalIndices(&events);
      auto muons2 = std::get<soa::Filtered<MyMuonTracksSelected>>(im2.associatedTables());
      muons2.bindExternalIndices(&events);

      for (auto& muon1 : muons1) {
        for (auto& muon2 : muons2) {
          twoTrackFilter = muon1.isMuonSelected() & muon2.isMuonSelected();
          if (!twoTrackFilter) { // the tracks must have at least one filter bit in common to continue
            continue;
          }
          VarManager::FillPair(muon1, muon2, fValues, VarManager::kJpsiToMuMu);
          if (muon1.sign() * muon2.sign() < 0) {
            fHistMan->FillHistClass("PairsMuonMEPM", fValues);
          } else {
            if (muon1.sign() > 0) {
              fHistMan->FillHistClass("PairsMuonMEPP", fValues);
            } else {
              fHistMan->FillHistClass("PairsMuonMEMM", fValues);
            }
          }
        } // end for (muon2)
      }   // end for (muon1)
    }     // end for (event combinations)
  }       // end process()
};

struct DQDileptonMuMu {
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  //NOTE: one could define also a dilepton cut, but for now basic selections can be supported using Partition

  float* fValues;
  // NOTE: The track filter produced by the barrel track selection contain a number of electron cut decisions and one last cut for hadrons used in the
  //           dilepton - hadron task downstream. So the bit mask is required to select pairs just based on the electron cuts
  uint8_t fTwoTrackFilterMask = 0;
  std::vector<TString> fCutNames;

  Filter filterEventSelected = aod::reducedevent::isEventSelected == 1;
  Filter filterMuonTrackSelected = aod::reducedtrack::isMuonSelected > uint8_t(0);

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString histNames = "";
    histNames += "PairsMuonSEPM;PairsMuonSEPP;PairsMuonSEMM;";

    DefineHistograms(fHistMan, histNames.Data());    // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
  }

  void process(soa::Filtered<MyEventsVtxCovSelected>::iterator const& event, soa::Filtered<MyMuonTracksSelected> const& muons)
  {
    if (!event.isEventSelected()) {
      return;
    }
    // Reset the fValues array
    VarManager::ResetValues(0, VarManager::kNVars, fValues);
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    // Run the same event pairing for barrel tracks
    uint8_t twoTrackFilter = 0;

    // same event pairing for muons
    for (auto& [muon1, muon2] : combinations(muons, muons)) {
      twoTrackFilter = muon1.isMuonSelected() & muon2.isMuonSelected();
      if (!twoTrackFilter) { // the muons must have at least one filter bit in common to continue
        continue;
      }
      VarManager::FillPair(muon1, muon2, fValues, VarManager::kJpsiToMuMu);
      if (muon1.sign() * muon2.sign() < 0) {
        fHistMan->FillHistClass("PairsMuonSEPM", fValues);
      } else {
        if (muon1.sign() > 0) {
          fHistMan->FillHistClass("PairsMuonSEPP", fValues);
        } else {
          fHistMan->FillHistClass("PairsMuonSEMM", fValues);
        }
      }
    } // end loop over muon track pairs
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<DQEventSelection>(cfgc),
    adaptAnalysisTask<DQMuonTrackSelection>(cfgc),
    adaptAnalysisTask<DQEventMixing>(cfgc),
    adaptAnalysisTask<DQDileptonMuMu>(cfgc)};
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
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "event", "trigger,cent,muon");
    }

    if (classStr.Contains("Track")) {
      if (classStr.Contains("Muon")) {
        dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "track", "muon");
      }
    }

    if (classStr.Contains("Pairs")) {
      if (classStr.Contains("Muon")) {
        dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "pair_dimuon");
      }
    }
  } // end loop over histogram classes
}
