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
} // namespace reducedevent

namespace reducedtrack
{
DECLARE_SOA_COLUMN(IsBarrelSelected, isBarrelSelected, uint8_t);
DECLARE_SOA_COLUMN(IsMuonSelected, isMuonSelected, uint8_t);
} // namespace reducedtrack

DECLARE_SOA_TABLE(EventCuts, "AOD", "EVENTCUTS", reducedevent::IsEventSelected);
DECLARE_SOA_TABLE(MixingHashes, "AOD", "MIXINGHASHES", reducedevent::MixingHash);
DECLARE_SOA_TABLE(BarrelTrackCuts, "AOD", "BARRELTRACKCUTS", reducedtrack::IsBarrelSelected);
DECLARE_SOA_TABLE(MuonTrackCuts, "AOD", "MUONTRACKCUTS", reducedtrack::IsMuonSelected);
} // namespace o2::aod

using MyEvents = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended>;
using MyEventsSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::EventCuts>;
using MyEventsHashSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::EventCuts, aod::MixingHashes>;
using MyEventsVtxCov = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov>;
using MyEventsVtxCovSelected = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended, aod::ReducedEventsVtxCov, aod::EventCuts>;

using MyBarrelTracks = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov, aod::ReducedTracksBarrelPID>;
using MyBarrelTracksSelected = soa::Join<aod::ReducedTracks, aod::ReducedTracksBarrel, aod::ReducedTracksBarrelCov, aod::ReducedTracksBarrelPID, aod::BarrelTrackCuts>;

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
constexpr static uint32_t gkTrackFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::ReducedTrackBarrel | VarManager::ObjTypes::ReducedTrackBarrelCov | VarManager::ObjTypes::ReducedTrackBarrelPID;
constexpr static uint32_t gkMuonFillMap = VarManager::ObjTypes::ReducedMuon | VarManager::ObjTypes::ReducedMuonExtra | VarManager::ObjTypes::ReducedMuonCov;

struct DQEventSelection {
  Produces<aod::EventCuts> eventSel;
  Produces<aod::MixingHashes> hash;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  AnalysisCompositeCut* fEventCut;
  float* fValues;

  // TODO: make mixing binning configurable
  std::vector<float> fCentLimsHashing{0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f};

  Configurable<std::string> fConfigEventCuts{"cfgEventCuts", "eventStandard", "Event selection"};

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
    int hh = getHash(fValues[VarManager::kCentVZERO]);
    hash(hh);
  }
};

struct DQBarrelTrackSelection {
  Produces<aod::BarrelTrackCuts> trackSel;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  std::vector<AnalysisCompositeCut> fTrackCuts;

  float* fValues; // array to be used by the VarManager

  Configurable<std::string> fConfigElectronCuts{"cfgElectronCuts", "jpsiPID1", "Comma separated list of barrel track cuts"};

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
    // Defines track cuts for both the dielectron candidates and also for the dilepton - hadron task
    TString cutNamesStr = fConfigElectronCuts.value;
    if (!cutNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(cutNamesStr.Tokenize(","));
      for (int icut = 0; icut < objArray->GetEntries(); ++icut) {
        fTrackCuts.push_back(*dqcuts::GetCompositeCut(objArray->At(icut)->GetName()));
      }
    }

    AnalysisCompositeCut correlationsHadronCut("hadronCut", "hadronCut", true);
    correlationsHadronCut.AddCut(dqcuts::GetAnalysisCut("electronStandardQuality"));
    correlationsHadronCut.AddCut(dqcuts::GetAnalysisCut("standardPrimaryTrack"));
    AnalysisCut hadronKine;
    hadronKine.AddCut(VarManager::kPt, 4.0, 100.0);
    correlationsHadronCut.AddCut(&hadronKine);
    fTrackCuts.push_back(correlationsHadronCut);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvents::iterator const& event, MyBarrelTracks const& tracks)
  {
    VarManager::ResetValues(0, VarManager::kNBarrelTrackVariables, fValues);
    // fill event information which might be needed in histograms that combine track and event properties
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    uint8_t filterMap = uint8_t(0);

    trackSel.reserve(tracks.size());

    for (auto& track : tracks) {
      filterMap = uint8_t(0);
      VarManager::FillTrack<gkTrackFillMap>(track, fValues);
      fHistMan->FillHistClass("TrackBarrel_BeforeCuts", fValues);

      int i = 0;
      for (auto cut = fTrackCuts.begin(); cut != fTrackCuts.end(); cut++, i++) {
        if ((*cut).IsSelected(fValues)) {
          filterMap |= (uint8_t(1) << i);
          fHistMan->FillHistClass(Form("TrackBarrel_%s", (*cut).GetName()), fValues);
        }
      }
      trackSel(filterMap);
    }
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
    kineMuonCut.AddCut(VarManager::kPt, fConfigMuonPtLow, 100.0);
    fTrackCut->AddCut(&kineMuonCut);

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void process(MyEvents::iterator const& event, MyMuonTracks const& muons)
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

  Configurable<std::string> fConfigElectronCuts{"cfgElectronCuts", "jpsiPID1", "Comma separated list of barrel track cuts"};

  Filter filterEventSelected = aod::reducedevent::isEventSelected == 1;
  Filter filterTrackSelected = aod::reducedtrack::isBarrelSelected > uint8_t(0);
  Filter filterMuonTrackSelected = aod::reducedtrack::isMuonSelected > uint8_t(0);

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString histNames = "";
    TString configCutNamesStr = fConfigElectronCuts.value;
    if (!configCutNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(configCutNamesStr.Tokenize(","));
      for (int icut = 0; icut < objArray->GetEntries(); ++icut) {
        fCutNames.push_back(objArray->At(icut)->GetName());
        histNames += Form("PairsBarrelMEPM_%s;PairsBarrelMEPP_%s;PairsBarrelMEMM_%s;", fCutNames[icut].Data(), fCutNames[icut].Data(), fCutNames[icut].Data());
        histNames += Form("PairsEleMuMEPM_%s;PairsEleMuMEPP_%s;PairsEleMuMEMM_%s;", fCutNames[icut].Data(), fCutNames[icut].Data(), fCutNames[icut].Data());
        fTwoTrackFilterMask |= (uint8_t(1) << icut);
      }
    }
    histNames += "PairsMuonMEPM;PairsMuonMEPP;PairsMuonMEMM;";

    DefineHistograms(fHistMan, histNames.Data());    // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
  }

  void process(soa::Filtered<MyEventsHashSelected>& events, soa::Filtered<MyBarrelTracksSelected> const& tracks, soa::Filtered<MyMuonTracksSelected> const& muons)
  {
    uint8_t twoTrackFilter = 0;

    events.bindExternalIndices(&tracks);
    events.bindExternalIndices(&muons);
    auto tracksTuple = std::make_tuple(tracks);
    auto muonsTuple = std::make_tuple(muons);
    AnalysisDataProcessorBuilder::GroupSlicer slicerTracks(events, tracksTuple);
    AnalysisDataProcessorBuilder::GroupSlicer slicerMuons(events, muonsTuple);

    // Strictly upper categorised collisions, for 100 combinations per bin, skipping those in entry -1
    for (auto& [event1, event2] : selfCombinations("fMixingHash", 100, -1, events, events)) {

      // event informaiton is required to fill histograms where both event and pair information is required (e.g. inv.mass vs centrality)
      VarManager::ResetValues(0, VarManager::kNVars, fValues);
      VarManager::FillEvent<gkEventFillMap>(event1, fValues);

      auto it1 = slicerTracks.begin();
      auto it2 = slicerTracks.begin();
      for (auto& slice : slicerTracks) {
        if (slice.groupingElement().index() == event1.index()) {
          it1 = slice;
          break;
        }
      }
      for (auto& slice : slicerTracks) {
        if (slice.groupingElement().index() == event2.index()) {
          it2 = slice;
          break;
        }
      }

      auto tracks1 = std::get<soa::Filtered<MyBarrelTracksSelected>>(it1.associatedTables());
      tracks1.bindExternalIndices(&events);
      auto tracks2 = std::get<soa::Filtered<MyBarrelTracksSelected>>(it2.associatedTables());
      tracks2.bindExternalIndices(&events);

      for (auto& track1 : tracks1) {
        for (auto& track2 : tracks2) {
          twoTrackFilter = track1.isBarrelSelected() & track2.isBarrelSelected() & fTwoTrackFilterMask;

          if (!twoTrackFilter) { // the tracks must have at least one filter bit in common to continue
            continue;
          }
          VarManager::FillPair(track1, track2, fValues);

          for (int i = 0; i < fCutNames.size(); ++i) {
            if (twoTrackFilter & (uint8_t(1) << i)) {
              if (track1.sign() * track2.sign() < 0) {
                fHistMan->FillHistClass(Form("PairsBarrelMEPM_%s", fCutNames[i].Data()), fValues);
              } else {
                if (track1.sign() > 0) {
                  fHistMan->FillHistClass(Form("PairsBarrelMEPP_%s", fCutNames[i].Data()), fValues);
                } else {
                  fHistMan->FillHistClass(Form("PairsBarrelMEMM_%s", fCutNames[i].Data()), fValues);
                }
              }
            } // end if (filter bits)
          }   // end for (cuts)
        }     // end for (track2)
      }       // end for (track1)

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

      for (auto& track1 : tracks1) {
        for (auto& muon2 : muons2) {
          twoTrackFilter = (track1.isBarrelSelected() & fTwoTrackFilterMask) && muon2.isMuonSelected();

          if (!twoTrackFilter) { // the tracks must have at least one filter bit in common to continue
            continue;
          }
          VarManager::FillPair(track1, muon2, fValues, VarManager::kElectronMuon);

          for (int i = 0; i < fCutNames.size(); ++i) {
            if (twoTrackFilter & (uint8_t(1) << i)) {
              if (track1.sign() * muon2.sign() < 0) {
                fHistMan->FillHistClass(Form("PairsEleMuMEPM_%s", fCutNames[i].Data()), fValues);
              } else {
                if (track1.sign() > 0) {
                  fHistMan->FillHistClass(Form("PairsEleMuMEPP_%s", fCutNames[i].Data()), fValues);
                } else {
                  fHistMan->FillHistClass(Form("PairsEleMuMEMM_%s", fCutNames[i].Data()), fValues);
                }
              }
            } // end if (filter bits)
          }   // end for (cuts)
        }     // end for (muon2)
      }       // end for (track1)
    }         // end for (event combinations)
  }           // end process()
};

struct DQTableReader {
  Produces<aod::Dileptons> dileptonList;
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;
  //NOTE: one could define also a dilepton cut, but for now basic selections can be supported using Partition

  float* fValues;
  // NOTE: The track filter produced by the barrel track selection contain a number of electron cut decisions and one last cut for hadrons used in the
  //           dilepton - hadron task downstream. So the bit mask is required to select pairs just based on the electron cuts
  uint8_t fTwoTrackFilterMask = 0;
  std::vector<TString> fCutNames;

  Filter filterEventSelected = aod::reducedevent::isEventSelected == 1;
  // NOTE: the barrel filter map contains decisions for both electrons and hadrons used in the correlation task
  Filter filterBarrelTrackSelected = aod::reducedtrack::isBarrelSelected > uint8_t(0);
  Filter filterMuonTrackSelected = aod::reducedtrack::isMuonSelected > uint8_t(0);

  Configurable<std::string> fConfigElectronCuts{"cfgElectronCuts", "jpsiPID1", "Comma separated list of barrel track cuts"};

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    TString histNames = "";
    TString configCutNamesStr = fConfigElectronCuts.value;
    if (!configCutNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(configCutNamesStr.Tokenize(","));
      for (int icut = 0; icut < objArray->GetEntries(); ++icut) {
        fCutNames.push_back(objArray->At(icut)->GetName());
        histNames += Form("PairsBarrelSEPM_%s;PairsBarrelSEPP_%s;PairsBarrelSEMM_%s;", fCutNames[icut].Data(), fCutNames[icut].Data(), fCutNames[icut].Data());
        histNames += Form("PairsEleMuSEPM_%s;PairsEleMuSEPP_%s;PairsEleMuSEMM_%s;", fCutNames[icut].Data(), fCutNames[icut].Data(), fCutNames[icut].Data());
        fTwoTrackFilterMask |= (uint8_t(1) << icut);
      }
    }
    histNames += "PairsMuonSEPM;PairsMuonSEPP;PairsMuonSEMM;";

    DefineHistograms(fHistMan, histNames.Data());    // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    VarManager::SetupTwoProngDCAFitter(5.0f, true, 200.0f, 4.0f, 1.0e-3f, 0.9f, true); // TODO: get these parameters from Configurables
  }

  void process(soa::Filtered<MyEventsVtxCovSelected>::iterator const& event, soa::Filtered<MyBarrelTracksSelected> const& tracks, soa::Filtered<MyMuonTracksSelected> const& muons)
  {
    if (!event.isEventSelected()) {
      return;
    }
    // Reset the fValues array
    VarManager::ResetValues(0, VarManager::kNVars, fValues);
    VarManager::FillEvent<gkEventFillMap>(event, fValues);

    // Run the same event pairing for barrel tracks
    uint8_t twoTrackFilter = 0;
    uint32_t dileptonFilterMap = 0;
    for (auto& [t1, t2] : combinations(tracks, tracks)) {
      twoTrackFilter = t1.isBarrelSelected() & t2.isBarrelSelected() & fTwoTrackFilterMask;
      if (!twoTrackFilter) { // the tracks must have at least one filter bit in common to continue
        continue;
      }
      dileptonFilterMap = uint32_t(twoTrackFilter);
      VarManager::FillPair(t1, t2, fValues);
      VarManager::FillPairVertexing(event, t1, t2, fValues);
      dileptonList(event, fValues[VarManager::kMass], fValues[VarManager::kPt], fValues[VarManager::kEta], fValues[VarManager::kPhi], t1.sign() + t2.sign(), dileptonFilterMap);
      for (int i = 0; i < fCutNames.size(); ++i) {
        if (twoTrackFilter & (uint8_t(1) << i)) {
          if (t1.sign() * t2.sign() < 0) {
            fHistMan->FillHistClass(Form("PairsBarrelSEPM_%s", fCutNames[i].Data()), fValues);
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

    // same event pairing for muons
    for (auto& [muon1, muon2] : combinations(muons, muons)) {
      twoTrackFilter = muon1.isMuonSelected() & muon2.isMuonSelected();
      if (!twoTrackFilter) { // the muons must have at least one filter bit in common to continue
        continue;
      }
      // NOTE: The dimuons and electron-muon pairs in this task are pushed in the same table as the dielectrons.
      //       In order to discriminate them, the dileptonFilterMap uses the first 8 bits for dielectrons, the next 8 for dimuons and the rest for electron-muon
      // TBD:  Other implementations may be possible, for example add a column to the dilepton table to specify the pair type (dielectron, dimuon, electron-muon, etc.)
      dileptonFilterMap = uint32_t(twoTrackFilter) << 8;
      VarManager::FillPair(muon1, muon2, fValues, VarManager::kJpsiToMuMu);
      dileptonList(event, fValues[VarManager::kMass], fValues[VarManager::kPt], fValues[VarManager::kEta], fValues[VarManager::kPhi], muon1.sign() + muon2.sign(), dileptonFilterMap);
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

    for (auto& [trackBarrel, trackMuon] : combinations(tracks, muons)) {
      twoTrackFilter = (trackBarrel.isBarrelSelected() & fTwoTrackFilterMask) && trackMuon.isMuonSelected();
      if (!twoTrackFilter) { // the muon and barrel track must have at least one filter bit in common to continue
        continue;
      }
      // NOTE: The dimuons and electron-muon pairs in this task are pushed in the same table as the dielectrons.
      //       In order to discriminate them, the dileptonFilterMap uses the first 8 bits for dielectrons, the next 8 for dimuons and the rest for electron-muon
      dileptonFilterMap = uint32_t(twoTrackFilter) << 16;
      VarManager::FillPair(trackBarrel, trackMuon, fValues, VarManager::kElectronMuon);
      dileptonList(event, fValues[VarManager::kMass], fValues[VarManager::kPt], fValues[VarManager::kEta], fValues[VarManager::kPhi], trackBarrel.sign() + trackMuon.sign(), dileptonFilterMap);
      for (int i = 0; i < fCutNames.size(); ++i) {
        if (twoTrackFilter & (uint8_t(1) << i)) {
          if (trackBarrel.sign() * trackMuon.sign() < 0) {
            fHistMan->FillHistClass(Form("PairsEleMuSEPM_%s", fCutNames[i].Data()), fValues);
          } else {
            if (trackBarrel.sign() > 0) {
              fHistMan->FillHistClass(Form("PairsEleMuSEPP_%s", fCutNames[i].Data()), fValues);
            } else {
              fHistMan->FillHistClass(Form("PairsEleMuSEMM_%s", fCutNames[i].Data()), fValues);
            }
          }
        } // end if (filter bits)
      }   // end for (cuts)
    }     // end loop over electron-muon pairs
  }
};

struct DQDileptonHadronAnalysis {
  //
  // This task combines dilepton candidates with a track and could be used for example
  //  in analyses with the dilepton as one of the decay products of a higher mass resonance (e.g. B0 -> Jpsi + K)
  //    or in dilepton + hadron correlations, etc.
  // It requires the TableReader task to be in the workflow and produce the dilepton table
  //
  //  The barrel and muon track filtering tasks can produce multiple parallel decisions, which are used to produce
  //   dileptons which inherit the logical intersection of the track level decisions (see the TableReader task). This can be used
  //   also in the dilepton-hadron correlation analysis. However, in this model of the task, we use all the dileptons produced in the
  //     TableReader task to combine them with the hadrons selected by the barrel track selection.
  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;

  // use two values array to avoid mixing up the quantities
  float* fValuesDilepton;
  float* fValuesHadron;

  Filter eventFilter = aod::reducedevent::isEventSelected == 1;
  Filter dileptonFilter = aod::reducedpair::mass > 2.92f && aod::reducedpair::mass<3.16f && aod::reducedpair::pt> 0.0f && aod::reducedpair::sign == 0;
  // NOTE: the barrel track filter is shared between the filters for dilepton electron candidates (first n-bits)
  //       and the associated hadrons (n+1 bit) --> see the barrel track selection task
  //      The current condition should be replaced when bitwise operators will become available in Filter expresions
  // NOTE: the name of this configurable has to be the same as the one used in the barrel track selection task
  Configurable<std::string> fConfigElectronCuts{"cfgElectronCuts", "jpsiPID1", "Comma separated list of barrel track cuts"};
  int fNHadronCutBit;

  constexpr static uint32_t fgDileptonFillMap = VarManager::ObjTypes::ReducedTrack | VarManager::ObjTypes::Pair;

  void init(o2::framework::InitContext&)
  {
    fValuesDilepton = new float[VarManager::kNVars];
    fValuesHadron = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    DefineHistograms(fHistMan, "DileptonsSelected;DileptonHadronInvMass;DileptonHadronCorrelation"); // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars());
    fOutputList.setObject(fHistMan->GetMainHistogramList());

    TString configCutNamesStr = fConfigElectronCuts.value;
    if (!configCutNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(configCutNamesStr.Tokenize(","));
      fNHadronCutBit = objArray->GetEntries();
    } else {
      fNHadronCutBit = 0;
    }
  }

  void process(soa::Filtered<MyEventsVtxCovSelected>::iterator const& event, MyBarrelTracksSelected const& hadrons, soa::Filtered<aod::Dileptons> const& dileptons)
  {
    VarManager::ResetValues(0, VarManager::kNVars, fValuesHadron);
    VarManager::ResetValues(0, VarManager::kNVars, fValuesDilepton);
    // fill event information which might be needed in histograms that combine track/pair and event properties
    VarManager::FillEvent<gkEventFillMap>(event, fValuesHadron);
    VarManager::FillEvent<gkEventFillMap>(event, fValuesDilepton); // TODO: check if needed (just for dilepton QA which might be depending on event wise variables)

    // loop once over dileptons for QA purposes
    for (auto dilepton : dileptons) {
      VarManager::FillTrack<fgDileptonFillMap>(dilepton, fValuesDilepton);
      fHistMan->FillHistClass("DileptonsSelected", fValuesDilepton);
    }

    // loop over hadrons
    for (auto& hadron : hadrons) {
      if (!(hadron.isBarrelSelected() & (uint8_t(1) << fNHadronCutBit))) {
        continue;
      }
      for (auto dilepton : dileptons) {
        // TODO: At the moment there is no check on whether this hadron is one of the dilepton daughters!
        VarManager::FillDileptonHadron(dilepton, hadron, fValuesHadron);
        fHistMan->FillHistClass("DileptonHadronInvMass", fValuesHadron);
        fHistMan->FillHistClass("DileptonHadronCorrelation", fValuesHadron);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<DQEventSelection>(cfgc),
    adaptAnalysisTask<DQBarrelTrackSelection>(cfgc),
    adaptAnalysisTask<DQEventMixing>(cfgc),
    adaptAnalysisTask<DQMuonTrackSelection>(cfgc),
    adaptAnalysisTask<DQTableReader>(cfgc),
    adaptAnalysisTask<DQDileptonHadronAnalysis>(cfgc)};
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
      if (classStr.Contains("Barrel")) {
        dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "track", "its,tpcpid,dca,tofpid");
      }
      if (classStr.Contains("Muon")) {
        dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "track", "muon");
      }
    }

    if (classStr.Contains("Pairs")) {
      if (classStr.Contains("Barrel")) {
        dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "pair_barrel", "vertexing-barrel");
      }
      if (classStr.Contains("Muon")) {
        dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "pair_dimuon");
      }
      if (classStr.Contains("EleMu")) {
        dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "pair_electronmuon");
      }
    }

    if (classStr.Contains("DileptonsSelected")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "pair");
    }

    if (classStr.Contains("HadronsSelected")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "track", "kine");
    }

    if (classStr.Contains("DileptonHadronInvMass")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "dilepton-hadron-mass");
    }

    if (classStr.Contains("DileptonHadronCorrelation")) {
      dqhistograms::DefineHistograms(histMan, objArray->At(iclass)->GetName(), "dilepton-hadron-correlation");
    }
  } // end loop over histogram classes
}
