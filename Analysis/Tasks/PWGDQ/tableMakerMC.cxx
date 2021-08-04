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
// Analysis task for skimming MC AODs
// Similar to tableMaker.cxx. The written skimmed data model includes, besides the reconstructed data tables, a skimmed MC stack.
//   The skimmed MC stack includes the MC truth particles corresponding to the list of user specified MC signals (see MCsignal.h)
//    and the MC truth particles corresponding to the reconstructed tracks selected by the specified track cuts on reconstructed data.
//   The event MC truth is written as a joinable table with the ReducedEvents table (!simplification wrt Framework data model).
// TODO: There are no MC labels for MUON tracks yet

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/DataTypes.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisCore/TriggerAliases.h"
#include "AnalysisDataModel/ReducedInfoTables.h"
#include "PWGDQCore/VarManager.h"
#include "PWGDQCore/HistogramManager.h"
#include "PWGDQCore/AnalysisCut.h"
#include "PWGDQCore/AnalysisCompositeCut.h"
#include "PWGDQCore/HistogramsLibrary.h"
#include "PWGDQCore/CutsLibrary.h"
#include "PWGDQCore/MCSignal.h"
#include "PWGDQCore/MCSignalLibrary.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "TList.h"
#include <iostream>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod;

#include "Framework/runDataProcessing.h"

using MyBarrelTracks = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksExtended, aod::TrackSelection,
                                 aod::pidTPCFullEl, aod::pidTPCFullMu, aod::pidTPCFullPi,
                                 aod::pidTPCFullKa, aod::pidTPCFullPr,
                                 aod::pidTOFFullEl, aod::pidTOFFullMu, aod::pidTOFFullPi,
                                 aod::pidTOFFullKa, aod::pidTOFFullPr, aod::pidTOFbeta,
                                 aod::McTrackLabels>;
//using MyEvents = soa::Join<aod::Collisions, aod::EvSels, aod::Cents, aod::McCollisionLabels>;
using MyEvents = soa::Join<aod::Collisions, aod::EvSels, aod::McCollisionLabels>; // TODO: remove centrality temporarily

// HACK: In order to be able to deduce which kind of aod object is transmitted to the templated VarManager::Fill functions
//         a constexpr static bit map must be defined and sent as template argument
//        The user has to include in this bit map all the tables used in analysis, as defined in VarManager::ObjTypes
//        Additionally, one should make sure that the requested tables are actually provided in the process() function,
//       otherwise a compile time error will be thrown.
//        This is a temporary fix until the arrow/ROOT issues are solved, at which point it will be possible
//           to automatically detect the object types transmitted to the VarManager
//constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::BC | VarManager::ObjTypes::Collision | VarManager::ObjTypes::CollisionCent;
constexpr static uint32_t gkEventFillMap = VarManager::ObjTypes::BC | VarManager::ObjTypes::Collision;
constexpr static uint32_t gkEventMCFillMap = VarManager::ObjTypes::CollisionMC;
constexpr static uint32_t gkTrackFillMap = VarManager::ObjTypes::Track | VarManager::ObjTypes::TrackExtra | VarManager::ObjTypes::TrackDCA | VarManager::ObjTypes::TrackSelection | VarManager::ObjTypes::TrackCov | VarManager::ObjTypes::TrackPID;
constexpr static uint32_t gkParticleMCFillMap = VarManager::ObjTypes::ParticleMC;

struct TableMakerMC {

  Produces<ReducedEvents> event;
  Produces<ReducedEventsExtended> eventExtended;
  Produces<ReducedEventsVtxCov> eventVtxCov;
  Produces<ReducedEventsMC> eventMC;
  Produces<ReducedTracks> trackBasic;
  Produces<ReducedTracksBarrel> trackBarrel;
  Produces<ReducedTracksBarrelCov> trackBarrelCov;
  Produces<ReducedTracksBarrelPID> trackBarrelPID;
  Produces<ReducedTracksBarrelLabels> trackBarrelLabels;
  Produces<ReducedMCTracks> trackMC;

  float* fValues;

  // temporary variables used for the indexing of the skimmed MC stack
  std::map<uint64_t, int> fNewLabels;
  std::map<uint64_t, int> fNewLabelsReversed;
  std::map<uint64_t, int16_t> fMCFlags;
  std::map<uint64_t, int> fEventIdx;
  int fCounter;

  // list of MCsignal objects
  std::vector<MCSignal> fMCSignals;

  OutputObj<THashList> fOutputList{"output"};
  HistogramManager* fHistMan;

  Configurable<std::string> fConfigEventCuts{"cfgEventCuts", "eventStandard", "Event selection"};
  Configurable<std::string> fConfigTrackCuts{"cfgBarrelTrackCuts", "jpsiPID1", "barrel track cut"};
  Configurable<float> fConfigBarrelTrackPtLow{"cfgBarrelLowPt", 1.0f, "Low pt cut for tracks in the barrel"};
  Configurable<std::string> fConfigMCSignals{"cfgMCsignals", "", "Comma separated list of MC signals"};

  // TODO: Enable multiple track selections in parallel, and decisions stored in the respective bit fields
  AnalysisCompositeCut* fEventCut;
  AnalysisCompositeCut* fTrackCut;

  // TODO: filter on TPC dedx used temporarily until electron PID will be improved
  Filter barrelSelectedTracks = aod::track::trackType == uint8_t(aod::track::Run2Track) && o2::aod::track::pt >= fConfigBarrelTrackPtLow && nabs(o2::aod::track::eta) <= 0.9f && o2::aod::track::tpcSignal >= 70.0f && o2::aod::track::tpcSignal <= 100.0f && o2::aod::track::tpcChi2NCl < 4.0f && o2::aod::track::itsChi2NCl < 36.0f;

  void init(o2::framework::InitContext&)
  {
    fValues = new float[VarManager::kNVars];
    VarManager::SetDefaultVarNames();
    fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
    fHistMan->SetUseDefaultVariableNames(kTRUE);
    fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);

    fCounter = 0;
    fNewLabels.clear();
    fNewLabelsReversed.clear();
    fMCFlags.clear();
    fEventIdx.clear();

    TString histClasses = "Event_BeforeCuts;Event_AfterCuts;TrackBarrel_BeforeCuts;TrackBarrel_AfterCuts;";

    TString configNamesStr = fConfigMCSignals.value;
    if (!configNamesStr.IsNull()) {
      std::unique_ptr<TObjArray> objArray(configNamesStr.Tokenize(","));
      for (int isig = 0; isig < objArray->GetEntries(); ++isig) {
        MCSignal* sig = o2::aod::dqmcsignals::GetMCSignal(objArray->At(isig)->GetName());
        if (sig) {
          fMCSignals.push_back(*sig);
          histClasses += Form("TrackBarrel_AfterCuts_%s;MCtruth_%s;", objArray->At(isig)->GetName(), objArray->At(isig)->GetName());
        }
      }
    }

    DefineHistograms(histClasses);                   // define all histograms
    VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
    fOutputList.setObject(fHistMan->GetMainHistogramList());
    DefineCuts();

    cout << "===================== Print configured signals ==========================" << endl;
    for (MCSignal sig : fMCSignals) {
      sig.Print();
    }
  }

  void DefineCuts()
  {
    fEventCut = new AnalysisCompositeCut(true);
    TString eventCutStr = fConfigEventCuts.value;
    fEventCut->AddCut(dqcuts::GetAnalysisCut(eventCutStr.Data()));

    // NOTE: for now, the model of this task is that just one track cut is applied; multiple parallel cuts should be enabled in the future
    fTrackCut = new AnalysisCompositeCut(true);
    TString trackCutStr = fConfigTrackCuts.value;
    fTrackCut->AddCut(dqcuts::GetCompositeCut(trackCutStr.Data()));

    VarManager::SetUseVars(AnalysisCut::fgUsedVars); // provide the list of required variables so that VarManager knows what to fill
  }

  void processRec(MyEvents::iterator const& collision, aod::BCs const& bcs, soa::Filtered<MyBarrelTracks> const& tracksBarrel,
                  aod::McParticles const& mcTracks, aod::McCollisions const& mcEvents)
  {
    uint32_t triggerAliases = 0;
    for (int i = 0; i < kNaliases; i++) {
      if (collision.alias()[i] > 0) {
        triggerAliases |= (uint32_t(1) << i);
      }
    }
    uint64_t tag = 0; // TODO: add here available computed event cuts (e.g. run2bcinfo().eventCuts()) or other event wise decisions

    VarManager::ResetValues(0, VarManager::kNEventWiseVariables, fValues);
    VarManager::FillEvent<gkEventFillMap>(collision, fValues); // extract event information and place it in the fValues array
    VarManager::FillEvent<gkEventMCFillMap>(collision.mcCollision(), fValues);

    fHistMan->FillHistClass("Event_BeforeCuts", fValues); // automatically fill all the histograms in the class Event
    if (!fEventCut->IsSelected(fValues)) {
      return;
    }
    fHistMan->FillHistClass("Event_AfterCuts", fValues);

    event(tag, collision.bc().runNumber(), collision.posX(), collision.posY(), collision.posZ(), collision.numContrib());
    eventExtended(collision.bc().globalBC(), collision.bc().triggerMask(), 0, triggerAliases, fValues[VarManager::kCentVZERO]);
    eventVtxCov(collision.covXX(), collision.covXY(), collision.covXZ(), collision.covYY(), collision.covYZ(), collision.covZZ(), collision.chi2());
    eventMC(collision.mcCollision().generatorsID(), collision.mcCollision().posX(), collision.mcCollision().posY(),
            collision.mcCollision().posZ(), collision.mcCollision().t(), collision.mcCollision().weight(), collision.mcCollision().impactParameter());

    // loop over the MC truth tracks and find those that need to be written
    uint16_t mcflags = 0;
    for (auto& mctrack : mcTracks) {
      // grouping will be needed here
      if (mctrack.mcCollision().globalIndex() != collision.mcCollision().globalIndex()) {
        continue;
      }

      // check all the requested MC signals and fill a decision bit map
      mcflags = 0;
      int i = 0;
      for (auto& sig : fMCSignals) {
        if (sig.CheckSignal(true, mcTracks, mctrack)) {
          mcflags |= (uint16_t(1) << i);
        }
        i++;
      }
      // if any of the MC signals was matched, then fill histograms and write that MC particle into the new stack
      if (mcflags) {
        // fill histograms for each of the signals, if found
        VarManager::FillTrack<gkParticleMCFillMap>(mctrack, fValues);
        for (int i = 0; i < fMCSignals.size(); i++) {
          if (mcflags & (uint16_t(1) << i)) {
            fHistMan->FillHistClass(Form("MCtruth_%s", fMCSignals[i].GetName()), fValues);
          }
        }

        fNewLabels[mctrack.index()] = fCounter;
        fNewLabelsReversed[fCounter] = mctrack.index();
        fMCFlags[mctrack.index()] = mcflags;
        fEventIdx[mctrack.index()] = event.lastIndex();
        fCounter++;
      }
    } // end loop over mc stack

    // loop over reconstructed tracks
    uint64_t trackFilteringTag = 0;
    //uint16_t mcflags = 0;
    trackBasic.reserve(tracksBarrel.size());
    trackBarrel.reserve(tracksBarrel.size());
    trackBarrelCov.reserve(tracksBarrel.size());
    trackBarrelPID.reserve(tracksBarrel.size());
    trackBarrelLabels.reserve(tracksBarrel.size());
    for (auto& track : tracksBarrel) {
      VarManager::FillTrack<gkTrackFillMap>(track, fValues);

      fHistMan->FillHistClass("TrackBarrel_BeforeCuts", fValues);
      if (!fTrackCut->IsSelected(fValues)) {
        continue;
      }
      auto mctrack = track.mcParticle();
      VarManager::FillTrack<gkParticleMCFillMap>(mctrack, fValues);
      fHistMan->FillHistClass("TrackBarrel_AfterCuts", fValues);

      // if the MC truth particle corresponding to this reconstructed track is not already written,
      //   add it to the skimmed stack
      mcflags = 0;
      if (!(fNewLabels.find(mctrack.index()) != fNewLabels.end())) {
        int i = 0;
        // check all the specified signals
        for (auto& sig : fMCSignals) {
          if (sig.CheckSignal(true, mcTracks, mctrack)) {
            mcflags |= (uint16_t(1) << i);
            fHistMan->FillHistClass(Form("TrackBarrel_AfterCuts_%s", sig.GetName()), fValues); // fill the reconstructed truth
            fHistMan->FillHistClass(Form("MCtruth_%s", sig.GetName()), fValues);               // fill the generated truth
          }
          i++;
        }

        fNewLabels[mctrack.index()] = fCounter;
        fNewLabelsReversed[fCounter] = mctrack.index();
        fMCFlags[mctrack.index()] = mcflags;
        fEventIdx[mctrack.index()] = event.lastIndex();
        fCounter++;
      }

      if (track.isGlobalTrack()) {
        trackFilteringTag |= (uint64_t(1) << 0);
      }
      if (track.isGlobalTrackSDD()) {
        trackFilteringTag |= (uint64_t(1) << 1);
      }
      trackBasic(event.lastIndex(), track.globalIndex(), trackFilteringTag, track.pt(), track.eta(), track.phi(), track.sign());
      trackBarrel(track.tpcInnerParam(), track.flags(), track.itsClusterMap(), track.itsChi2NCl(),
                  track.tpcNClsFindable(), track.tpcNClsFindableMinusFound(), track.tpcNClsFindableMinusCrossedRows(),
                  track.tpcNClsShared(), track.tpcChi2NCl(),
                  track.trdChi2(), track.trdPattern(), track.tofChi2(),
                  track.length(), track.dcaXY(), track.dcaZ());
      trackBarrelCov(track.x(), track.alpha(), track.y(), track.z(), track.snp(), track.tgl(), track.signed1Pt(),
                     track.cYY(), track.cZY(), track.cZZ(), track.cSnpY(), track.cSnpZ(),
                     track.cSnpSnp(), track.cTglY(), track.cTglZ(), track.cTglSnp(), track.cTglTgl(),
                     track.c1PtY(), track.c1PtZ(), track.c1PtSnp(), track.c1PtTgl(), track.c1Pt21Pt2());
      trackBarrelPID(track.tpcSignal(),
                     track.tpcNSigmaEl(), track.tpcNSigmaMu(),
                     track.tpcNSigmaPi(), track.tpcNSigmaKa(), track.tpcNSigmaPr(),
                     track.beta(),
                     track.tofNSigmaEl(), track.tofNSigmaMu(),
                     track.tofNSigmaPi(), track.tofNSigmaKa(), track.tofNSigmaPr(),
                     track.trdSignal());
      trackBarrelLabels(fNewLabels.find(mctrack.index())->second, track.mcMask(), mcflags);
    } // end loop over reconstructed tracks
  }

  void processWriteStack(aod::McParticles const& mcTracks)
  {
    // loop over the selected MC tracks and write them to the new stack
    // NOTE: At this point, all the particles which should go into the new stack should had been
    //     selected, such that the mother/daughter new labels can be computed
    // NOTE: We need to write the particles to the new stack in the exact same order as the
    //   one in which the new labels were assigned. So we use the fNewLabelsReversed which has as key the new label
    //   and as value the old stack label, which we then use to retrieve the particle to be written
    for (const auto& [newLabel, oldLabel] : fNewLabelsReversed) {
      auto mctrack = mcTracks.iteratorAt(oldLabel);
      int mcflags = fMCFlags.find(oldLabel)->second;

      int m0Label = -1;
      if (mctrack.has_mother0() && (fNewLabels.find(mctrack.mother0Id()) != fNewLabels.end())) {
        m0Label = fNewLabels.find(mctrack.mother0Id())->second;
      }
      int m1Label = -1;
      if (mctrack.has_mother1() && (fNewLabels.find(mctrack.mother1Id()) != fNewLabels.end())) {
        m1Label = fNewLabels.find(mctrack.mother1Id())->second;
      }
      int d0Label = -1;
      if (mctrack.has_daughter0() && (fNewLabels.find(mctrack.daughter0Id()) != fNewLabels.end())) {
        d0Label = fNewLabels.find(mctrack.daughter0Id())->second;
      }
      int d1Label = -1;
      if (mctrack.has_daughter1() && (fNewLabels.find(mctrack.daughter1Id()) != fNewLabels.end())) {
        d1Label = fNewLabels.find(mctrack.daughter1Id())->second;
      }
      trackMC(fEventIdx.find(oldLabel)->second, mctrack.pdgCode(), mctrack.statusCode(), mctrack.flags(),
              m0Label, m1Label, d0Label, d1Label,
              mctrack.weight(), mctrack.pt(), mctrack.eta(), mctrack.phi(), mctrack.e(),
              mctrack.vx(), mctrack.vy(), mctrack.vz(), mctrack.vt(), mcflags);
    }

    fCounter = 0;
    fNewLabels.clear();
    fNewLabelsReversed.clear();
    fMCFlags.clear();
    fEventIdx.clear();
  }

  void DefineHistograms(TString histClasses)
  {
    std::unique_ptr<TObjArray> objArray(histClasses.Tokenize(";"));
    for (Int_t iclass = 0; iclass < objArray->GetEntries(); ++iclass) {
      TString classStr = objArray->At(iclass)->GetName();
      fHistMan->AddHistClass(classStr.Data());

      // NOTE: The level of detail for histogramming can be controlled via configurables
      if (classStr.Contains("Event")) {
        dqhistograms::DefineHistograms(fHistMan, objArray->At(iclass)->GetName(), "event", "triggerall,cent,mc");
      }

      if (classStr.Contains("Track")) {
        if (classStr.Contains("BeforeCuts")) {
          dqhistograms::DefineHistograms(fHistMan, objArray->At(iclass)->GetName(), "track", "dca,its,tpcpid,tofpid");
        } else {
          dqhistograms::DefineHistograms(fHistMan, objArray->At(iclass)->GetName(), "track", "dca,its,tpcpid,tofpid,mc");
        }
      }
      if (classStr.Contains("MCtruth")) {
        dqhistograms::DefineHistograms(fHistMan, objArray->At(iclass)->GetName(), "mctruth");
      }
    }
  }

  template <typename T>
  void PrintBits(T bitMap, int n)
  {
    for (int i = 0; i < n; i++) {
      if (i < sizeof(T) * 8) {
        cout << (bitMap & (T(1) << i) ? "1" : "0");
      }
    }
  };

  PROCESS_SWITCH(TableMakerMC, processRec, "Process reco level", true);
  PROCESS_SWITCH(TableMakerMC, processWriteStack, "Write the skimmed MC stack", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  // TODO: For now TableMakerMC works just for PbPb (cent table is present)
  //      Implement workflow arguments for pp/PbPb and possibly merge the task with tableMaker.cxx
  return WorkflowSpec{
    adaptAnalysisTask<TableMakerMC>(cfgc)};
}
