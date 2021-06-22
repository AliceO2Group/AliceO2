// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include <CCDB/BasicCCDBManager.h>
#include "Framework/StepTHn.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/RunningWorkflowInfo.h"

#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisCore/CorrelationContainer.h"
#include "AnalysisCore/PairCuts.h"

#include <TH1F.h>
#include <cmath>
#include <TDirectory.h>
#include <THn.h>

namespace o2::aod
{
namespace hash
{
DECLARE_SOA_COLUMN(Bin, bin, int);
} // namespace hash
DECLARE_SOA_TABLE(Hashes, "AOD", "HASH", hash::Bin);

using Hash = Hashes::iterator;
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#define O2_DEFINE_CONFIGURABLE(NAME, TYPE, DEFAULT, HELP) Configurable<TYPE> NAME{#NAME, DEFAULT, HELP};

struct CorrelationTask {

  // Configuration
  O2_DEFINE_CONFIGURABLE(cfgCutVertex, float, 7.0f, "Accepted z-vertex range")
  O2_DEFINE_CONFIGURABLE(cfgCutPt, float, 0.5f, "Minimal pT for tracks")
  O2_DEFINE_CONFIGURABLE(cfgCutEta, float, 0.8f, "Eta range for tracks")

  O2_DEFINE_CONFIGURABLE(cfgPtOrder, int, 1, "Only consider pairs for which pT,1 < pT,2 (0 = OFF, 1 = ON)");
  O2_DEFINE_CONFIGURABLE(cfgTriggerCharge, int, 0, "Select on charge of trigger particle: 0 = all; 1 = positive; -1 = negative");
  O2_DEFINE_CONFIGURABLE(cfgAssociatedCharge, int, 0, "Select on charge of associated particle: 0 = all; 1 = positive; -1 = negative");
  O2_DEFINE_CONFIGURABLE(cfgPairCharge, int, 0, "Select on charge of particle pair: 0 = all; 1 = like sign; -1 = unlike sign");

  O2_DEFINE_CONFIGURABLE(cfgTwoTrackCut, float, -1, "Two track cut: -1 = off; >0 otherwise distance value (suggested: 0.02)");
  O2_DEFINE_CONFIGURABLE(cfgTwoTrackCutMinRadius, float, 0.8f, "Two track cut: radius in m from which two track cuts are applied");

  O2_DEFINE_CONFIGURABLE(cfgPairCutPhoton, float, -1, "Pair cut on photons: -1 = off; >0 otherwise distance value (suggested: 0.004)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutK0, float, -1, "Pair cut on K0s: -1 = off; >0 otherwise distance value (suggested: 0.005)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutLambda, float, -1, "Pair cut on Lambda: -1 = off; >0 otherwise distance value (suggested: 0.005)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutPhi, float, -1, "Pair cut on Phi: -1 = off; >0 otherwise distance value")
  O2_DEFINE_CONFIGURABLE(cfgPairCutRho, float, -1, "Pair cut on Rho: -1 = off; >0 otherwise distance value")

  O2_DEFINE_CONFIGURABLE(cfgEfficiencyTrigger, std::string, "", "CCDB path to efficiency object for trigger particles")
  O2_DEFINE_CONFIGURABLE(cfgEfficiencyAssociated, std::string, "", "CCDB path to efficiency object for associated particles")

  O2_DEFINE_CONFIGURABLE(cfgNoMixedEvents, int, 5, "Number of mixed events per event")

  ConfigurableAxis axisVertex{"axisVertex", {7, -7, 7}, "vertex axis for histograms"};
  ConfigurableAxis axisDeltaPhi{"axisDeltaPhi", {72, -M_PI / 2, M_PI / 2 * 3}, "delta phi axis for histograms"};
  ConfigurableAxis axisDeltaEta{"axisDeltaEta", {40, -2, 2}, "delta eta axis for histograms"};
  ConfigurableAxis axisPtTrigger{"axisPtTrigger", {VARIABLE_WIDTH, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0}, "pt trigger axis for histograms"};
  ConfigurableAxis axisPtAssoc{"axisPtAssoc", {VARIABLE_WIDTH, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, "pt associated axis for histograms"};
  ConfigurableAxis axisMultiplicity{"axisMultiplicity", {VARIABLE_WIDTH, 0, 5, 10, 20, 30, 40, 50, 100.1}, "multiplicity / centrality axis for histograms"};

  ConfigurableAxis axisVertexEfficiency{"axisVertexEfficiency", {10, -10, 10}, "vertex axis for efficiency histograms"};
  ConfigurableAxis axisEtaEfficiency{"axisEtaEfficiency", {20, -1.0, 1.0}, "eta axis for efficiency histograms"};
  ConfigurableAxis axisPtEfficiency{"axisPtEfficiency", {VARIABLE_WIDTH, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0}, "pt axis for efficiency histograms"};

  // Filters and input definitions
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  // TODO bitwise operations not supported, yet
  // Filter vertexTypeFilter = aod::collision::flags & (uint16_t) aod::collision::CollisionFlagsRun2::Run2VertexerTracks;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::pt > cfgCutPt) && ((aod::track::isGlobalTrack == (uint8_t) true) || (aod::track::isGlobalTrackSDD == (uint8_t) true));
  using myTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>>;

  // Output definitions
  OutputObj<CorrelationContainer> same{"sameEvent"};
  OutputObj<CorrelationContainer> mixed{"mixedEvent"};

  struct Config {
    bool mPairCuts = false;
    THn* mEfficiencyTrigger = nullptr;
    THn* mEfficiencyAssociated = nullptr;
  } cfg;

  HistogramRegistry registry{"registry"};
  PairCuts mPairCuts;

  Service<o2::ccdb::BasicCCDBManager> ccdb;

  void init(o2::framework::InitContext&)
  {
    registry.add("yields", "centrality vs pT vs eta", {HistType::kTH3F, {{100, 0, 100, "centrality"}, {40, 0, 20, "p_{T}"}, {100, -2, 2, "#eta"}}});
    registry.add("etaphi", "centrality vs eta vs phi", {HistType::kTH3F, {{100, 0, 100, "centrality"}, {100, -2, 2, "#eta"}, {200, 0, 2 * M_PI, "#varphi"}}});

    mPairCuts.SetHistogramRegistry(&registry);

    if (cfgPairCutPhoton > 0 || cfgPairCutK0 > 0 || cfgPairCutLambda > 0 || cfgPairCutPhi > 0 || cfgPairCutRho > 0) {
      mPairCuts.SetPairCut(PairCuts::Photon, cfgPairCutPhoton);
      mPairCuts.SetPairCut(PairCuts::K0, cfgPairCutK0);
      mPairCuts.SetPairCut(PairCuts::Lambda, cfgPairCutLambda);
      mPairCuts.SetPairCut(PairCuts::Phi, cfgPairCutPhi);
      mPairCuts.SetPairCut(PairCuts::Rho, cfgPairCutRho);
      cfg.mPairCuts = true;
    }

    if (cfgTwoTrackCut > 0) {
      mPairCuts.SetTwoTrackCuts(cfgTwoTrackCut, cfgTwoTrackCutMinRadius);
    }

    // --- OBJECT INIT ---

    std::vector<AxisSpec> axisList = {{axisDeltaEta, "#Delta#eta"},
                                      {axisPtAssoc, "p_{T} (GeV/c)"},
                                      {axisPtTrigger, "p_{T} (GeV/c)"},
                                      {axisMultiplicity, "multiplicity / centrality"},
                                      {axisDeltaPhi, "#Delta#varphi (rad)"},
                                      {axisVertex, "z-vtx (cm)"},
                                      {axisEtaEfficiency, "#eta"},
                                      {axisPtEfficiency, "p_{T} (GeV/c)"},
                                      {axisVertexEfficiency, "z-vtx (cm)"}};
    same.setObject(new CorrelationContainer("sameEvent", "sameEvent", axisList));
    mixed.setObject(new CorrelationContainer("mixedEvent", "mixedEvent", axisList));

    // o2-ccdb-upload -p Users/jgrosseo/correlations/LHC15o -f /tmp/correction_2011_global.root -k correction

    ccdb->setURL("http://ccdb-test.cern.ch:8080");
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();

    long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    ccdb->setCreatedNotAfter(now); // TODO must become global parameter from the train creation time

    if (cfgEfficiencyTrigger.value.empty() == false) {
      cfg.mEfficiencyTrigger = ccdb->getForTimeStamp<THnT<float>>(cfgEfficiencyTrigger, now);
      LOGF(info, "Loaded efficiency histogram for trigger particles from %s (%p)", cfgEfficiencyTrigger.value.c_str(), (void*)cfg.mEfficiencyTrigger);
    }
    if (cfgEfficiencyAssociated.value.empty() == false) {
      cfg.mEfficiencyAssociated = ccdb->getForTimeStamp<THnT<float>>(cfgEfficiencyAssociated, now);
      LOGF(info, "Loaded efficiency histogram for associated particles from %s (%p)", cfgEfficiencyAssociated.value.c_str(), (void*)cfg.mEfficiencyAssociated);
    }
  }

  template <typename TTarget, typename TCollision>
  bool fillCollision(TTarget target, TCollision collision, float centrality)
  {
    same->fillEvent(centrality, CorrelationContainer::kCFStepAll);

    if (!collision.alias()[kINT7] || !collision.sel7()) {
      return false;
    }

    same->fillEvent(centrality, CorrelationContainer::kCFStepTriggered);

    // vertex range already checked as filter, but bitwise operations not yet supported
    // TODO (collision.flags() != 0) can be removed with next conversion (AliPhysics >= 20210305)
    if ((collision.flags() != 0) && ((collision.flags() & aod::collision::CollisionFlagsRun2::Run2VertexerTracks) != aod::collision::CollisionFlagsRun2::Run2VertexerTracks)) {
      return false;
    }

    same->fillEvent(centrality, CorrelationContainer::kCFStepVertex);
    same->fillEvent(centrality, CorrelationContainer::kCFStepReconstructed);

    return true;
  }

  template <typename TTarget, typename TTracks>
  void fillCorrelations(TTarget target, TTracks tracks1, TTracks tracks2, float centrality, float posZ, int bSign)
  {
    // Cache efficiency for particles (too many FindBin lookups)
    float* efficiencyAssociated = nullptr;
    if (cfg.mEfficiencyAssociated) {
      efficiencyAssociated = new float[tracks2.size()];
      int i = 0;
      for (auto& track : tracks2) {
        efficiencyAssociated[i++] = getEfficiency(cfg.mEfficiencyAssociated, track.eta(), track.pt(), centrality, posZ);
      }
    }

    for (auto& track1 : tracks1) {
      // LOGF(info, "Track %f | %f | %f  %d %d", track1.eta(), track1.phi(), track1.pt(), track1.isGlobalTrack(), track1.isGlobalTrackSDD());

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.sign() < 0) {
        continue;
      }

      float triggerWeight = 1.0;
      if (cfg.mEfficiencyTrigger) {
        triggerWeight = getEfficiency(cfg.mEfficiencyTrigger, track1.eta(), track1.pt(), centrality, posZ);
      }

      target->getTriggerHist()->Fill(CorrelationContainer::kCFStepReconstructed, track1.pt(), centrality, posZ, triggerWeight);

      int i = -1;
      for (auto& track2 : tracks2) {
        i++; // HACK
        if (track1 == track2) {
          continue;
        }

        if (cfgPtOrder != 0 && track2.pt() >= track1.pt()) {
          continue;
        }

        if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.sign() < 0) {
          continue;
        }
        if (cfgPairCharge != 0 && cfgPairCharge * track1.sign() * track2.sign() < 0) {
          continue;
        }

        if (cfg.mPairCuts && mPairCuts.conversionCuts(track1, track2)) {
          continue;
        }

        if (cfgTwoTrackCut > 0 && mPairCuts.twoTrackCut(track1, track2, bSign)) {
          continue;
        }

        float associatedWeight = 1.0;
        if (cfg.mEfficiencyAssociated) {
          associatedWeight = efficiencyAssociated[i];
        }

        float deltaPhi = track1.phi() - track2.phi();
        if (deltaPhi > 1.5 * M_PI) {
          deltaPhi -= M_PI * 2;
        }
        if (deltaPhi < -0.5 * M_PI) {
          deltaPhi += M_PI * 2;
        }

        target->getPairHist()->Fill(CorrelationContainer::kCFStepReconstructed,
                                    track1.eta() - track2.eta(), track2.pt(), track1.pt(), centrality, deltaPhi, posZ,
                                    triggerWeight * associatedWeight);
      }
    }

    delete[] efficiencyAssociated;
  }

  // Version with explicit nested loop
  void processSame(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents>>::iterator const& collision, aod::BCsWithTimestamps const&, myTracks const& tracks)
  {
    auto bc = collision.bc_as<aod::BCsWithTimestamps>();
    if (cfgEfficiencyTrigger.value.empty() == false) {
      cfg.mEfficiencyTrigger = ccdb->getForTimeStamp<THnT<float>>(cfgEfficiencyTrigger, bc.timestamp());
      LOGF(info, "Loaded efficiency histogram for trigger particles from %s (%p)", cfgEfficiencyTrigger.value.c_str(), (void*)cfg.mEfficiencyTrigger);
    }
    if (cfgEfficiencyAssociated.value.empty() == false) {
      cfg.mEfficiencyAssociated = ccdb->getForTimeStamp<THnT<float>>(cfgEfficiencyAssociated, bc.timestamp());
      LOGF(info, "Loaded efficiency histogram for associated particles from %s (%p)", cfgEfficiencyAssociated.value.c_str(), (void*)cfg.mEfficiencyAssociated);
    }

    LOGF(info, "Tracks for collision: %d | Vertex: %.1f | INT7: %d | V0M: %.1f", tracks.size(), collision.posZ(), collision.sel7(), collision.centV0M());

    if (std::abs(collision.posZ()) > cfgCutVertex) {
      LOGF(warning, "Unexpected: Vertex %f outside of cut %f", collision.posZ(), cfgCutVertex);
    }

    int bSign = 1; // TODO magnetic field from CCDB
    const auto centrality = collision.centV0M();

    if (fillCollision(same, collision, centrality) == false) {
      return;
    }

    for (auto& track1 : tracks) {
      registry.fill(HIST("yields"), centrality, track1.pt(), track1.eta());
      registry.fill(HIST("etaphi"), centrality, track1.eta(), track1.phi());
    }

    fillCorrelations(same, tracks, tracks, centrality, collision.posZ(), bSign);
  }

  void processMixed(soa::Join<aod::Collisions, aod::Hashes, aod::EvSels, aod::Cents>& collisions, myTracks const& tracks)
  {
    // TODO loading of efficiency histogram missing here, because it will happen somehow in the CCDBConfigurable

    int bSign = 1; // TODO magnetic field from CCDB

    collisions.bindExternalIndices(&tracks);
    auto tracksTuple = std::make_tuple(tracks);
    AnalysisDataProcessorBuilder::GroupSlicer slicer(collisions, tracksTuple);

    // Strictly upper categorised collisions, for cfgNoMixedEvents combinations per bin, skipping those in entry -1
    for (auto& [collision1, collision2] : selfCombinations("fBin", cfgNoMixedEvents, -1, collisions, collisions)) {

      LOGF(info, "Mixed collisions bin: %d pair: %d (%f), %d (%f)", collision1.bin(), collision1.index(), collision1.posZ(), collision2.index(), collision2.posZ());

      // TODO in principle these should be already checked on hash level, because in this way we don't check collision 2
      if (fillCollision(mixed, collision1, collision1.centV0M()) == false) {
        continue;
      }

      auto it1 = slicer.begin();
      auto it2 = slicer.begin();
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == collision1.index()) {
          it1 = slice;
          break;
        }
      }
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == collision2.index()) {
          it2 = slice;
          break;
        }
      }

      auto tracks1 = std::get<myTracks>(it1.associatedTables());
      tracks1.bindExternalIndices(&collisions);
      auto tracks2 = std::get<myTracks>(it2.associatedTables());
      tracks2.bindExternalIndices(&collisions);

      // LOGF(info, "Tracks: %d and %d entries", tracks1.size(), tracks2.size());

      fillCorrelations(mixed, tracks1, tracks2, collision1.centV0M(), collision1.posZ(), bSign);
    }
  }

  // Version with combinations
  void processWithCombinations(soa::Join<aod::Collisions, aod::Cents>::iterator const& collision, soa::Filtered<aod::Tracks> const& tracks)
  {
    LOGF(info, "Tracks for collision (Combination run): %d", tracks.size());

    const auto centrality = collision.centV0M();

    int bSign = 1; // TODO magnetic field from CCDB

    for (auto track1 = tracks.begin(); track1 != tracks.end(); ++track1) {

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.sign() < 0) {
        continue;
      }

      //       LOGF(info, "TRACK %f %f | %f %f | %f %f", track1.eta(), track1.eta(), track1.phi(), track1.phi2(), track1.pt(), track1.pt());

      same->getTriggerHist()->Fill(CorrelationContainer::kCFStepReconstructed, track1.pt(), centrality, collision.posZ());
      //mixed->getTriggerHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);
    }

    for (auto& [track1, track2] : combinations(tracks, tracks)) {
      //LOGF(info, "Combination %d %d", track1.index(), track2.index());

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.sign() < 0) {
        continue;
      }
      if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.sign() < 0) {
        continue;
      }
      if (cfgPairCharge != 0 && cfgPairCharge * track1.sign() * track2.sign() < 0) {
        continue;
      }

      if (cfg.mPairCuts && mPairCuts.conversionCuts(track1, track2)) {
        continue;
      }

      if (cfgTwoTrackCut > 0 && mPairCuts.twoTrackCut(track1, track2, bSign)) {
        continue;
      }

      float deltaPhi = track1.phi() - track2.phi();
      if (deltaPhi > 1.5 * M_PI) {
        deltaPhi -= M_PI * 2;
      }
      if (deltaPhi < -0.5 * M_PI) {
        deltaPhi += M_PI * 2;
      }

      same->getPairHist()->Fill(CorrelationContainer::kCFStepReconstructed,
                                track1.eta() - track2.eta(), track2.pt(), track1.pt(), centrality, deltaPhi, collision.posZ());
      //mixed->getPairHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
    }
  }

  double getEfficiency(THn* eff, float eta, float pt, float centrality, float posZ)
  {
    int effVars[4];
    effVars[0] = eff->GetAxis(0)->FindBin(eta);
    effVars[1] = eff->GetAxis(1)->FindBin(pt);
    effVars[2] = eff->GetAxis(2)->FindBin(centrality);
    effVars[3] = eff->GetAxis(3)->FindBin(posZ);
    return eff->GetBinContent(effVars);
  }
};

struct CorrelationHashTask {
  std::vector<float> vtxBins;
  std::vector<float> multBins;

  Produces<aod::Hashes> hashes;

  void fillArray(int length, double* source, std::vector<float>& target)
  {
    // Expand binning from Configurable. Can we let some code in AxisSpec do this?

    target.clear();
    if (source[0] == VARIABLE_WIDTH) {
      for (int i = 1; i < length; i++) {
        target.push_back(source[i]);
      }
    } else {
      for (int i = 0; i <= source[0]; i++) {
        target.push_back(source[1] + (source[2] - source[1]) / source[0] * i);
      }
    }
  }

  void init(o2::framework::InitContext& initContext)
  {
    // get own suffix. Is there a better way?
    auto& deviceSpec = initContext.services().get<DeviceSpec const>();
    std::string suffix(deviceSpec.name);
    suffix.replace(0, strlen("correlation-hash-task"), "");

    // get axis config from CorrelationTask
    auto& workflows = initContext.services().get<RunningWorkflowInfo const>();
    for (DeviceSpec device : workflows.devices) {
      if (device.name == "correlation-task" + suffix) {
        for (auto option : device.options) {
          if (option.name == "axisVertex") {
            fillArray(option.defaultValue.size(), option.defaultValue.get<double*>(), vtxBins);
            LOGF(info, "Initialized vertex binning for mixing from configurable %s", option.name);
          }
          if (option.name == "axisMultiplicity") {
            fillArray(option.defaultValue.size(), option.defaultValue.get<double*>(), multBins);
            LOGF(info, "Initialized multiplicity binning for mixing from configurable %s", option.name);
          }
        }
      }
    }

    if (vtxBins.size() == 0) {
      LOGF(fatal, "vtxBins not configured. Check configuration.");
    }
    if (multBins.size() == 0) {
      LOGF(fatal, "multBins not configured. Check configuration.");
    }
  }

  // Calculate hash for an element based on 2 properties and their bins.
  int getHash(float vtx, float mult)
  {
    // underflow
    if (vtx < vtxBins[0]) {
      return -1;
    }
    if (mult < multBins[0]) {
      return -1;
    }

    for (int i = 1; i < vtxBins.size(); i++) {
      if (vtx < vtxBins[i]) {
        for (int j = 1; j < multBins.size(); j++) {
          if (mult < multBins[j]) {
            return i + j * (vtxBins.size() + 1);
          }
        }
      }
    }
    // overflow
    return -1;
  }

  void process(soa::Join<aod::Collisions, aod::Cents> const& collisions)
  {
    for (auto& collision : collisions) {
      int hash = getHash(collision.posZ(), collision.centV0M());
      LOGF(info, "Collision: %d (%f, %f) hash: %d", collision.index(), collision.posZ(), collision.centV0M(), hash);
      hashes(hash);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<CorrelationHashTask>(cfgc),
    adaptAnalysisTask<CorrelationTask>(cfgc, Processes{&CorrelationTask::processSame, &CorrelationTask::processMixed})};
}
