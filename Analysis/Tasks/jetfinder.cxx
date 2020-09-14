// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// jet finder task
//
// Author: Jochen Klein

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/AreaDefinition.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"
#include "fastjet/tools/Subtractor.hh"

#include "Analysis/Jet.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct JetFinderTask {
  Produces<o2::aod::Jets> jets;
  Produces<o2::aod::JetConstituents> constituents;

  // options for jet finding
  Configurable<float> rParam{"rParam", 0.4, "jet radius"};
  Configurable<float> ghostEtamax{"ghostEtamax", 1.0, "eta max for ghosts"};
  Configurable<float> minJetPt{"minJetPt", 0., "minimum jet pt"};
  // TODO: initialize with 0.9 - rParam (requires lazy evaluation)
  Configurable<float> maxJetEta{"maxJetEta", 0.5, "eta range for jets"};
  // TODO: use configurables also for enums
  fastjet::JetAlgorithm algorithm{fastjet::antikt_algorithm};
  fastjet::RecombinationScheme recombScheme{fastjet::E_scheme};
  fastjet::Strategy strategy{fastjet::Best};
  fastjet::AreaType areaType{fastjet::passive_area};
  fastjet::GhostedAreaSpec ghostSpec{ghostEtamax};
  fastjet::AreaDefinition areaDef{areaType, ghostSpec};
  fastjet::JetDefinition jetDef{algorithm, rParam, recombScheme, strategy};
  fastjet::Selector selJet = fastjet::SelectorPtMin(minJetPt) &&
                             fastjet::SelectorAbsRapMax(maxJetEta);

  // options for background subtraction
  enum class BkgMode { none,
                       rhoArea };
  BkgMode bkgMode = BkgMode::none;
  Configurable<double> rParamBkg{"rParamBkg", 0.2, "jet radius for background"};
  Configurable<double> rapBkg{"rapBkg", .9, "rapidity range for background"};
  // TODO: use configurables also for enums
  fastjet::JetAlgorithm algorithmBkg{fastjet::kt_algorithm};
  fastjet::RecombinationScheme recombSchemeBkg{fastjet::E_scheme};
  fastjet::JetDefinition jetDefBkg{algorithmBkg, rParamBkg, recombSchemeBkg, strategy};
  fastjet::AreaDefinition areaDefBkg{areaType, ghostSpec};
  fastjet::Selector selBkg = fastjet::SelectorAbsRapMax(rapBkg);

  // TODO: use values from configurables
  // TODO: add eta cuts
  Filter trackCuts = aod::track::pt > 0.1f;

  std::unique_ptr<fastjet::BackgroundEstimatorBase> bge;
  std::unique_ptr<fastjet::Subtractor> sub;

  std::vector<fastjet::PseudoJet> pJets;

  void init(InitContext const&)
  {
    if (bkgMode == BkgMode::none) {
    } else if (bkgMode == BkgMode::rhoArea) {
      bge = decltype(bge)(new fastjet::JetMedianBackgroundEstimator(selBkg, jetDefBkg, areaDefBkg));
      sub = decltype(sub){new fastjet::Subtractor{bge.get()}};
    } else {
      LOGF(ERROR, "requested subtraction mode not implemented!");
    }
  }

  void process(aod::Collision const& collision,
               soa::Filtered<aod::Tracks> const& fullTracks)
  {
    // TODO: retrieve pion mass from somewhere
    const float mPionSquared = 0.139 * 0.139;

    pJets.clear();

    std::vector<fastjet::PseudoJet> inputParticles;
    for (auto& track : fullTracks) {
      auto energy = std::sqrt(track.p() * track.p() + mPionSquared);
      inputParticles.emplace_back(track.px(), track.py(), track.pz(), energy);
      inputParticles.back().set_user_index(track.globalIndex());
    }

    fastjet::ClusterSequenceArea clust_seq(inputParticles, jetDef, areaDef);
    if (bge)
      bge->set_particles(inputParticles);
    pJets = sub ? (*sub)(clust_seq.inclusive_jets()) : clust_seq.inclusive_jets();

    pJets = selJet(pJets);

    for (const auto& pjet : pJets) {
      jets(collision, pjet.eta(), pjet.phi(), pjet.pt(),
           pjet.area(), pjet.Et(), pjet.m());
      for (const auto& track : pjet.constituents()) {
        LOGF(DEBUG, "jet %d constituent %d: %f %f %f", jets.lastIndex(),
             track.user_index(), track.eta(), track.phi(), track.pt());
        constituents(jets.lastIndex(), track.user_index());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetFinderTask>("jet-finder")};
}
