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

#include "Analysis/Jet.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct JetFinderTask {
  Produces<o2::aod::Jets> jets;
  Produces<o2::aod::JetConstituents> constituents;

  // TODO: use abs, eventually use pt
  Filter trackCuts = (aod::track::signed1Pt < 10.f) &&
                     (aod::track::signed1Pt > -10.f);

  // TODO: use configurables for all parameters
  Configurable<double> rParam{"rParam", 0.4, "jet radius"};
  Configurable<float> ghostEtamax{"ghostEtamax", 1.0, "eta max for ghosts"};
  fastjet::Strategy strategy{fastjet::Best};
  fastjet::RecombinationScheme recombScheme{fastjet::E_scheme};
  fastjet::JetAlgorithm algorithm{fastjet::antikt_algorithm};
  fastjet::AreaType areaType{fastjet::passive_area};

  void process(aod::Collision const& collision,
               soa::Filtered<aod::Tracks> const& fullTracks)
  {
    // TODO: retrieve pion mass from somewhere
    const float mPionSquared = 0.139 * 0.139;

    std::vector<fastjet::PseudoJet> inputParticles;
    for (auto& track : fullTracks) {
      auto energy = std::sqrt(track.p2() + mPionSquared);
      inputParticles.emplace_back(track.px(), track.py(), track.pz(), energy);
      inputParticles.back().set_user_index(track.globalIndex());
    }

    fastjet::GhostedAreaSpec ghostSpec(ghostEtamax);
    fastjet::AreaDefinition areaDef(areaType, ghostSpec);
    fastjet::JetDefinition jetDef(algorithm, rParam, recombScheme, strategy);
    fastjet::ClusterSequenceArea clust_seq(inputParticles, jetDef, areaDef);

    std::vector<fastjet::PseudoJet> inclusiveJets = clust_seq.inclusive_jets();
    for (const auto& pjet : inclusiveJets) {
      jets(collision, pjet.eta(), pjet.phi(), pjet.pt(),
           pjet.area(), pjet.Et(), pjet.m());
      for (const auto& track : pjet.constituents()) {
        LOGF(INFO, "jet %d constituent %d: %f %f %f", jets.lastIndex(),
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
