// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file jetmatching.cxx
/// \brief Jet matching
///
/// \author Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"
#include "AnalysisDataModel/EventSelection.h"

#include "AnalysisDataModel/Jet.h"
#include "AnalysisCore/JetUtilities.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec jetMatching = {"jet-matching",
                               VariantType::String,
                               "",
                               {"Jet collections to match, separated by commas. Example: \"MCDetectorLevel-MCParticleLevel\". Possible components: MCParticleLevel, MCDetectorLevel, HybridIntermediate, Hybrid"}};
  workflowOptions.push_back(jetMatching);
}

#include "Framework/runDataProcessing.h"

//template<typename BaseJetCollection, typename TagJetCollection, typename Collision>
template<typename BaseJetCollection, typename BaseJetCollectionMatching, typename TagJetCollection, typename TagJetCollectionMatching>
struct JetMatching {
  Configurable<float> maxMatchingDistance{"maxMatchingDistance", 0.4f, "Max matching distance"};
  Produces<BaseJetCollectionMatching> jetsBaseMatching;
  Produces<TagJetCollectionMatching> jetsBaseMatching;

  void init(InitContext const&)
  {
  }

  void process(
    soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision,
    //Collision const& collision,
    BaseJetCollection const& jetsBase,
    TagJetCollection const& jetsTag)
  {
    std::vector<double> jetsBasePhi(jetsBase.size());
    std::vector<double> jetsBaseEta(jetsBase.size());
    for (auto jet : jetsBase) {
      jetsBasePhi.emplace_back(jet.phi());
      jetsBaseEta.emplace_back(jet.eta());
    }
    std::vector<double> jetsTagPhi(jetsTag.size());
    std::vector<double> jetsTagEta(jetsTag.size());
    for (auto & jet : jetsTag) {
      jetsTagPhi.emplace_back(jet.phi());
      jetsTagEta.emplace_back(jet.eta());
    }
    auto && [baseToTagIndexMap, tagToBaseIndexMap] = JetUtilities::MatchJetsGeometrically(jetsBasePhi, jetsBaseEta, jetsTagPhi, jetsTagEta, maxMatchingDistance);

    unsigned int i = 0;
    for (auto & jet : jetsBase) {
      // Store results
      //jetsBaseMatching(jet.lastIndex(), baseToTagIndexMap[i]);
      //jet.MatchedJetIndex(baseToTagIndexMap[i]);
      ++i;
    }
    /*for (std::size_t i; i < baseToTagIndexMap.size(); ++i) {
      jetsBase[i].MatchedJetIndex = baseToTagIndexMap[i];
    }*/
    i = 0;
    //for (std::size_t i; i < tagToBaseIndexMap.size(); ++i) {
    for (auto & jet : jetsTag) {
      // Store results...
      //jetsTag[i].MatchedJetIndex = tagToBaseIndexMap[i];
      //jet.MatchedJetIndex(tagToBaseIndexMap[i]);
      ++i;
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  // TODO: Is there a better way to do this?
  auto jetMatching = cfgc.options().get<std::string>("jet-matching");
  // Tokenize using stringstream
  std::vector<std::string> matchingOptions;
  std::stringstream ss;
  ss << jetMatching;
  while (ss.good()) {
    std::string substring;
    getline(ss, substring, ',');
    matchingOptions.push_back(substring);
  }
  std::vector<o2::framework::DataProcessorSpec> tasks;
  /*for (auto opt : matchingOptions) {
    // If there is a hybrid subtracted jet collection.
    if (opt == "Hybrid-HybridIntermediate") {
      //tasks.emplace_back(adaptAnalysisTask<
        //JetMatching<o2::aod::Jets, o2::aod::JetsHybridIntermediate, o2::aod::Collisions::iterator>(
        WorkflowSpec{
        JetMatching<o2::aod::Jets, o2::aod::JetsHybridIntermediate>(
          cfgc, TaskName{"jet-matching-hybrid-sub-to-hybrid-intermedaite"}
        )};
        //);
    }
    // If there are two classes of hybrid jets, this will match from the second class to the detector level
    if (opt == "HybridIntermediate-MCDetectorLevel") {
      tasks.emplace_back(adaptAnalysisTask<JetMatching<o2::aod::JetsHybridIntermediate, o2::aod::JetsMCDetectorLevel>(cfgc, TaskName{"jet-matching-hybrid-intermediate-to-MC-detector-level"}));
    }
    // If there is just a single standard hybrid jet collection, it can be matched directly to MC detector level.
    if (opt == "Hybrid-MCDetectorLevel") {
      tasks.emplace_back(adaptAnalysisTask<JetMatching<o2::aod::Jets, o2::aod::JetsMCDetectorLevel>(cfgc, TaskName{"jet-matching-hybrid-to-MC-detector-level"}));
    }
    // Finally, match MC detector level to MC particle level.
    if (opt == "MCParticleLevel-MCDetectorLevel") {
      tasks.emplace_back(adaptAnalysisTask<JetMatching<o2::aod::JetsMCDetectorLevel, o2::aod::JetsMCParticleLevel>(cfgc, TaskName{"jet-matching-MC"}));
    }

  }
  return WorkflowSpec{tasks};*/
  return WorkflowSpec{
      adaptAnalysisTask<JetMatching<o2::aod::Jets, o2::aod::MatchedJets, o2::aod::HybridIntermediateJets, o2::aod::MatchedHybridIntermediateJets>>(
        cfgc, TaskName{"jet-matching-hybrid-sub-to-hybrid-intermedaite"})};
  /*return WorkflowSpec{
    adaptAnalysisTask<JetFinderTaskCharged>(cfgc, TaskName{"jet-finder-charged"})};*/
}

