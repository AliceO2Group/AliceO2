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

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/EventSelection.h"

#include "AnalysisDataModel/Jet.h"
#include "AnalysisCore/JetUtilities.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

template<class T, class U>
struct JetMatching {
  Configurable<float> maxMatchingDistance{"maxMatchingDistance", 0.4f, "Max matching distance"};

  void init(InitContext const&)
  {
  }

  void process(T & jetsBase, U & jetsTag)
  {
    std::vector<double> jetsBasePhi(jetsBase.size());
    std::vector<double> jetsBaseEta(jetsBase.size());
    for (auto jet : jetsBase) {
    {
      jetsBasePhi.emplace_back(jets.Phi());
      jetsBaseEta.emplace_back(jets.Eta());
    }
    std::vector<double> jetsTagPhi(jetsTag.size());
    std::vector<double> jetsTagEta(jetsTag.size());
    for (auto jet : jetsTag) {
    {
      jetsTagPhi.emplace_back(jets.Phi());
      jetsTagEta.emplace_back(jets.Eta());
    }
    auto && [baseToTagIndexMap, tagToBaseIndexMap] = JetUtilities::MatchJetsGeometrically(jetsBasePhi, jetsBaseEta, jetsTagPhi, jetsTagEta, maxMatchingDistance);

    for (std::size_t i; i < baseToTagIndexMap.size(); ++i) {
      jetsBase[i].MatchedJetIndex = baseToTagIndexMap[i];
    }
    for (std::size_t i; i < tagToBaseIndexMap.size(); ++i) {
      jetsTag[i].MatchedJetIndex = tagToBaseIndexMap[i];
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  // TODO: Is there a better way to do this?
  auto jetMatching = cfgc.options().get<str>("jet-matching");
  //if (jetMatching == "MC") {
    return WorkflowSpec{
      adaptAnalysisTask<JetMatching<o2::AOD::Jets, o2::AOD::JetsDetLevel>(cfgc, TaskName{"jet-matching-MC"})};
  //}
  /*return WorkflowSpec{
    adaptAnalysisTask<JetFinderTaskCharged>(cfgc, TaskName{"jet-finder-charged"})};*/
}

