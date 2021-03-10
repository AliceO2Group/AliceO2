// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

// O2 includes
#include "ReconstructionDataFormats/Track.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "DataModel/JEDerived.h"
#include "AnalysisDataModel/Jet.h"
#include "AnalysisCore/JetFinder.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#include "Framework/runDataProcessing.h"

struct JetProviderTask {

  Produces<aod::JEJets> outputJets;
  Produces<aod::JEConstituents> outputConstituents;

  Configurable<float> jetPtMin{"jetPtMin", 0.0, "minimum jet pT cut"};
  Configurable<bool> keepConstituents{"keepConstituents", true, "Constituent table is filled"};
  Configurable<bool> DoConstSub{"DoConstSub", false, "do constituent subtraction"};
  Filter jetCuts = aod::jet::pt > jetPtMin;

  void process(soa::Filtered<aod::Jets>::iterator const& jet,
               aod::Tracks const& tracks,
               aod::JetConstituents const& constituents,
               aod::JetConstituentsSub const& constituentsSub)
  {
    outputJets(jet.pt(), jet.eta(), jet.phi(), jet.energy(), jet.mass(), jet.area());
    if (keepConstituents) {
      if (DoConstSub) {
        outputConstituents.reserve(constituentsSub.size());
        for (const auto constituent : constituentsSub) {
          outputConstituents(outputJets.lastIndex(), constituent.pt(), constituent.eta(), constituent.phi());
        }
      } else {
        outputConstituents.reserve(constituents.size());
        for (const auto constituentIndex : constituents) {
          auto constituent = constituentIndex.track();
          outputConstituents(outputJets.lastIndex(), constituent.pt(), constituent.eta(), constituent.phi());
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<JetProviderTask>(cfgc, TaskName{"jet-task-skim-provider"})};
  return workflow;
}