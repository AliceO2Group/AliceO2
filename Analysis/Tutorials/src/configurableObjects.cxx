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
#include "Analysis/configurableCut.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

/// This task demonstrates how to use configurable to wrap classes
/// use it with supplied configuration: "configurableObject.json"

struct ConfigurableObjectDemo {
  Configurable<configurableCut> cut{"cut", {0.5, 1, true}, "generic cut"};
  MutableConfigurable<configurableCut> mutable_cut{"mutable_cut", {1., 2, false}, "generic cut"};
  void init(InitContext const&){};
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Cut1: %.3f; Cut2: %.3f", cut, mutable_cut);
    for (auto& track : tracks) {
      if (track.globalIndex() % 500 == 0) {
        std::string decision1;
        std::string decision2;
        if (cut->method(std::abs(track.eta()))) {
          decision1 = "true";
        } else {
          decision1 = "false";
        }
        if (mutable_cut->method(std::abs(track.eta()))) {
          decision2 = "true";
        } else {
          decision2 = "false";
        }
        LOGF(INFO, "Cut1: %s; Cut2: %s", decision1, decision2);
        if (decision2 == "false") {
          mutable_cut->setState(-1);
        } else {
          mutable_cut->setState(1);
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ConfigurableObjectDemo>("configurable-demo")};
}
