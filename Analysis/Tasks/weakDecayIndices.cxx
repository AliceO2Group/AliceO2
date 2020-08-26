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

using namespace o2;
using namespace o2::framework;

struct ATask {
  Produces<aod::TransientV0s> transientV0s;
  Produces<aod::TransientCascades> transientCascades;

  void process(aod::StoredV0s const& v0s, aod::StoredCascades const& cascades, aod::FullTracks const& tracks)
  {
    for (auto& v0 : v0s) {
      transientV0s(v0.posTrack().collisionId());
    }
    for (auto& cascade : cascades) {
      transientCascades(cascade.bachelor().collisionId());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("weak-decay-indices"),
  };
}
