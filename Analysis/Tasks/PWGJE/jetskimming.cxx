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

// jet skimming task
//
// Author: Nima Zardoshti

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{
namespace jetskim
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Energy, energy, float);
} // namespace jetskim
DECLARE_SOA_TABLE(JetSkim, "AOD", "JETSKIM1",
                  jetskim::CollisionId,
                  jetskim::Pt, jetskim::Eta, jetskim::Phi, jetskim::Energy);
} // namespace o2::aod

struct JetSkimmingTask1 {
  Produces<o2::aod::JetSkim> skim;

  Filter trackCuts = aod::track::pt > 0.15f;
  float mPionSquared = 0.139 * 0.139;

  void process(aod::Collision const& collision,
               soa::Filtered<aod::Tracks> const& tracks)
  {
    for (auto& track : tracks) {
      float energy = std::sqrt(track.p() * track.p() + mPionSquared);
      skim(collision, track.pt(), track.eta(), track.phi(), energy);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetSkimmingTask1>(cfgc, TaskName{"jet-skimmer"})};
}
