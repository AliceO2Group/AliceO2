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
///
/// \brief Demonstration of conditions in filter expressions

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct ConditionalExpressions {
  Configurable<bool> useFlags{"useFlags", false, "Switch to enable using track flags for selection"};
  Filter trackFilter = nabs(aod::track::eta) < 0.9f && aod::track::pt > 0.5f && ifnode(useFlags.node() == true, (aod::track::flags & static_cast<uint32_t>(o2::aod::track::ITSrefit)) != 0u, true);
  OutputObj<TH2F> etapt{TH2F("etapt", ";#eta;#p_{T}", 201, -2.1, 2.1, 601, 0, 60.1)};
  void process(aod::Collision const&, soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra>> const& tracks)
  {
    for (auto& track : tracks) {
      etapt->Fill(track.eta(), track.pt());
    }
  }
};

struct BasicOperations {
  Configurable<bool> useFlags{"useFlags", false, "Switch to enable using track flags for selection"};
  Filter trackFilter = nabs(aod::track::eta) < 0.9f && aod::track::pt > 0.5f;
  OutputObj<TH2F> etapt{TH2F("etapt", ";#eta;#p_{T}", 201, -2.1, 2.1, 601, 0, 60.1)};
  void process(aod::Collision const&, soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra>> const& tracks)
  {
    for (auto& track : tracks) {
      if (useFlags) {
        if ((track.flags() & o2::aod::track::ITSrefit) != 0u) {
          etapt->Fill(track.eta(), track.pt());
        }
      } else {
        etapt->Fill(track.eta(), track.pt());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ConditionalExpressions>(cfgc),
    adaptAnalysisTask<BasicOperations>(cfgc)};
}
