// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN

// O2 includes
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct ALICE3MultTask {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<float> MinEta{"MinEta", -0.8f, "Minimum eta in range"};
  Configurable<float> MaxEta{"MaxEta", 0.8f, "Maximum eta in range"};
  Configurable<float> MaxMult{"MaxMult", 1000.f, "Maximum multiplicity in range"};

  void init(InitContext&)
  {
    histos.add("multiplicity/numberOfTracks", ";Reconstructed tracks", kTH1D, {{(int)MaxMult, 0, MaxMult}});
  }

  void process(const o2::aod::Collision& collision, const o2::aod::Tracks& tracks)
  {
    if (collision.numContrib() < 2) {
      return;
    }

    int nTracks = 0;
    for (const auto& track : tracks) {
      if (track.eta() < MinEta || track.eta() > MaxEta) {
        continue;
      }
      nTracks++;
    }

    histos.fill(HIST("multiplicity/numberOfTracks"), nTracks);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<ALICE3MultTask>(cfgc)};
}
