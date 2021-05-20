// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Joined tables can be used as argument to the process function.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;

struct UseJoins {
  void init(InitContext&)
  {
    count = 0;
  }

  void process(soa::Join<aod::Tracks, aod::TracksExtra> const& fullTracks)
  {
    for (auto& track : fullTracks) {
      LOGF(info, "%d, %f %f", count, track.alpha(), track.tpcSignal());
      count++;
    }
  }

  size_t count = 2016927;
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<UseJoins>(cfgc),
  };
}
