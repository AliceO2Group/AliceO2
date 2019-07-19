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

#include <TFile.h>
#include <TH1F.h>

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to iterate over tracks
// and operate on them.
struct ATask : AnalysisTask {
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    // FIXME: to see some output, we create the histogram
    // for every timeframe. In general this is not the way it
    // should be done.
    LOGF(info, "Tracks for collision: %d", tracks.size());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("collision-tracks-iteration-tutorial")
  };
}
