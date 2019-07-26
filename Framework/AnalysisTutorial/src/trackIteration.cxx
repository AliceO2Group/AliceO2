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

// Another example
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  void init(InitContext&)
  {
    count = 0;
  }

  void process(aod::Track const& track)
  {
    LOGF(info, "%d, %f", count, track.alpha());
    count++;
  }

  size_t count = 2016927;
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("track-iteration-tutorial")
  };
}
