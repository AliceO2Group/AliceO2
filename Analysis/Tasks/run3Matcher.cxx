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

// This task prepares the matching between collisions and other tables available
// in run 3 data.
// For an example how to use this (but based on run 2 data),
// please see: o2-analysistutorial-zdc-vzero-iteration

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct Run3Matcher {
  BuildsExclusive<aod::Run3MatchedExclusive> matched_e;
  Builds<aod::Run3MatchedSparse> matched_s;
  BuildsExclusive<aod::BCCollisionsExclusive> bc_e;
  Builds<aod::BCCollisionsSparse> bc;
  void init(o2::framework::InitContext&)
  {
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<Run3Matcher>("produce-run3-bc-matching"),
  };
}
