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
// in run2 converted data through the bc column.
// For an example how to use this, please see: o2-analysistutorial-zdc-vzero-iteration

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct Run2Matcher {
  BuildsExclusive<aod::Run2MatchedExclusive> matched_e;
  Builds<aod::Run2MatchedSparse> matched_s;
  void init(o2::framework::InitContext&)
  {
  }
};

struct BCMatcher {
  BuildsExclusive<aod::BCCollisionsExclusive> matched_e;
  Builds<aod::BCCollisionsSparse> matched;
  void init(o2::framework::InitContext&)
  {
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<Run2Matcher>("produce-run2-bc-matching"),
    adaptAnalysisTask<BCMatcher>("bc-matcher"),
  };
}
