// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework AnalysisTask
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/AnalysisTask.h"

#include <boost/test/unit_test.hpp>

using namespace o2::framework;

class ATask : public AnalysisTask
{
 public:
  void init(InitContext& ic) final
  {
  }
  void run(ProcessingContext& pc) final
  {
  }

  void processTrack(o2::aod::Track const& track)
  {
  }
};

class BTask : public AnalysisTask
{
 public:
  void init(InitContext& ic) final
  {
  }
  void run(ProcessingContext& pc) final
  {
  }

  void processCollisionTrack(o2::aod::Collision const&, o2::aod::Track const&)
  {
  }
};

BOOST_AUTO_TEST_CASE(AdaptorCompilation)
{
  auto task1 = adaptAnalysisTask<ATask>("test1");
  BOOST_CHECK_EQUAL(task1.inputs.size(), 1);
  auto task2 = adaptAnalysisTask<BTask>("test2");
  BOOST_CHECK_EQUAL(task2.inputs.size(), 2);
  BOOST_CHECK_EQUAL(task2.inputs[0].binding, "Collisions");
  BOOST_CHECK_EQUAL(task2.inputs[1].binding, "Tracks");
}
