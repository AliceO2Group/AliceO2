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
#include "Framework/AnalysisDataModel.h"

#include <boost/test/unit_test.hpp>

using namespace o2;
using namespace o2::framework;

namespace o2::aod
{
namespace track
{
DECLARE_SOA_COLUMN(Foo, foo, float, "fBar");
DECLARE_SOA_COLUMN(Bar, bar, float, "fFoo");
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](float x, float y) { return x + y; });
} // namespace track
DECLARE_SOA_TABLE(FooBars, "AOD", "FOOBAR",
                  track::Foo, track::Bar,
                  track::Sum<track::Foo, track::Bar>);
} // namespace o2::aod

// FIXME: for the moment we do not derive from AnalysisTask as
// we need GCC 7.4+ to fix a bug.
struct ATask {
  Produces<aod::FooBars> foobars;

  void process(o2::aod::Track const& track)
  {
    foobars(0.01102005, 0.27092016); // dummy value for phi for now...
  }
};

// FIXME: for the moment we do not derive from AnalysisTask as
// we need GCC 7.4+ to fix a bug.
struct BTask {
  void process(o2::aod::Collision const&, o2::aod::Track const&)
  {
  }
};

// FIXME: for the moment we do not derive from AnalysisTask as
// we need GCC 7.4+ to fix a bug.
struct CTask {
  void process(o2::aod::Collision const&, o2::aod::Tracks const&)
  {
  }
};

// FIXME: for the moment we do not derive from AnalysisTask as
// we need GCC 7.4+ to fix a bug.
struct DTask {
  void process(o2::aod::Tracks const&)
  {
  }
};

// FIXME: for the moment we do not derive from AnalysisTask as
// we need GCC 7.4+ to fix a bug.
struct ETask {
  void process(o2::aod::FooBars::iterator const& foobar)
  {
    foobar.sum();
  }
};

// FIXME: for the moment we do not derive from AnalysisTask as
// we need GCC 7.4+ to fix a bug.
struct FTask {
  expressions::Filter fooFilter = aod::track::foo > 1.;
  void process(soa::Filtered<o2::aod::FooBars>::iterator const& foobar)
  {
    foobar.sum();
  }
};

BOOST_AUTO_TEST_CASE(AdaptorCompilation)
{
  auto task1 = adaptAnalysisTask<ATask>("test1");
  BOOST_CHECK_EQUAL(task1.inputs.size(), 1);
  BOOST_CHECK_EQUAL(task1.outputs.size(), 2);
  BOOST_CHECK_EQUAL(task1.inputs[0].binding, std::string("Tracks"));
  BOOST_CHECK_EQUAL(task1.outputs[0].binding.value, std::string("FooBars"));

  auto task2 = adaptAnalysisTask<BTask>("test2");
  BOOST_CHECK_EQUAL(task2.inputs.size(), 2);
  BOOST_CHECK_EQUAL(task2.inputs[0].binding, "Collisions");
  BOOST_CHECK_EQUAL(task2.inputs[1].binding, "Tracks");

  auto task3 = adaptAnalysisTask<CTask>("test3");
  BOOST_CHECK_EQUAL(task3.inputs.size(), 2);
  BOOST_CHECK_EQUAL(task3.inputs[0].binding, "Collisions");
  BOOST_CHECK_EQUAL(task3.inputs[1].binding, "Tracks");

  auto task4 = adaptAnalysisTask<DTask>("test4");
  BOOST_CHECK_EQUAL(task4.inputs.size(), 1);
  BOOST_CHECK_EQUAL(task4.inputs[0].binding, "Tracks");

  auto task5 = adaptAnalysisTask<ETask>("test5");
  BOOST_CHECK_EQUAL(task5.inputs.size(), 1);
  BOOST_CHECK_EQUAL(task5.inputs[0].binding, "FooBars");

  auto task6 = adaptAnalysisTask<FTask>("test6");
  BOOST_CHECK_EQUAL(task6.inputs.size(), 1);
  BOOST_CHECK_EQUAL(task6.inputs[0].binding, "FooBars");
}
