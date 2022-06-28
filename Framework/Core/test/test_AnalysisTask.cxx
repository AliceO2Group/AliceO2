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
#define BOOST_TEST_MODULE Test Framework AnalysisTask
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Mocking.h"
#include "TestClasses.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"

#include <boost/test/unit_test.hpp>

using namespace o2;
using namespace o2::framework;

namespace o2::aod
{
namespace test
{
DECLARE_SOA_COLUMN(X, x, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(Z, z, float);
DECLARE_SOA_COLUMN(Foo, foo, float);
DECLARE_SOA_COLUMN(Bar, bar, float);
DECLARE_SOA_COLUMN(EventProperty, eventProperty, float);
DECLARE_SOA_DYNAMIC_COLUMN(Sum, sum, [](float x, float y) { return x + y; });
} // namespace test
DECLARE_SOA_TABLE(Foos, "AOD", "FOO",
                  test::Foo);
DECLARE_SOA_TABLE(Bars, "AOD", "BAR",
                  test::Bar);
DECLARE_SOA_TABLE(FooBars, "AOD", "FOOBAR",
                  test::Foo, test::Bar,
                  test::Sum<test::Foo, test::Bar>);
DECLARE_SOA_TABLE(XYZ, "AOD", "XYZ",
                  test::X, test::Y, test::Z);
DECLARE_SOA_TABLE(Events, "AOD", "EVENTS",
                  test::EventProperty);
} // namespace o2::aod

struct ATask {
  Produces<aod::FooBars> foobars;

  void process(o2::aod::Track const&)
  {
    foobars(0.01102005, 0.27092016); // dummy value for phi for now...
  }
};

struct BTask {
  void process(o2::aod::Collision const&, o2::soa::Join<o2::aod::Tracks, o2::aod::TracksExtra, o2::aod::TracksCov> const&, o2::aod::AmbiguousTracks const&, o2::aod::Calos const&, o2::aod::CaloTriggers const&)
  {
  }
};

struct CTask {
  void process(o2::aod::Collision const&, o2::aod::Tracks const&)
  {
  }
};

struct DTask {
  void process(o2::aod::Tracks const&)
  {
  }
};

struct ETask {
  void process(o2::aod::FooBars::iterator const& foobar)
  {
    foobar.sum();
  }
};

struct FTask {
  expressions::Filter fooFilter = aod::test::foo > 1.;
  void process(soa::Filtered<o2::aod::FooBars>::iterator const& foobar)
  {
    foobar.sum();
  }
};

struct GTask {
  void process(o2::soa::Join<o2::aod::Foos, o2::aod::Bars, o2::aod::XYZ> const& foobars)
  {
    for (auto foobar : foobars) {
      foobar.x();
      foobar.foo();
      foobar.bar();
    }
  }
};

struct HTask {
  void process(o2::soa::Join<o2::aod::Foos, o2::aod::Bars, o2::aod::XYZ>::iterator const& foobar)
  {
    foobar.x();
    foobar.foo();
    foobar.bar();
  }
};

struct ITask {
  expressions::Filter flt = aod::test::bar > 0.;
  void process(o2::aod::Collision const&, o2::soa::Filtered<o2::soa::Join<o2::aod::Foos, o2::aod::Bars, o2::aod::XYZ>> const& foobars)
  {
    for (auto foobar : foobars) {
      foobar.x();
      foobar.foo();
      foobar.bar();
    }
  }
};

struct JTask {
  Configurable<o2::test::SimplePODClass> cfg{"someConfigurable", {}, "Some Configurable Object"};
  void process(o2::aod::Collision const&)
  {
    BOOST_CHECK_EQUAL(cfg->x, 1);
  }
};

struct TestCCDBObject {
  int SomeObject;
};

struct KTask {
  struct : public ConfigurableGroup {
    Configurable<int> anInt{"someConfigurable", {}, "Some Configurable Object"};
    Configurable<int> anotherInt{"someOtherConfigurable", {}, "Some Configurable Object"};
  } foo;
  Configurable<int> anThirdInt{"someThirdConfigurable", {}, "Some Configurable Object"};
  struct : public ConditionGroup {
    Condition<TestCCDBObject> test{"path"};
  } conditions;
  std::unique_ptr<int> someInt;
  std::shared_ptr<int> someSharedInt;
};

BOOST_AUTO_TEST_CASE(AdaptorCompilation)
{
  auto cfgc = makeEmptyConfigContext();

  auto task1 = adaptAnalysisTask<ATask>(*cfgc, TaskName{"test1"});
  BOOST_CHECK_EQUAL(task1.inputs.size(), 2);
  BOOST_CHECK_EQUAL(task1.outputs.size(), 1);
  BOOST_CHECK_EQUAL(task1.inputs[1].binding, std::string("Tracks"));
  BOOST_CHECK_EQUAL(task1.inputs[0].binding, std::string("TracksExtension"));
  BOOST_CHECK_EQUAL(task1.outputs[0].binding.value, std::string("FooBars"));

  auto task2 = adaptAnalysisTask<BTask>(*cfgc, TaskName{"test2"});
  BOOST_CHECK_EQUAL(task2.inputs.size(), 10);
  BOOST_CHECK_EQUAL(task2.inputs[1].binding, "TracksExtension");
  BOOST_CHECK_EQUAL(task2.inputs[2].binding, "Tracks");
  BOOST_CHECK_EQUAL(task2.inputs[3].binding, "TracksExtraExtension");
  BOOST_CHECK_EQUAL(task2.inputs[4].binding, "TracksExtra");
  BOOST_CHECK_EQUAL(task2.inputs[5].binding, "TracksCovExtension");
  BOOST_CHECK_EQUAL(task2.inputs[6].binding, "TracksCov");
  BOOST_CHECK_EQUAL(task2.inputs[7].binding, "AmbiguousTracks");
  BOOST_CHECK_EQUAL(task2.inputs[8].binding, "Calos");
  BOOST_CHECK_EQUAL(task2.inputs[9].binding, "CaloTriggers");
  BOOST_CHECK_EQUAL(task2.inputs[0].binding, "Collisions");

  auto task3 = adaptAnalysisTask<CTask>(*cfgc, TaskName{"test3"});
  BOOST_CHECK_EQUAL(task3.inputs.size(), 3);
  BOOST_CHECK_EQUAL(task3.inputs[0].binding, "Collisions");
  BOOST_CHECK_EQUAL(task3.inputs[2].binding, "Tracks");
  BOOST_CHECK_EQUAL(task3.inputs[1].binding, "TracksExtension");

  auto task4 = adaptAnalysisTask<DTask>(*cfgc, TaskName{"test4"});
  BOOST_CHECK_EQUAL(task4.inputs.size(), 2);
  BOOST_CHECK_EQUAL(task4.inputs[1].binding, "Tracks");
  BOOST_CHECK_EQUAL(task4.inputs[0].binding, "TracksExtension");

  auto task5 = adaptAnalysisTask<ETask>(*cfgc, TaskName{"test5"});
  BOOST_CHECK_EQUAL(task5.inputs.size(), 1);
  BOOST_CHECK_EQUAL(task5.inputs[0].binding, "FooBars");

  auto task6 = adaptAnalysisTask<FTask>(*cfgc, TaskName{"test6"});
  BOOST_CHECK_EQUAL(task6.inputs.size(), 1);
  BOOST_CHECK_EQUAL(task6.inputs[0].binding, "FooBars");

  auto task7 = adaptAnalysisTask<GTask>(*cfgc, TaskName{"test7"});
  BOOST_CHECK_EQUAL(task7.inputs.size(), 3);

  auto task8 = adaptAnalysisTask<HTask>(*cfgc, TaskName{"test8"});
  BOOST_CHECK_EQUAL(task8.inputs.size(), 3);

  auto task9 = adaptAnalysisTask<ITask>(*cfgc, TaskName{"test9"});
  BOOST_CHECK_EQUAL(task9.inputs.size(), 4);

  auto task10 = adaptAnalysisTask<JTask>(*cfgc, TaskName{"test10"});
  BOOST_CHECK_EQUAL(task10.inputs.size(), 1);

  auto task11 = adaptAnalysisTask<KTask>(*cfgc, TaskName{"test11"});
  BOOST_CHECK_EQUAL(task11.options.size(), 3);
  BOOST_CHECK_EQUAL(task11.inputs.size(), 1);
}

BOOST_AUTO_TEST_CASE(TestPartitionIteration)
{
  TableBuilder builderA;
  auto rowWriterA = builderA.persist<float, float>({"fX", "fY"});
  rowWriterA(0, 0.0f, 8.0f);
  rowWriterA(0, 1.0f, 9.0f);
  rowWriterA(0, 2.0f, 10.0f);
  rowWriterA(0, 3.0f, 11.0f);
  rowWriterA(0, 4.0f, 12.0f);
  rowWriterA(0, 5.0f, 13.0f);
  rowWriterA(0, 6.0f, 14.0f);
  rowWriterA(0, 7.0f, 15.0f);
  auto tableA = builderA.finalize();
  BOOST_REQUIRE_EQUAL(tableA->num_rows(), 8);

  using TestA = o2::soa::Table<o2::soa::Index<>, aod::test::X, aod::test::Y>;
  using FilteredTest = o2::soa::Filtered<TestA>;
  using PartitionTest = Partition<TestA>;
  using PartitionFilteredTest = Partition<o2::soa::Filtered<TestA>>;
  using PartitionNestedFilteredTest = Partition<o2::soa::Filtered<o2::soa::Filtered<TestA>>>;
  using namespace o2::framework;

  TestA testA{tableA};

  PartitionTest p1 = aod::test::x < 4.0f;
  p1.setTable(testA);
  BOOST_CHECK_EQUAL(4, p1.size());
  BOOST_CHECK(p1.begin() != p1.end());
  auto i = 0;
  for (auto& p : p1) {
    BOOST_CHECK_EQUAL(i, p.x());
    BOOST_CHECK_EQUAL(i + 8, p.y());
    BOOST_CHECK_EQUAL(i, p.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 4);

  expressions::Filter f1 = aod::test::x < 4.0f;
  FilteredTest filtered{{testA.asArrowTable()}, o2::soa::selectionToVector(expressions::createSelection(testA.asArrowTable(), f1))};
  PartitionFilteredTest p2 = aod::test::y > 9.0f;
  p2.setTable(filtered);

  BOOST_CHECK_EQUAL(2, p2.size());
  i = 0;
  for (auto& p : p2) {
    BOOST_CHECK_EQUAL(i + 2, p.x());
    BOOST_CHECK_EQUAL(i + 10, p.y());
    BOOST_CHECK_EQUAL(i + 2, p.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 2);

  PartitionNestedFilteredTest p3 = aod::test::x < 3.0f;
  p3.setTable(*(p2.mFiltered));
  BOOST_CHECK_EQUAL(1, p3.size());
  i = 0;
  for (auto& p : p3) {
    BOOST_CHECK_EQUAL(i + 2, p.x());
    BOOST_CHECK_EQUAL(i + 10, p.y());
    BOOST_CHECK_EQUAL(i + 2, p.index());
    i++;
  }
  BOOST_CHECK_EQUAL(i, 1);
}
