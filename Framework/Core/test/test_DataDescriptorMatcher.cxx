// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework DataDescriptorMatcher
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/InputSpec.h"

#include <boost/test/unit_test.hpp>
#include <variant>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::framework::data_matcher;

BOOST_AUTO_TEST_CASE(TestMatcherInvariants)
{
  DataHeader header0;
  header0.dataOrigin = "TPC";
  header0.dataDescription = "CLUSTERS";
  header0.subSpecification = 1;
  VariableContext context;

  DataHeader header1;
  header1.dataOrigin = "ITS";
  header1.dataDescription = "TRACKLET";
  header1.subSpecification = 2;

  DataHeader header2;
  header2.dataOrigin = "TPC";
  header2.dataDescription = "TRACKLET";
  header2.subSpecification = 1;

  DataHeader header3;
  header3.dataOrigin = "TPC";
  header3.dataDescription = "CLUSTERS";
  header3.subSpecification = 0;

  DataHeader header4;
  header4.dataOrigin = "TRD";
  header4.dataDescription = "TRACKLET";
  header4.subSpecification = 0;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ "TPC" },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "CLUSTERS" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 1 },
        ConstantValueMatcher{ true }))
  };
  DataDescriptorMatcher matcher2 = matcher;
  BOOST_CHECK(matcher.match(header0, context) == true);
  BOOST_CHECK(matcher.match(header1, context) == false);
  BOOST_CHECK(matcher.match(header2, context) == false);
  BOOST_CHECK(matcher.match(header3, context) == false);
  BOOST_CHECK(matcher.match(header4, context) == false);
  BOOST_CHECK(matcher2.match(header0, context) == true);
  BOOST_CHECK(matcher2.match(header1, context) == false);
  BOOST_CHECK(matcher2.match(header2, context) == false);
  BOOST_CHECK(matcher2.match(header3, context) == false);
  BOOST_CHECK(matcher2.match(header4, context) == false);

  BOOST_CHECK(matcher2 == matcher);

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      OriginValueMatcher{ "TPC" }
    };
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      OriginValueMatcher{ "TPC" }
    };
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      DescriptionValueMatcher{ "TRACKS" }
    };
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      DescriptionValueMatcher{ "TRACKS" }
    };
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      SubSpecificationTypeValueMatcher{ 1 }
    };
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      SubSpecificationTypeValueMatcher{ 1 }
    };
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      ConstantValueMatcher{ 1 }
    };
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      ConstantValueMatcher{ 1 }
    };
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::And,
      ConstantValueMatcher{ 1 },
      DescriptionValueMatcher{ "TPC" }
    };
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      ConstantValueMatcher{ 1 },
      DescriptionValueMatcher{ "TPC" }
    };
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::And,
      OriginValueMatcher{ "TPC" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        DescriptionValueMatcher{ "CLUSTERS" },
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::And,
          SubSpecificationTypeValueMatcher{ 1 },
          ConstantValueMatcher{ true }))
    };

    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::And,
      OriginValueMatcher{ "TPC" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        DescriptionValueMatcher{ "CLUSTERS" },
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::And,
          SubSpecificationTypeValueMatcher{ 1 },
          ConstantValueMatcher{ true }))
    };
    BOOST_CHECK(matcherA == matcherB);
  }
}

BOOST_AUTO_TEST_CASE(TestSimpleMatching)
{
  DataHeader header0;
  header0.dataOrigin = "TPC";
  header0.dataDescription = "CLUSTERS";
  header0.subSpecification = 1;

  DataHeader header1;
  header1.dataOrigin = "ITS";
  header1.dataDescription = "TRACKLET";
  header1.subSpecification = 2;

  DataHeader header2;
  header2.dataOrigin = "TPC";
  header2.dataDescription = "TRACKLET";
  header2.subSpecification = 1;

  DataHeader header3;
  header3.dataOrigin = "TPC";
  header3.dataDescription = "CLUSTERS";
  header3.subSpecification = 0;

  DataHeader header4;
  header4.dataOrigin = "TRD";
  header4.dataDescription = "TRACKLET";
  header4.subSpecification = 0;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ "TPC" },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "CLUSTERS" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 1 },
        ConstantValueMatcher{ true }))
  };

  VariableContext context;
  BOOST_CHECK(matcher.match(header0, context) == true);
  BOOST_CHECK(matcher.match(header1, context) == false);
  BOOST_CHECK(matcher.match(header2, context) == false);
  BOOST_CHECK(matcher.match(header3, context) == false);
  BOOST_CHECK(matcher.match(header4, context) == false);

  DataDescriptorMatcher matcher1{
    DataDescriptorMatcher::Op::Or,
    OriginValueMatcher{ "TPC" },
    OriginValueMatcher{ "ITS" }
  };

  BOOST_CHECK(matcher1.match(header0, context) == true);
  BOOST_CHECK(matcher1.match(header1, context) == true);
  BOOST_CHECK(matcher1.match(header2, context) == true);
  BOOST_CHECK(matcher1.match(header3, context) == true);
  BOOST_CHECK(matcher1.match(header4, context) == false);

  DataDescriptorMatcher matcher2{
    DataDescriptorMatcher::Op::Just,
    DescriptionValueMatcher{ "TRACKLET" }
  };

  BOOST_CHECK(matcher2.match(header0, context) == false);
  BOOST_CHECK(matcher2.match(header1, context) == true);
  BOOST_CHECK(matcher2.match(header2, context) == true);
  BOOST_CHECK(matcher2.match(header3, context) == false);
  BOOST_CHECK(matcher2.match(header4, context) == true);
}

BOOST_AUTO_TEST_CASE(TestQueryBuilder)
{
  DataHeader header0;
  header0.dataOrigin = "TPC";
  header0.dataDescription = "CLUSTERS";
  header0.subSpecification = 1;

  DataHeader header1;
  header1.dataOrigin = "ITS";
  header1.dataDescription = "TRACKLET";
  header1.subSpecification = 2;

  DataHeader header2;
  header2.dataOrigin = "TPC";
  header2.dataDescription = "TRACKLET";
  header2.subSpecification = 1;

  DataHeader header3;
  header3.dataOrigin = "TPC";
  header3.dataDescription = "CLUSTERS";
  header3.subSpecification = 0;

  DataHeader header4;
  header4.dataOrigin = "TRD";
  header4.dataDescription = "TRACKLET";
  header4.subSpecification = 0;

  /// In this test the context is empty, since we do not use any variables.
  VariableContext context;

  auto matcher1 = DataDescriptorQueryBuilder::buildFromKeepConfig("TPC/CLUSTERS/1");
  BOOST_CHECK(matcher1.matcher->match(header0, context) == true);
  BOOST_CHECK(matcher1.matcher->match(header1, context) == false);
  BOOST_CHECK(matcher1.matcher->match(header2, context) == false);
  BOOST_CHECK(matcher1.matcher->match(header3, context) == false);
  BOOST_CHECK(matcher1.matcher->match(header4, context) == false);

  auto matcher2 = DataDescriptorQueryBuilder::buildFromKeepConfig("ITS/TRACKLET/2");
  BOOST_CHECK(matcher2.matcher->match(header0, context) == false);
  BOOST_CHECK(matcher2.matcher->match(header1, context) == true);
  BOOST_CHECK(matcher2.matcher->match(header2, context) == false);
  BOOST_CHECK(matcher2.matcher->match(header3, context) == false);
  BOOST_CHECK(matcher2.matcher->match(header4, context) == false);

  auto matcher3 = DataDescriptorQueryBuilder::buildFromKeepConfig("TPC/CLUSTERS/1,ITS/TRACKLET/2");
  BOOST_CHECK(matcher3.matcher->match(header0, context) == true);
  BOOST_CHECK(matcher3.matcher->match(header1, context) == true);
  BOOST_CHECK(matcher3.matcher->match(header2, context) == false);
  BOOST_CHECK(matcher3.matcher->match(header3, context) == false);
  BOOST_CHECK(matcher3.matcher->match(header4, context) == false);

  auto matcher4 = DataDescriptorQueryBuilder::buildFromKeepConfig("");
  BOOST_CHECK(matcher4.matcher->match(header0, context) == false);
  BOOST_CHECK(matcher4.matcher->match(header1, context) == false);
  BOOST_CHECK(matcher4.matcher->match(header2, context) == false);
  BOOST_CHECK(matcher4.matcher->match(header3, context) == false);
  BOOST_CHECK(matcher4.matcher->match(header4, context) == false);
}

// This checks matching using variables
BOOST_AUTO_TEST_CASE(TestMatchingVariables)
{
  VariableContext context;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ ContextRef{ 0 } },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "CLUSTERS" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ ContextRef{ 1 } },
        ConstantValueMatcher{ true }))
  };

  DataHeader header0;
  header0.dataOrigin = "TPC";
  header0.dataDescription = "CLUSTERS";
  header0.subSpecification = 1;

  BOOST_CHECK(matcher.match(header0, context) == true);
  auto s = std::get_if<std::string>(&context.get(0));
  BOOST_CHECK(s != nullptr);
  BOOST_CHECK(*s == "TPC");
  auto v = std::get_if<uint64_t>(&context.get(1));
  BOOST_CHECK(v != nullptr);
  BOOST_CHECK(*v == 1);

  // This will not match, because ContextRef{0} is bound
  // to TPC already.
  DataHeader header1;
  header1.dataOrigin = "ITS";
  header1.dataDescription = "CLUSTERS";
  header1.subSpecification = 1;

  BOOST_CHECK(matcher.match(header1, context) == false);
  auto s1 = std::get_if<std::string>(&context.get(0));
  BOOST_CHECK(s1 != nullptr);
  BOOST_CHECK(*s1 == "TPC");
}

BOOST_AUTO_TEST_CASE(TestInputSpecMatching)
{
  ConcreteDataMatcher spec0{ "TPC", "CLUSTERS", 1 };
  ConcreteDataMatcher spec1{ "ITS", "TRACKLET", 2 };
  ConcreteDataMatcher spec2{ "ITS", "TRACKLET", 1 };
  ConcreteDataMatcher spec3{ "TPC", "CLUSTERS", 0 };
  ConcreteDataMatcher spec4{ "TRD", "TRACKLET", 0 };

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ "TPC" },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "CLUSTERS" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 1 },
        ConstantValueMatcher{ true }))
  };

  VariableContext context;

  BOOST_CHECK(matcher.match(spec0, context) == true);
  BOOST_CHECK(matcher.match(spec1, context) == false);
  BOOST_CHECK(matcher.match(spec2, context) == false);
  BOOST_CHECK(matcher.match(spec3, context) == false);
  BOOST_CHECK(matcher.match(spec4, context) == false);

  DataDescriptorMatcher matcher1{
    DataDescriptorMatcher::Op::Or,
    OriginValueMatcher{ "TPC" },
    OriginValueMatcher{ "ITS" }
  };

  BOOST_CHECK(matcher1.match(spec0, context) == true);
  BOOST_CHECK(matcher1.match(spec1, context) == true);
  BOOST_CHECK(matcher1.match(spec2, context) == true);
  BOOST_CHECK(matcher1.match(spec3, context) == true);
  BOOST_CHECK(matcher1.match(spec4, context) == false);

  DataDescriptorMatcher matcher2{
    DataDescriptorMatcher::Op::Just,
    DescriptionValueMatcher{ "TRACKLET" }
  };

  BOOST_CHECK(matcher2.match(spec0, context) == false);
  BOOST_CHECK(matcher2.match(spec1, context) == true);
  BOOST_CHECK(matcher2.match(spec2, context) == true);
  BOOST_CHECK(matcher2.match(spec3, context) == false);
  BOOST_CHECK(matcher2.match(spec4, context) == true);
}

BOOST_AUTO_TEST_CASE(TestStartTimeMatching)
{
  VariableContext context;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::Just,
    StartTimeValueMatcher{ ContextRef{ 0 } }
  };

  DataHeader dh;
  dh.dataOrigin = "TPC";
  dh.dataDescription = "CLUSTERS";
  dh.subSpecification = 1;

  DataProcessingHeader dph;
  dph.startTime = 123;

  Stack s{ dh, dph };
  auto s2dph = o2::header::get<DataProcessingHeader*>(s.data());
  BOOST_CHECK(s2dph != nullptr);
  BOOST_CHECK_EQUAL(s2dph->startTime, 123);
  BOOST_CHECK(matcher.match(s, context) == true);
  auto vPtr = std::get_if<uint64_t>(&context.get(0));
  BOOST_REQUIRE(vPtr != nullptr);
  BOOST_CHECK_EQUAL(*vPtr, 123);
}

/// If a query matches only partially, we do not want
/// to pollute the context with partial results.
BOOST_AUTO_TEST_CASE(TestAtomicUpdatesOfContext)
{
  VariableContext context;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ ContextRef{ 0 } },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "CLUSTERS" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::Just,
        SubSpecificationTypeValueMatcher{ ContextRef{ 1 } }))
  };

  /// This will match TPC, but not TRACKS, so the context should
  /// be left pristine.
  DataHeader dh;
  dh.dataOrigin = "TPC";
  dh.dataDescription = "TRACKS";
  dh.subSpecification = 1;

  auto vPtr0 = std::get_if<None>(&context.get(0));
  auto vPtr1 = std::get_if<None>(&context.get(1));
  BOOST_CHECK(vPtr0 != nullptr);
  BOOST_CHECK(vPtr1 != nullptr);
  BOOST_REQUIRE_EQUAL(matcher.match(dh, context), false);
  // We discard the updates, because there was no match
  context.discard();
  vPtr0 = std::get_if<None>(&context.get(0));
  vPtr1 = std::get_if<None>(&context.get(1));
  BOOST_CHECK(vPtr0 != nullptr);
  BOOST_CHECK(vPtr1 != nullptr);
}

BOOST_AUTO_TEST_CASE(TestVariableContext)
{
  VariableContext context;
  // Put some updates, but do not commit them
  // we should still be able to retrieve them
  // (just slower).
  context.put(ContextUpdate{ 0, "A TEST" });
  context.put(ContextUpdate{ 10, 77 });
  auto v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "A TEST");
  auto v2 = std::get_if<std::string>(&context.get(1));
  BOOST_CHECK(v2 == nullptr);
  auto v3 = std::get_if<uint64_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 77);
  context.commit();
  // After commits everything is the same
  v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "A TEST");
  v2 = std::get_if<std::string>(&context.get(1));
  BOOST_CHECK(v2 == nullptr);
  v3 = std::get_if<uint64_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 77);

  // Let's update again. New values should win.
  context.put(ContextUpdate{ 0, "SOME MORE" });
  context.put(ContextUpdate{ 10, 16 });
  v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "SOME MORE");
  v2 = std::get_if<std::string>(&context.get(1));
  BOOST_CHECK(v2 == nullptr);
  v3 = std::get_if<uint64_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 16);

  // Until we discard
  context.discard();
  v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "A TEST");
  auto n = std::get_if<None>(&context.get(1));
  BOOST_CHECK(n != nullptr);
  v3 = std::get_if<uint64_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 77);

  // Let's update again. New values should win.
  context.put(ContextUpdate{ 0, "SOME MORE" });
  context.put(ContextUpdate{ 10, 16 });
  v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "SOME MORE");
  v2 = std::get_if<std::string>(&context.get(1));
  BOOST_CHECK(v2 == nullptr);
  v3 = std::get_if<uint64_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 16);

  // Until we discard again, using reset
  context.reset();
  auto n1 = std::get_if<None>(&context.get(0));
  BOOST_REQUIRE(n1 != nullptr);
  auto n2 = std::get_if<None>(&context.get(1));
  BOOST_CHECK(n2 != nullptr);
  auto n3 = std::get_if<None>(&context.get(10));
  BOOST_CHECK(n3 != nullptr);

  //auto d3 = std::get_if<uint64_t>(&context.get(0));
  //BOOST_CHECK(d1 == nullptr);
  //BOOST_CHECK(d2 == nullptr);
  //BOOST_CHECK(d3 == nullptr);
}
