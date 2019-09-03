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
    OriginValueMatcher{"TPC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        ConstantValueMatcher{true}))};
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
      OriginValueMatcher{"TPC"}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      OriginValueMatcher{"TPC"}};
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      DescriptionValueMatcher{"TRACKS"}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      DescriptionValueMatcher{"TRACKS"}};
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      SubSpecificationTypeValueMatcher{1}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      SubSpecificationTypeValueMatcher{1}};
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      ConstantValueMatcher{1}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      ConstantValueMatcher{1}};
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::And,
      ConstantValueMatcher{1},
      DescriptionValueMatcher{"TPC"}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      ConstantValueMatcher{1},
      DescriptionValueMatcher{"TPC"}};
    BOOST_CHECK(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::And,
      OriginValueMatcher{"TPC"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        DescriptionValueMatcher{"CLUSTERS"},
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::And,
          SubSpecificationTypeValueMatcher{1},
          ConstantValueMatcher{true}))};

    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::And,
      OriginValueMatcher{"TPC"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        DescriptionValueMatcher{"CLUSTERS"},
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::And,
          SubSpecificationTypeValueMatcher{1},
          ConstantValueMatcher{true}))};
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
    OriginValueMatcher{"TPC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        ConstantValueMatcher{true}))};

  VariableContext context;
  BOOST_CHECK(matcher.match(header0, context) == true);
  BOOST_CHECK(matcher.match(header1, context) == false);
  BOOST_CHECK(matcher.match(header2, context) == false);
  BOOST_CHECK(matcher.match(header3, context) == false);
  BOOST_CHECK(matcher.match(header4, context) == false);

  DataDescriptorMatcher matcher1{
    DataDescriptorMatcher::Op::Or,
    OriginValueMatcher{"TPC"},
    OriginValueMatcher{"ITS"}};

  BOOST_CHECK(matcher1.match(header0, context) == true);
  BOOST_CHECK(matcher1.match(header1, context) == true);
  BOOST_CHECK(matcher1.match(header2, context) == true);
  BOOST_CHECK(matcher1.match(header3, context) == true);
  BOOST_CHECK(matcher1.match(header4, context) == false);

  DataDescriptorMatcher matcher2{
    DataDescriptorMatcher::Op::Just,
    DescriptionValueMatcher{"TRACKLET"}};

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
    OriginValueMatcher{ContextRef{0}},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ContextRef{1}},
        ConstantValueMatcher{true}))};

  DataHeader header0;
  header0.dataOrigin = "TPC";
  header0.dataDescription = "CLUSTERS";
  header0.subSpecification = 1;

  BOOST_CHECK(matcher.match(header0, context) == true);
  auto s = std::get_if<std::string>(&context.get(0));
  BOOST_CHECK(s != nullptr);
  BOOST_CHECK(*s == "TPC");
  auto v = std::get_if<o2::header::DataHeader::SubSpecificationType>(&context.get(1));
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
  ConcreteDataMatcher spec0{"TPC", "CLUSTERS", 1};
  ConcreteDataMatcher spec1{"ITS", "TRACKLET", 2};
  ConcreteDataMatcher spec2{"ITS", "TRACKLET", 1};
  ConcreteDataMatcher spec3{"TPC", "CLUSTERS", 0};
  ConcreteDataMatcher spec4{"TRD", "TRACKLET", 0};

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TPC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        ConstantValueMatcher{true}))};

  VariableContext context;

  BOOST_CHECK(matcher.match(spec0, context) == true);
  BOOST_CHECK(matcher.match(spec1, context) == false);
  BOOST_CHECK(matcher.match(spec2, context) == false);
  BOOST_CHECK(matcher.match(spec3, context) == false);
  BOOST_CHECK(matcher.match(spec4, context) == false);

  DataDescriptorMatcher matcher1{
    DataDescriptorMatcher::Op::Or,
    OriginValueMatcher{"TPC"},
    OriginValueMatcher{"ITS"}};

  BOOST_CHECK(matcher1.match(spec0, context) == true);
  BOOST_CHECK(matcher1.match(spec1, context) == true);
  BOOST_CHECK(matcher1.match(spec2, context) == true);
  BOOST_CHECK(matcher1.match(spec3, context) == true);
  BOOST_CHECK(matcher1.match(spec4, context) == false);

  DataDescriptorMatcher matcher2{
    DataDescriptorMatcher::Op::Just,
    DescriptionValueMatcher{"TRACKLET"}};

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
    StartTimeValueMatcher{ContextRef{0}}};

  DataHeader dh;
  dh.dataOrigin = "TPC";
  dh.dataDescription = "CLUSTERS";
  dh.subSpecification = 1;

  DataProcessingHeader dph;
  dph.startTime = 123;

  Stack s{dh, dph};
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
    OriginValueMatcher{ContextRef{0}},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::Just,
        SubSpecificationTypeValueMatcher{ContextRef{1}}))};

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
  context.put(ContextUpdate{0, "A TEST"});
  context.put(ContextUpdate{10, uint32_t{77}});
  auto v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "A TEST");
  auto v2 = std::get_if<std::string>(&context.get(1));
  BOOST_CHECK(v2 == nullptr);
  auto v3 = std::get_if<uint32_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 77);
  context.commit();
  // After commits everything is the same
  v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "A TEST");
  v2 = std::get_if<std::string>(&context.get(1));
  BOOST_CHECK(v2 == nullptr);
  v3 = std::get_if<uint32_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 77);

  // Let's update again. New values should win.
  context.put(ContextUpdate{0, "SOME MORE"});
  context.put(ContextUpdate{10, uint32_t{16}});
  v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "SOME MORE");
  v2 = std::get_if<std::string>(&context.get(1));
  BOOST_CHECK(v2 == nullptr);
  v3 = std::get_if<uint32_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 16);

  // Until we discard
  context.discard();
  v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "A TEST");
  auto n = std::get_if<None>(&context.get(1));
  BOOST_CHECK(n != nullptr);
  v3 = std::get_if<uint32_t>(&context.get(10));
  BOOST_CHECK(v3 != nullptr);
  BOOST_CHECK(*v3 == 77);

  // Let's update again. New values should win.
  context.put(ContextUpdate{0, "SOME MORE"});
  context.put(ContextUpdate{10, uint32_t{16}});
  v1 = std::get_if<std::string>(&context.get(0));
  BOOST_REQUIRE(v1 != nullptr);
  BOOST_CHECK(*v1 == "SOME MORE");
  v2 = std::get_if<std::string>(&context.get(1));
  BOOST_CHECK(v2 == nullptr);
  v3 = std::get_if<uint32_t>(&context.get(10));
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

BOOST_AUTO_TEST_CASE(DataQuery)
{
  auto empty_bindings = [](std::runtime_error const& ex) -> bool {
    BOOST_CHECK_EQUAL(ex.what(), "Parse error: empty binding string");
    return true;
  };
  auto missing_origin = [](std::runtime_error const& ex) -> bool {
    BOOST_CHECK_EQUAL(ex.what(), "Parse error: origin needs to be between 1 and 4 char long");
    return true;
  };
  auto missing_description = [](std::runtime_error const& ex) -> bool {
    BOOST_CHECK_EQUAL(ex.what(), "Parse error: description needs to be between 1 and 16 char long");
    return true;
  };
  auto missing_subspec = [](std::runtime_error const& ex) -> bool {
    BOOST_CHECK_EQUAL(ex.what(), "Parse error: Expected a number");
    return true;
  };
  auto missing_timemodulo = [](std::runtime_error const& ex) -> bool {
    BOOST_CHECK_EQUAL(ex.what(), "Parse error: Expected a number");
    return true;
  };
  auto trailing_semicolon = [](std::runtime_error const& ex) -> bool {
    BOOST_CHECK_EQUAL(ex.what(), "Parse error: Remove trailing ;");
    return true;
  };
  // Empty query.
  BOOST_CHECK(DataDescriptorQueryBuilder::parse().empty() == true);
  // Empty bindings.
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse(":"), std::runtime_error, empty_bindings);
  // Missing origin
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:"), std::runtime_error, missing_origin);
  // Origin too long
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:bacjasbjkca"), std::runtime_error, missing_origin);
  // This is a valid expression, short for x:TST/*/* or x:TST/$1/$2
  BOOST_CHECK_NO_THROW(DataDescriptorQueryBuilder::parse("x:TST"));
  // This one is not, as we expect a description after a /
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:TST/"), std::runtime_error, missing_description);
  // This one is not, as the description is too long
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:TST/cdjancajncjancjkancjkadncancnacaklmcak"), std::runtime_error, missing_description);
  // This one is ok, short for "x:TST/A1/*"
  BOOST_CHECK_NO_THROW(DataDescriptorQueryBuilder::parse("x:TST/A1"));
  // This one is not, as subspec needs to be a value or a range.
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:TST/A1/"), std::runtime_error, missing_subspec);
  // Not valid as subspec should be a number.
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:TST/A1/a0"), std::runtime_error, missing_subspec);

  // Let's verify that the contents are correct.
  auto result0 = DataDescriptorQueryBuilder::parse("x:TST/A1/77");
  BOOST_CHECK_EQUAL(result0.size(), 1);
  DataDescriptorMatcher expectedMatcher00{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TST"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"A1"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{77},
        std::make_unique<DataDescriptorMatcher>(DataDescriptorMatcher::Op::Just,
                                                StartTimeValueMatcher{ContextRef{0}})))};
  auto matcher = std::get_if<DataDescriptorMatcher>(&result0[0].matcher);
  BOOST_REQUIRE(matcher != nullptr);
  BOOST_CHECK(expectedMatcher00 == *matcher);
  std::ostringstream ss0;
  ss0 << *matcher;
  std::ostringstream expectedSS00;
  expectedSS00 << expectedMatcher00;
  BOOST_CHECK_EQUAL(ss0.str(), "(and origin:TST (and description:A1 (and subSpec:77 (just startTime:$0 ))))");
  BOOST_CHECK_EQUAL(expectedSS00.str(), "(and origin:TST (and description:A1 (and subSpec:77 (just startTime:$0 ))))");
  BOOST_CHECK_EQUAL(ss0.str(), expectedSS00.str());

  // This is valid. TimeModulo is 1.
  BOOST_CHECK_NO_THROW(DataDescriptorQueryBuilder::parse("x:TST/A1/0"));
  // Not valid as timemodulo should be a number.
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:TST/A1/0%"), std::runtime_error, missing_timemodulo);
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:TST/A1/0\%oabdian"), std::runtime_error, missing_timemodulo);
  // This is valid.
  BOOST_CHECK_NO_THROW(DataDescriptorQueryBuilder::parse("x:TST/A1/0%1"));
  // This is not valid.
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:TST/A1/0%1;:"), std::runtime_error, empty_bindings);
  // This is not valid.
  BOOST_CHECK_EXCEPTION(DataDescriptorQueryBuilder::parse("x:TST/A1/0%1;"), std::runtime_error, trailing_semicolon);
  // This is valid.
  BOOST_CHECK_NO_THROW(DataDescriptorQueryBuilder::parse("x:TST/A1/0%1;x:TST/A2"));
  // Let's verify that the contents are correct.
  auto result1 = DataDescriptorQueryBuilder::parse("x:TST/A1/0%1;y:TST/A2");
  BOOST_CHECK_EQUAL(result1.size(), 2);

  std::ostringstream ops;
  ops << DataDescriptorMatcher::Op::And
      << DataDescriptorMatcher::Op::Or
      << DataDescriptorMatcher::Op::Xor
      << DataDescriptorMatcher::Op::Just;
  BOOST_CHECK_EQUAL(ops.str(), "andorxorjust");
}
