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

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/InputSpec.h"
#include "Headers/Stack.h"

#include <catch_amalgamated.hpp>
#include <variant>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::framework::data_matcher;

TEST_CASE("TestMatcherInvariants")
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
  REQUIRE(matcher.match(header0, context) == true);
  REQUIRE(matcher.match(header1, context) == false);
  REQUIRE(matcher.match(header2, context) == false);
  REQUIRE(matcher.match(header3, context) == false);
  REQUIRE(matcher.match(header4, context) == false);
  REQUIRE(matcher2.match(header0, context) == true);
  REQUIRE(matcher2.match(header1, context) == false);
  REQUIRE(matcher2.match(header2, context) == false);
  REQUIRE(matcher2.match(header3, context) == false);
  REQUIRE(matcher2.match(header4, context) == false);

  REQUIRE(matcher2 == matcher);

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      OriginValueMatcher{"TPC"}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      OriginValueMatcher{"TPC"}};
    REQUIRE(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      DescriptionValueMatcher{"TRACKS"}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      DescriptionValueMatcher{"TRACKS"}};
    REQUIRE(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      SubSpecificationTypeValueMatcher{1}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      SubSpecificationTypeValueMatcher{1}};
    REQUIRE(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Just,
      ConstantValueMatcher{1}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Just,
      ConstantValueMatcher{1}};
    REQUIRE(matcherA == matcherB);
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
    REQUIRE(!(matcherA == matcherB));
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
    REQUIRE(matcherA == matcherB);
  }

  {
    DataDescriptorMatcher matcherA{
      DataDescriptorMatcher::Op::Not,
      OriginValueMatcher{"TPC"}};
    DataDescriptorMatcher matcherB{
      DataDescriptorMatcher::Op::Not,
      DescriptionValueMatcher{"TRACKLET"}};
    DataDescriptorMatcher matcherC{
      DataDescriptorMatcher::Op::Not,
      SubSpecificationTypeValueMatcher{1}};

    REQUIRE(matcherA.match(header0, context) == false);
    REQUIRE(matcherA.match(header1, context) == true);
    REQUIRE(matcherA.match(header4, context) == true);
    REQUIRE(matcherB.match(header0, context) == true);
    REQUIRE(matcherB.match(header1, context) == false);
    REQUIRE(matcherB.match(header4, context) == false);
    REQUIRE(matcherC.match(header0, context) == false);
    REQUIRE(matcherC.match(header1, context) == true);
    REQUIRE(matcherC.match(header4, context) == true);
  }
}

TEST_CASE("TestSimpleMatching")
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
  REQUIRE(matcher.match(header0, context) == true);
  REQUIRE(matcher.match(header1, context) == false);
  REQUIRE(matcher.match(header2, context) == false);
  REQUIRE(matcher.match(header3, context) == false);
  REQUIRE(matcher.match(header4, context) == false);

  DataDescriptorMatcher matcher1{
    DataDescriptorMatcher::Op::Or,
    OriginValueMatcher{"TPC"},
    OriginValueMatcher{"ITS"}};

  REQUIRE(matcher1.match(header0, context) == true);
  REQUIRE(matcher1.match(header1, context) == true);
  REQUIRE(matcher1.match(header2, context) == true);
  REQUIRE(matcher1.match(header3, context) == true);
  REQUIRE(matcher1.match(header4, context) == false);

  DataDescriptorMatcher matcher2{
    DataDescriptorMatcher::Op::Just,
    DescriptionValueMatcher{"TRACKLET"}};

  REQUIRE(matcher2.match(header0, context) == false);
  REQUIRE(matcher2.match(header1, context) == true);
  REQUIRE(matcher2.match(header2, context) == true);
  REQUIRE(matcher2.match(header3, context) == false);
  REQUIRE(matcher2.match(header4, context) == true);
}

TEST_CASE("TestDataDescriptorQueryBuilder")
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
  REQUIRE(matcher1.matcher->match(header0, context) == true);
  REQUIRE(matcher1.matcher->match(header1, context) == false);
  REQUIRE(matcher1.matcher->match(header2, context) == false);
  REQUIRE(matcher1.matcher->match(header3, context) == false);
  REQUIRE(matcher1.matcher->match(header4, context) == false);

  auto matcher2 = DataDescriptorQueryBuilder::buildFromKeepConfig("ITS/TRACKLET/2");
  REQUIRE(matcher2.matcher->match(header0, context) == false);
  REQUIRE(matcher2.matcher->match(header1, context) == true);
  REQUIRE(matcher2.matcher->match(header2, context) == false);
  REQUIRE(matcher2.matcher->match(header3, context) == false);
  REQUIRE(matcher2.matcher->match(header4, context) == false);

  auto matcher3 = DataDescriptorQueryBuilder::buildFromKeepConfig("TPC/CLUSTERS/1,ITS/TRACKLET/2");
  REQUIRE(matcher3.matcher->match(header0, context) == true);
  REQUIRE(matcher3.matcher->match(header1, context) == true);
  REQUIRE(matcher3.matcher->match(header2, context) == false);
  REQUIRE(matcher3.matcher->match(header3, context) == false);
  REQUIRE(matcher3.matcher->match(header4, context) == false);

  auto matcher4 = DataDescriptorQueryBuilder::buildFromKeepConfig("");
  REQUIRE(matcher4.matcher->match(header0, context) == false);
  REQUIRE(matcher4.matcher->match(header1, context) == false);
  REQUIRE(matcher4.matcher->match(header2, context) == false);
  REQUIRE(matcher4.matcher->match(header3, context) == false);
  REQUIRE(matcher4.matcher->match(header4, context) == false);
}

// This checks matching using variables
TEST_CASE("TestMatchingVariables")
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

  REQUIRE(matcher.match(header0, context) == true);
  auto s = std::get_if<std::string>(&context.get(0));
  REQUIRE(s != nullptr);
  REQUIRE(*s == "TPC");
  auto v = std::get_if<o2::header::DataHeader::SubSpecificationType>(&context.get(1));
  REQUIRE(v != nullptr);
  REQUIRE(*v == 1);

  // This will not match, because ContextRef{0} is bound
  // to TPC already.
  DataHeader header1;
  header1.dataOrigin = "ITS";
  header1.dataDescription = "CLUSTERS";
  header1.subSpecification = 1;

  REQUIRE(matcher.match(header1, context) == false);
  auto s1 = std::get_if<std::string>(&context.get(0));
  REQUIRE(s1 != nullptr);
  REQUIRE(*s1 == "TPC");
}

TEST_CASE("TestInputSpecMatching")
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

  REQUIRE(matcher.match(spec0, context) == true);
  REQUIRE(matcher.match(spec1, context) == false);
  REQUIRE(matcher.match(spec2, context) == false);
  REQUIRE(matcher.match(spec3, context) == false);
  REQUIRE(matcher.match(spec4, context) == false);

  DataDescriptorMatcher matcher1{
    DataDescriptorMatcher::Op::Or,
    OriginValueMatcher{"TPC"},
    OriginValueMatcher{"ITS"}};

  REQUIRE(matcher1.match(spec0, context) == true);
  REQUIRE(matcher1.match(spec1, context) == true);
  REQUIRE(matcher1.match(spec2, context) == true);
  REQUIRE(matcher1.match(spec3, context) == true);
  REQUIRE(matcher1.match(spec4, context) == false);

  DataDescriptorMatcher matcher2{
    DataDescriptorMatcher::Op::Just,
    DescriptionValueMatcher{"TRACKLET"}};

  REQUIRE(matcher2.match(spec0, context) == false);
  REQUIRE(matcher2.match(spec1, context) == true);
  REQUIRE(matcher2.match(spec2, context) == true);
  REQUIRE(matcher2.match(spec3, context) == false);
  REQUIRE(matcher2.match(spec4, context) == true);
}

TEST_CASE("TestStartTimeMatching")
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
  REQUIRE(s2dph != nullptr);
  REQUIRE(s2dph->startTime == 123);
  REQUIRE(matcher.match(s, context) == true);
  auto vPtr = std::get_if<uint64_t>(&context.get(0));
  REQUIRE(vPtr != nullptr);
  REQUIRE(*vPtr == 123);
}

/// If a query matches only partially, we do not want
/// to pollute the context with partial results.
TEST_CASE("TestAtomicUpdatesOfContext")
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
  REQUIRE(vPtr0 != nullptr);
  REQUIRE(vPtr1 != nullptr);
  REQUIRE(matcher.match(dh, context) == false);
  // We discard the updates, because there was no match
  context.discard();
  vPtr0 = std::get_if<None>(&context.get(0));
  vPtr1 = std::get_if<None>(&context.get(1));
  REQUIRE(vPtr0 != nullptr);
  REQUIRE(vPtr1 != nullptr);
}

TEST_CASE("TestVariableContext")
{
  VariableContext context;
  // Put some updates, but do not commit them
  // we should still be able to retrieve them
  // (just slower).
  context.put(ContextUpdate{0, "A TEST"});
  context.put(ContextUpdate{10, uint32_t{77}});
  auto v1 = std::get_if<std::string>(&context.get(0));
  REQUIRE(v1 != nullptr);
  REQUIRE(*v1 == "A TEST");
  auto v2 = std::get_if<std::string>(&context.get(1));
  REQUIRE(v2 == nullptr);
  auto v3 = std::get_if<uint32_t>(&context.get(10));
  REQUIRE(v3 != nullptr);
  REQUIRE(*v3 == 77);
  context.commit();
  // After commits everything is the same
  v1 = std::get_if<std::string>(&context.get(0));
  REQUIRE(v1 != nullptr);
  REQUIRE(*v1 == "A TEST");
  v2 = std::get_if<std::string>(&context.get(1));
  REQUIRE(v2 == nullptr);
  v3 = std::get_if<uint32_t>(&context.get(10));
  REQUIRE(v3 != nullptr);
  REQUIRE(*v3 == 77);

  // Let's update again. New values should win.
  context.put(ContextUpdate{0, "SOME MORE"});
  context.put(ContextUpdate{10, uint32_t{16}});
  v1 = std::get_if<std::string>(&context.get(0));
  REQUIRE(v1 != nullptr);
  REQUIRE(*v1 == "SOME MORE");
  v2 = std::get_if<std::string>(&context.get(1));
  REQUIRE(v2 == nullptr);
  v3 = std::get_if<uint32_t>(&context.get(10));
  REQUIRE(v3 != nullptr);
  REQUIRE(*v3 == 16);

  // Until we discard
  context.discard();
  v1 = std::get_if<std::string>(&context.get(0));
  REQUIRE(v1 != nullptr);
  REQUIRE(*v1 == "A TEST");
  auto n = std::get_if<None>(&context.get(1));
  REQUIRE(n != nullptr);
  v3 = std::get_if<uint32_t>(&context.get(10));
  REQUIRE(v3 != nullptr);
  REQUIRE(*v3 == 77);

  // Let's update again. New values should win.
  context.put(ContextUpdate{0, "SOME MORE"});
  context.put(ContextUpdate{10, uint32_t{16}});
  v1 = std::get_if<std::string>(&context.get(0));
  REQUIRE(v1 != nullptr);
  REQUIRE(*v1 == "SOME MORE");
  v2 = std::get_if<std::string>(&context.get(1));
  REQUIRE(v2 == nullptr);
  v3 = std::get_if<uint32_t>(&context.get(10));
  REQUIRE(v3 != nullptr);
  REQUIRE(*v3 == 16);

  // Until we discard again, using reset
  context.reset();
  auto n1 = std::get_if<None>(&context.get(0));
  REQUIRE(n1 != nullptr);
  auto n2 = std::get_if<None>(&context.get(1));
  REQUIRE(n2 != nullptr);
  auto n3 = std::get_if<None>(&context.get(10));
  REQUIRE(n3 != nullptr);

  // auto d3 = std::get_if<uint64_t>(&context.get(0));
  // REQUIRE(d1 == nullptr);;
  // REQUIRE(d2 == nullptr);;
  // REQUIRE(d3 == nullptr);;
}

TEST_CASE("DataQuery")
{
  auto empty_bindings = [](std::runtime_error const& ex) -> bool {
    REQUIRE(std::string(ex.what()) == "Parse error: empty binding string");
    return true;
  };
  auto missing_origin = [](std::runtime_error const& ex) -> bool {
    REQUIRE(std::string(ex.what()) == "Parse error: origin needs to be between 1 and 4 char long");
    return true;
  };
  auto missing_description = [](std::runtime_error const& ex) -> bool {
    REQUIRE(std::string(ex.what()) == "Parse error: description needs to be between 1 and 16 char long");
    return true;
  };
  auto missing_subspec = [](std::runtime_error const& ex) -> bool {
    REQUIRE(std::string(ex.what()) == "Parse error: Expected a number");
    return true;
  };
  auto missing_timemodulo = [](std::runtime_error const& ex) -> bool {
    REQUIRE(std::string(ex.what()) == "Parse error: Expected a number");
    return true;
  };
  auto trailing_semicolon = [](std::runtime_error const& ex) -> bool {
    REQUIRE(std::string(ex.what()) == "Parse error: Remove trailing ;");
    return true;
  };

  auto missing_value = [](std::runtime_error const& ex) -> bool {
    REQUIRE(std::string(ex.what()) == "Parse error: value needs to be between 1 and 1000 char long");
    return true;
  };
  auto missing_key = [](std::runtime_error const& ex) -> bool {
    REQUIRE(std::string(ex.what()) == "Parse error: missing value for attribute key");
    return true;
  };
  // Empty query.
  REQUIRE(DataDescriptorQueryBuilder::parse().empty() == true);
  // Empty bindings.
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse(":"), std::runtime_error);
  // Missing origin
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:"), std::runtime_error);
  // Origin too long
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:bacjasbjkca"), std::runtime_error);
  // This is a valid expression, short for x:TST/*/* or x:TST/$1/$2
  REQUIRE_NOTHROW(DataDescriptorQueryBuilder::parse("x:TST"));
  // This one is not, as we expect a description after a /
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/"), std::runtime_error);
  // This one is not, as the description is too long
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/cdjancajncjancjkancjkadncancnacaklmcak"), std::runtime_error);
  // This one is ok, short for "x:TST/A1/*"
  REQUIRE_NOTHROW(DataDescriptorQueryBuilder::parse("x:TST/A1"));
  // This one is not, as subspec needs to be a value or a range.
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/A1/"), std::runtime_error);
  // Not valid as subspec should be a number.
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/A1/a0"), std::runtime_error);

  // Let's verify that the contents are correct.
  auto result0 = DataDescriptorQueryBuilder::parse("x:TST/A1/77");
  REQUIRE(result0.size() == 1);
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
  REQUIRE(matcher != nullptr);
  REQUIRE(expectedMatcher00 == *matcher);
  std::ostringstream ss0;
  ss0 << *matcher;
  std::ostringstream expectedSS00;
  expectedSS00 << expectedMatcher00;
  REQUIRE(ss0.str() == "(and origin:TST (and description:A1 (and subSpec:77 (just startTime:$0 ))))");
  REQUIRE(expectedSS00.str() == "(and origin:TST (and description:A1 (and subSpec:77 (just startTime:$0 ))))");
  REQUIRE(ss0.str() == expectedSS00.str());

  // This is valid. TimeModulo is 1.
  REQUIRE_NOTHROW(DataDescriptorQueryBuilder::parse("x:TST/A1/0"));
  // Not valid as timemodulo should be a number.
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/A1/0%"), std::runtime_error);
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/A1/0\%oabdian"), std::runtime_error);
  // This is valid.
  REQUIRE_NOTHROW(DataDescriptorQueryBuilder::parse("x:TST/A1/0%1"));
  // This is not valid.
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/A1/0%1;:"), std::runtime_error);
  // This is not valid.
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/A1/0%1;"), std::runtime_error);
  // This is valid.
  REQUIRE_NOTHROW(DataDescriptorQueryBuilder::parse("x:TST/A1/0%1;x:TST/A2"));
  // Let's verify that the contents are correct.
  auto result1 = DataDescriptorQueryBuilder::parse("x:TST/A1/0%1;y:TST/A2");
  REQUIRE(result1.size() == 2);

  std::ostringstream ops;
  ops << DataDescriptorMatcher::Op::And
      << DataDescriptorMatcher::Op::Or
      << DataDescriptorMatcher::Op::Xor
      << DataDescriptorMatcher::Op::Just;
  REQUIRE(ops.str() == "andorxorjust");

  // Let's check the metadata associated to a query
  auto result2 = DataDescriptorQueryBuilder::parse("x:TST/A1/0?lifetime=condition");
  REQUIRE(result2[0].lifetime == Lifetime::Condition);

  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/A1/0?lifetime="), std::runtime_error);
  REQUIRE_THROWS_AS(DataDescriptorQueryBuilder::parse("x:TST/A1/0?"), std::runtime_error);

  auto result3 = DataDescriptorQueryBuilder::parse("x:TST/A1/0?key=value&key2=value2");
  REQUIRE(result3[0].metadata.size() == 2);

  auto result4 = DataDescriptorQueryBuilder::parse("x:TST/A1/0?lifetime=condition&ccdb-path=GLO/Config/GRPECS&key3=value3");
  REQUIRE(result4.size() == 1);
  result4[0].lifetime = Lifetime::Condition;
  REQUIRE(result4[0].metadata.size() == 3);
  REQUIRE(result4[0].metadata[0].name == "lifetime");
  REQUIRE(result4[0].metadata[0].defaultValue.get<std::string>() == "condition");
  REQUIRE(result4[0].metadata[1].name == "ccdb-path");
  REQUIRE(result4[0].metadata[1].defaultValue.get<std::string>() == "GLO/Config/GRPECS");
  REQUIRE(result4[0].metadata[2].name == "key3");
  REQUIRE(result4[0].metadata[2].defaultValue.get<std::string>() == "value3");

  // This is valid.
  REQUIRE_NOTHROW(DataDescriptorQueryBuilder::parse("x:TST/A1/0xccdb"));

  auto result5 = DataDescriptorQueryBuilder::parse("x:TST/A1/0?lifetime=sporadic&ccdb-path=GLO/Config/GRPECS&key3=value3");
  REQUIRE(result5.size() == 1);
  result5[0].lifetime = Lifetime::Sporadic;
  REQUIRE(result5[0].metadata.size() == 3);
  REQUIRE(result5[0].metadata[0].name == "lifetime");
  REQUIRE(result5[0].metadata[0].defaultValue.get<std::string>() == "sporadic");
  REQUIRE(result5[0].metadata[1].name == "ccdb-path");
  REQUIRE(result5[0].metadata[1].defaultValue.get<std::string>() == "GLO/Config/GRPECS");
  REQUIRE(result5[0].metadata[2].name == "key3");
  REQUIRE(result5[0].metadata[2].defaultValue.get<std::string>() == "value3");
}

// Make sure that 10 and 1 subspect are matched differently

TEST_CASE("MatchSubspec")
{
  DataHeader header0;
  header0.dataOrigin = "EMC";
  header0.dataDescription = "CELLSTRGR";
  header0.subSpecification = 10;
  VariableContext context;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"EMC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CELLSTRGR"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        ConstantValueMatcher{true}))};

  REQUIRE(matcher.match(header0, context) == false);
}
