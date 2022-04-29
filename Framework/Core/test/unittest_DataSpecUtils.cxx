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

#define BOOST_TEST_MODULE Test Framework DataSpecUtils
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/DataSpecUtils.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include <boost/test/unit_test.hpp>

#include <string>

using namespace o2;
using namespace o2::framework;

BOOST_AUTO_TEST_CASE(ConcreteData)
{
  OutputSpec spec{
    "TEST",
    "FOOO",
    1,
    Lifetime::Timeframe};

  InputSpec inputSpec{
    "binding",
    "TEST",
    "FOOO",
    1,
    Lifetime::Timeframe};

  BOOST_REQUIRE(DataSpecUtils::validate(inputSpec));

  {
    ConcreteDataMatcher concrete = DataSpecUtils::asConcreteDataMatcher(spec);
    BOOST_CHECK_EQUAL(std::string(concrete.origin.as<std::string>()), "TEST");
    BOOST_CHECK_EQUAL(std::string(concrete.description.as<std::string>()), "FOOO");
    BOOST_CHECK_EQUAL(concrete.subSpec, 1);
    BOOST_CHECK_EQUAL(DataSpecUtils::describe(spec), "TEST/FOOO/1");
    BOOST_CHECK_EQUAL(*DataSpecUtils::getOptionalSubSpec(spec), 1);

    ConcreteDataTypeMatcher dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
    BOOST_CHECK_EQUAL(std::string(dataType.origin.as<std::string>()), "TEST");
    BOOST_CHECK_EQUAL(std::string(dataType.description.as<std::string>()), "FOOO");

    BOOST_CHECK(DataSpecUtils::match(spec, ConcreteDataMatcher{"TEST", "FOOO", 1}));
    BOOST_CHECK(DataSpecUtils::match(spec, ConcreteDataMatcher{"TEST", "FOOO", 0}) == false);
    DataSpecUtils::updateMatchingSubspec(spec, 0);
    BOOST_CHECK(DataSpecUtils::match(spec, ConcreteDataMatcher{"TEST", "FOOO", 0}) == true);
    BOOST_CHECK(DataSpecUtils::match(inputSpec, ConcreteDataMatcher{"TEST", "FOOO", 1}));
    BOOST_CHECK(DataSpecUtils::match(inputSpec, ConcreteDataMatcher{"TEST", "FOOO", 0}) == false);
    DataSpecUtils::updateMatchingSubspec(inputSpec, 0);
    BOOST_CHECK(DataSpecUtils::match(inputSpec, ConcreteDataMatcher{"TEST", "FOOO", 0}) == true);
  }
}

BOOST_AUTO_TEST_CASE(WithWildCards)
{
  OutputSpec spec{
    {"TEST", "FOOO"},
    Lifetime::Timeframe};

  BOOST_CHECK_THROW(DataSpecUtils::asConcreteDataMatcher(spec), std::bad_variant_access);
  auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);

  BOOST_CHECK_EQUAL(std::string(dataType.origin.as<std::string>()), "TEST");
  BOOST_CHECK_EQUAL(std::string(dataType.description.as<std::string>()), "FOOO");

  BOOST_CHECK(DataSpecUtils::match(spec, ConcreteDataMatcher{"TEST", "FOOO", 1}));
  BOOST_CHECK(DataSpecUtils::match(spec, ConcreteDataMatcher{"TEST", "FOOO", 0}));
  BOOST_CHECK_EQUAL(DataSpecUtils::describe(spec), "TEST/FOOO");

  BOOST_CHECK(DataSpecUtils::getOptionalSubSpec(spec) == std::nullopt);
}

BOOST_AUTO_TEST_CASE(MatchingInputs)
{
  OutputSpec fullySpecified{
    "TEST",
    "FOOO",
    1,
    Lifetime::Timeframe};

  OutputSpec partialMatching{
    {"TEST", "FOOO"},
    Lifetime::Timeframe};

  auto matchingInput1 = DataSpecUtils::matchingInput(fullySpecified);
  ConcreteDataMatcher concrete = DataSpecUtils::asConcreteDataMatcher(matchingInput1);

  BOOST_CHECK_EQUAL(std::string(concrete.origin.as<std::string>()), "TEST");
  BOOST_CHECK_EQUAL(std::string(concrete.description.as<std::string>()), "FOOO");
  BOOST_CHECK_EQUAL(concrete.subSpec, 1);

  auto matchingInput2 = DataSpecUtils::matchingInput(partialMatching);
  // BOOST_CHECK_THROW(DataSpecUtils::asConcreteDataMatcher(matchingInput), std::bad_variant_access);
  // We need a ConcreteDataMatcher to check if the generated input is
  // correct, because a ConcreteDataTypeMatcher has one extra degree of
  // freedom.
  ConcreteDataMatcher concreteExample1{
    "TEST",
    "FOOO",
    0};
  ConcreteDataMatcher concreteExample2{
    "TEST",
    "FOOO",
    1};
  ConcreteDataMatcher concreteExample3{
    "BAR",
    "FOOO",
    0};
  ConcreteDataMatcher concreteExample4{
    "TEST",
    "BAR",
    0};
  BOOST_CHECK(DataSpecUtils::match(matchingInput2, concreteExample1) == true);
  BOOST_CHECK(DataSpecUtils::match(matchingInput2, concreteExample2) == true);
  BOOST_CHECK(DataSpecUtils::match(matchingInput2, concreteExample3) == false);
  BOOST_CHECK(DataSpecUtils::match(matchingInput2, concreteExample4) == false);

  BOOST_CHECK_THROW(DataSpecUtils::asConcreteDataMatcher(matchingInput2), std::runtime_error);
  auto concrete2 = DataSpecUtils::asConcreteDataMatcher(matchingInput1);
  BOOST_CHECK_EQUAL(concrete.origin.as<std::string>(), "TEST");
  BOOST_CHECK_EQUAL(concrete.description.as<std::string>(), "FOOO");
  BOOST_CHECK_EQUAL(concrete.subSpec, 1);
}

BOOST_AUTO_TEST_CASE(MatchingOutputs)
{
  OutputSpec output1{
    "TST", "A1", 0, Lifetime::Timeframe};

  OutputSpec output2{
    "TST", "B1", 0, Lifetime::Timeframe};

  OutputSpec output3{
    {"TST", "A1"}, Lifetime::Timeframe};

  InputSpec input1{
    "binding", "TST", "A1", 0, Lifetime::Timeframe};

  InputSpec input2{
    "binding", "TST", "A1", 1, Lifetime::Timeframe};

  InputSpec input3{
    "binding", "TST", "B1", 0, Lifetime::Timeframe};

  InputSpec input4{
    "binding", {"TST", "A1"}, Lifetime::Timeframe};

  // matching inputs to outputs
  BOOST_CHECK(DataSpecUtils::match(input1, output1) == true);
  BOOST_CHECK(DataSpecUtils::match(input1, output2) == false);
  BOOST_CHECK(DataSpecUtils::match(input1, output3) == false); // Wildcard on output!
  BOOST_CHECK(DataSpecUtils::match(input2, output1) == false);
  BOOST_CHECK(DataSpecUtils::match(input2, output2) == false);
  BOOST_CHECK(DataSpecUtils::match(input2, output3) == false); // Wildcard on output!
  BOOST_CHECK(DataSpecUtils::match(input3, output1) == false);
  BOOST_CHECK(DataSpecUtils::match(input3, output2) == true);
  BOOST_CHECK(DataSpecUtils::match(input3, output3) == false); // Wildcard on output!
  BOOST_CHECK(DataSpecUtils::match(input4, output1) == true);  // Wildcard in input!
  BOOST_CHECK(DataSpecUtils::match(input4, output2) == false);
  BOOST_CHECK(DataSpecUtils::match(input4, output3) == true); // Wildcard on both!

  // matching outputs to output definitions
  // ConcreteDataMatcher on both sides
  BOOST_CHECK(DataSpecUtils::match(output1, OutputSpec{"TST", "A1", 0}) == true);
  BOOST_CHECK(DataSpecUtils::match(output1, OutputSpec{"TST", "A1", 1}) == false);

  // ConcreteDataMatcher left, ConcreteDataTypeMatcher right (subspec ignored)
  BOOST_CHECK(DataSpecUtils::match(output1, OutputSpec{"TST", "A1"}) == true);

  // ConcreteDataTypeMatcher left (subspec ignored), ConcreteDataMatcher right
  BOOST_CHECK(DataSpecUtils::match(output3, OutputSpec{"TST", "A1", 0}) == true);
  BOOST_CHECK(DataSpecUtils::match(output3, OutputSpec{"TST", "A1", 1}) == true);

  // ConcreteDataTypeMatcher on both sides
  BOOST_CHECK(DataSpecUtils::match(output3, OutputSpec{"TST", "A1"}) == true);
}

BOOST_AUTO_TEST_CASE(PartialMatching)
{
  OutputSpec fullySpecifiedOutput{
    "TEST",
    "FOOO",
    1,
    Lifetime::Timeframe};

  InputSpec fullySpecifiedInput{
    "binding",
    "TSET",
    "FOOO",
    1,
    Lifetime::Timeframe};

  BOOST_CHECK(DataSpecUtils::partialMatch(fullySpecifiedOutput, header::DataOrigin("TEST")));
  BOOST_CHECK(DataSpecUtils::partialMatch(fullySpecifiedInput, header::DataOrigin("TSET")));

  BOOST_CHECK(DataSpecUtils::partialMatch(fullySpecifiedOutput, header::DataOrigin("FOO")) == false);
  BOOST_CHECK(DataSpecUtils::partialMatch(fullySpecifiedInput, header::DataOrigin("FOO")) == false);

  BOOST_CHECK(DataSpecUtils::partialMatch(fullySpecifiedOutput, header::DataDescription("TEST")) == false);
  BOOST_CHECK(DataSpecUtils::partialMatch(fullySpecifiedInput, header::DataDescription("TSET")) == false);

  BOOST_CHECK(DataSpecUtils::partialMatch(fullySpecifiedOutput, header::DataDescription("FOOO")) == true);
  BOOST_CHECK(DataSpecUtils::partialMatch(fullySpecifiedInput, header::DataDescription("FOOO")) == true);
}

BOOST_AUTO_TEST_CASE(GetOptionalSubSpecWithMatcher)
{
  InputSpec fullInputSpec{
    "binding",
    "TSET", "FOOO", 1,
    Lifetime::Timeframe};

  InputSpec wildcardInputSpec{
    "binding",
    {"TSET", "FOOO"},
    Lifetime::Timeframe};

  auto fromQueryInputSpec = DataDescriptorQueryBuilder::parse("x:TST/A1/77;y:STS/A2;z:FOO/A3");

  BOOST_CHECK(*DataSpecUtils::getOptionalSubSpec(fullInputSpec) == 1);
  BOOST_CHECK(DataSpecUtils::getOptionalSubSpec(wildcardInputSpec) == std::nullopt);
  BOOST_REQUIRE_EQUAL(fromQueryInputSpec.size(), 3);
  BOOST_CHECK(DataSpecUtils::getOptionalSubSpec(fromQueryInputSpec[0]) != std::nullopt);
  BOOST_CHECK(DataSpecUtils::getOptionalSubSpec(fromQueryInputSpec[0]) == 77);
  BOOST_CHECK(DataSpecUtils::getOptionalSubSpec(fromQueryInputSpec[1]) == std::nullopt);
  BOOST_CHECK(DataSpecUtils::getOptionalSubSpec(fromQueryInputSpec[2]) == std::nullopt);
  DataSpecUtils::updateMatchingSubspec(fromQueryInputSpec[2], 10);
  BOOST_CHECK(DataSpecUtils::getOptionalSubSpec(fromQueryInputSpec[2]) == 10);

  auto dataType1 = DataSpecUtils::asConcreteDataTypeMatcher(fullInputSpec);
  BOOST_CHECK_EQUAL(std::string(dataType1.origin.as<std::string>()), "TSET");
  BOOST_CHECK_EQUAL(std::string(dataType1.description.as<std::string>()), "FOOO");

  auto dataType2 = DataSpecUtils::asConcreteDataTypeMatcher(wildcardInputSpec);
  BOOST_CHECK_EQUAL(std::string(dataType2.origin.as<std::string>()), "TSET");
  BOOST_CHECK_EQUAL(std::string(dataType2.description.as<std::string>()), "FOOO");
}

BOOST_AUTO_TEST_CASE(TestMatcherFromDescription)
{
  auto fromQueryInputSpec = DataSpecUtils::dataDescriptorMatcherFrom(header::DataDescription{"TSET"});
  InputSpec ddSpec{
    "binding",
    std::move(fromQueryInputSpec)};

  BOOST_CHECK_EQUAL(DataSpecUtils::asConcreteDataDescription(ddSpec).as<std::string>(), "TSET");
}

BOOST_AUTO_TEST_CASE(TestMatcherFromConcrete)
{
  auto fromQueryInputSpec = DataSpecUtils::dataDescriptorMatcherFrom(ConcreteDataMatcher{"TSET", "FOO", 1});
  InputSpec ddSpec{
    "binding",
    std::move(fromQueryInputSpec)};

  auto concrete = DataSpecUtils::asConcreteDataMatcher(ddSpec);

  BOOST_CHECK_EQUAL(concrete.origin.as<std::string>(), "TSET");
  BOOST_CHECK_EQUAL(concrete.description.as<std::string>(), "FOO");
  BOOST_CHECK_EQUAL(concrete.subSpec, 1);
}

BOOST_AUTO_TEST_CASE(FindOutputSpec)
{
  std::vector<OutputSpec> specs = {
    {"TST", "DATA1", 0},
    {"TST", "DATA2", 0}};

  auto spec = DataSpecUtils::find(specs, {"TST"}, {"DATA1"}, 0);
  BOOST_CHECK(spec == specs[0]);
  BOOST_CHECK(DataSpecUtils::find(specs, {"TST"}, {"DATA3"}, 0) == std::nullopt);
}

BOOST_AUTO_TEST_CASE(FindInputSpec)
{
  std::vector<InputSpec> specs = {
    {"x", "TST", "DATA1", 0},
    {"y", "TST", "DATA2", 0}};

  auto spec = DataSpecUtils::find(specs, {"TST"}, {"DATA1"}, 0);
  BOOST_CHECK(spec == specs[0]);
  BOOST_CHECK(DataSpecUtils::find(specs, {"TST"}, {"DATA3"}, 0) == std::nullopt);
}

BOOST_AUTO_TEST_CASE(GettingConcreteMembers)
{
  InputSpec fullySpecifiedInput{
    "binding",
    "TSET",
    "FOOO",
    1,
    Lifetime::Timeframe};

  InputSpec wildcardInputSpec{
    "binding",
    {"TSET", "FOOO"},
    Lifetime::Timeframe};

  auto justOriginInputSpec = DataDescriptorQueryBuilder::parse("x:TST");

  BOOST_CHECK_EQUAL(DataSpecUtils::asConcreteOrigin(fullySpecifiedInput).as<std::string>(), "TSET");
  BOOST_CHECK_EQUAL(DataSpecUtils::asConcreteDataDescription(fullySpecifiedInput).as<std::string>(), "FOOO");

  BOOST_CHECK_EQUAL(DataSpecUtils::asConcreteOrigin(wildcardInputSpec).as<std::string>(), "TSET");
  BOOST_CHECK_EQUAL(DataSpecUtils::asConcreteDataDescription(wildcardInputSpec).as<std::string>(), "FOOO");

  BOOST_REQUIRE_EQUAL(justOriginInputSpec.size(), 1);
  BOOST_CHECK_EQUAL(DataSpecUtils::asConcreteOrigin(justOriginInputSpec.at(0)).as<std::string>(), "TST");
  BOOST_CHECK_THROW(DataSpecUtils::asConcreteDataDescription(justOriginInputSpec.at(0)).as<std::string>(), RuntimeErrorRef);
}

BOOST_AUTO_TEST_CASE(Includes)
{
  InputSpec concreteInput1{"binding", "TSET", "FOOO", 1, Lifetime::Timeframe};
  InputSpec concreteInput2{"binding", "TSET", "BAAAR", 1, Lifetime::Timeframe};
  InputSpec wildcardInput1{"binding", {"TSET", "FOOO"}, Lifetime::Timeframe};
  InputSpec wildcardInput2{"binding", {"TSET", "BAAAR"}, Lifetime::Timeframe};

  // wildcard and concrete
  BOOST_CHECK(DataSpecUtils::includes(wildcardInput1, concreteInput1));
  BOOST_CHECK(!DataSpecUtils::includes(wildcardInput1, concreteInput2));
  BOOST_CHECK(!DataSpecUtils::includes(concreteInput1, wildcardInput1));
  BOOST_CHECK(!DataSpecUtils::includes(concreteInput2, wildcardInput1));

  // concrete and concrete
  BOOST_CHECK(DataSpecUtils::includes(concreteInput1, concreteInput1));
  BOOST_CHECK(!DataSpecUtils::includes(concreteInput1, concreteInput2));
  BOOST_CHECK(!DataSpecUtils::includes(concreteInput2, concreteInput1));

  // wildcard and wildcard
  BOOST_CHECK(DataSpecUtils::includes(wildcardInput1, wildcardInput1));
  BOOST_CHECK(!DataSpecUtils::includes(wildcardInput1, wildcardInput2));
  BOOST_CHECK(!DataSpecUtils::includes(wildcardInput2, wildcardInput1));

  auto inputsFromQuery = DataDescriptorQueryBuilder::parse("b0:TST/FOO/0;b1:TST/FOO/1");
  BOOST_CHECK(!DataSpecUtils::includes(inputsFromQuery[0], inputsFromQuery[1]));
  BOOST_CHECK(!DataSpecUtils::includes(inputsFromQuery[1], inputsFromQuery[0]));
}

BOOST_AUTO_TEST_CASE(optionalConcreteDataMatcherFrom)
{
  using namespace data_matcher;

  // the standard structure of fully qualified data descriptor
  DataDescriptorMatcher matcher1{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TPC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          StartTimeValueMatcher(ContextRef{ContextPos::STARTTIME_POS}))))};

  // also fully qualified, but interchanged components
  DataDescriptorMatcher matcher2{
    DataDescriptorMatcher::Op::And,
    DescriptionValueMatcher{"CLUSTERS"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      OriginValueMatcher{"TPC"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          StartTimeValueMatcher(ContextRef{ContextPos::STARTTIME_POS}))))};

  // matcher with no unique subSpec
  DataDescriptorMatcher matcher3{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TPC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Not,
          SubSpecificationTypeValueMatcher{0}),
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          StartTimeValueMatcher(ContextRef{ContextPos::STARTTIME_POS}))))};

  // another matcher with no unique subSpec
  DataDescriptorMatcher matcher4{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TPC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Or,
          SubSpecificationTypeValueMatcher{0},
          SubSpecificationTypeValueMatcher{1}),
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          StartTimeValueMatcher(ContextRef{ContextPos::STARTTIME_POS}))))};

  // unique origin and description only
  DataDescriptorMatcher matcher5{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TPC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ContextRef{2}},
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          StartTimeValueMatcher(ContextRef{ContextPos::STARTTIME_POS}))))};

  // no subspec in the matcher
  DataDescriptorMatcher matcher6{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TPC"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"CLUSTERS"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        ConstantValueMatcher{true},
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          StartTimeValueMatcher(ContextRef{ContextPos::STARTTIME_POS}))))};

  DataDescriptorMatcher matcher7 = DataSpecUtils::dataDescriptorMatcherFrom(ConcreteDataMatcher{"ITS", "RAWDATA", 0});
  DataDescriptorMatcher matcher8 = DataSpecUtils::dataDescriptorMatcherFrom(ConcreteDataTypeMatcher{"ITS", "RAWDATA"});

  auto check = [](DataDescriptorMatcher const& matcher, bool expectConcreteDataMatcher, ConcreteDataMatcher compare = {"", "", 0}) {
    auto concrete = DataSpecUtils::optionalConcreteDataMatcherFrom(matcher);
    BOOST_CHECK_EQUAL(concrete.has_value(), expectConcreteDataMatcher);
    if (concrete.has_value()) {
      BOOST_CHECK(*concrete == compare);
    }
  };

  check(matcher1, true, ConcreteDataMatcher{"TPC", "CLUSTERS", 1});
  check(matcher2, true, ConcreteDataMatcher{"TPC", "CLUSTERS", 1});
  check(matcher3, false);
  check(matcher4, false);
  check(matcher5, false);
  check(matcher6, false);
  check(matcher7, true, ConcreteDataMatcher{"ITS", "RAWDATA", 0});
  check(matcher8, false);
}
