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

#include <boost/test/unit_test.hpp>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::framework::data_matcher;

BOOST_AUTO_TEST_CASE(TestMatcherInvariants)
{
  DataHeader header0;
  header0.dataOrigin = "TPC";
  header0.dataDescription = "CLUSTERS";
  header0.subSpecification = 1;
  std::vector<ContextElement> context;

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

  std::vector<ContextElement> context;
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
  std::vector<ContextElement> context;

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
  std::vector<ContextElement> context(1);

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ ContextRef{ 0 } },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "CLUSTERS" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 1 },
        ConstantValueMatcher{ true }))
  };

  DataHeader header0;
  header0.dataOrigin = "TPC";
  header0.dataDescription = "CLUSTERS";
  header0.subSpecification = 1;

  BOOST_CHECK(matcher.match(header0, context) == true);
  auto s = std::get_if<std::string>(&context[0].value);
  BOOST_CHECK(s != nullptr);
  BOOST_CHECK(*s == "TPC");

  // This will not match, because ContextRef{0} is bound
  // to TPC already.
  DataHeader header1;
  header1.dataOrigin = "ITS";
  header1.dataDescription = "CLUSTERS";
  header1.subSpecification = 1;

  BOOST_CHECK(matcher.match(header1, context) == false);
  auto s1 = std::get_if<std::string>(&context[0].value);
  BOOST_CHECK(s1 != nullptr);
  BOOST_CHECK(*s1 == "TPC");
}

BOOST_AUTO_TEST_CASE(TestInputSpecMatching)
{
  InputSpec spec0{ "spec0", "TPC", "CLUSTERS", 1 };
  InputSpec spec1{ "spec1", "ITS", "TRACKLET", 2 };
  InputSpec spec2{ "spec2", "ITS", "TRACKLET", 1 };
  InputSpec spec3{ "spec3", "TPC", "CLUSTERS", 0 };
  InputSpec spec4{ "spec4", "TRD", "TRACKLET", 0 };

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

  std::vector<ContextElement> context;

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
