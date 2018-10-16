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

  BOOST_CHECK(matcher.match(header0) == true);
  BOOST_CHECK(matcher.match(header1) == false);
  BOOST_CHECK(matcher.match(header2) == false);
  BOOST_CHECK(matcher.match(header3) == false);
  BOOST_CHECK(matcher.match(header4) == false);

  DataDescriptorMatcher matcher1{
    DataDescriptorMatcher::Op::Or,
    OriginValueMatcher{ "TPC" },
    OriginValueMatcher{ "ITS" }
  };

  BOOST_CHECK(matcher1.match(header0) == true);
  BOOST_CHECK(matcher1.match(header1) == true);
  BOOST_CHECK(matcher1.match(header2) == true);
  BOOST_CHECK(matcher1.match(header3) == true);
  BOOST_CHECK(matcher1.match(header4) == false);

  DataDescriptorMatcher matcher2{
    DataDescriptorMatcher::Op::Just,
    DescriptionValueMatcher{ "TRACKLET" }
  };

  BOOST_CHECK(matcher2.match(header0) == false);
  BOOST_CHECK(matcher2.match(header1) == true);
  BOOST_CHECK(matcher2.match(header2) == true);
  BOOST_CHECK(matcher2.match(header3) == false);
  BOOST_CHECK(matcher2.match(header4) == true);
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

  auto matcher1 = DataDescriptorQueryBuilder::buildFromKeepConfig("TPC/CLUSTERS/1");
  BOOST_CHECK(matcher1->match(header0) == true);
  BOOST_CHECK(matcher1->match(header1) == false);
  BOOST_CHECK(matcher1->match(header2) == false);
  BOOST_CHECK(matcher1->match(header3) == false);
  BOOST_CHECK(matcher1->match(header4) == false);

  auto matcher2 = DataDescriptorQueryBuilder::buildFromKeepConfig("ITS/TRACKLET/2");
  BOOST_CHECK(matcher2->match(header0) == false);
  BOOST_CHECK(matcher2->match(header1) == true);
  BOOST_CHECK(matcher2->match(header2) == false);
  BOOST_CHECK(matcher2->match(header3) == false);
  BOOST_CHECK(matcher2->match(header4) == false);

  auto matcher3 = DataDescriptorQueryBuilder::buildFromKeepConfig("TPC/CLUSTERS/1,ITS/TRACKLET/2");
  BOOST_CHECK(matcher3->match(header0) == true);
  BOOST_CHECK(matcher3->match(header1) == true);
  BOOST_CHECK(matcher3->match(header2) == false);
  BOOST_CHECK(matcher3->match(header3) == false);
  BOOST_CHECK(matcher3->match(header4) == false);

  auto matcher4 = DataDescriptorQueryBuilder::buildFromKeepConfig("");
  BOOST_CHECK(matcher4->match(header0) == false);
  BOOST_CHECK(matcher4->match(header1) == false);
  BOOST_CHECK(matcher4->match(header2) == false);
  BOOST_CHECK(matcher4->match(header3) == false);
  BOOST_CHECK(matcher4->match(header4) == false);
}
