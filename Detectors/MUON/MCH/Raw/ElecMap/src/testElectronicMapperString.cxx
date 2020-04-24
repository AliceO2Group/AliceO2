// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHWorkflow MapperString
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MapCRU.h"
#include "MapFEC.h"
#include <sstream>
#include <string>

// clang-format off
// solarId feeId cruLink  
//       #     #   0..11
std::string cruString = R"(
      1        0       0
     20        3      11
)";
std::string cruString2 = R"(
      1        0       0    anything
     20        3      11    anything extra will not matter
)";
// clang-format on

// clang-format off
// solarId   groupId    de   dsid0   dsid1   dsid2   dsid3   dsid4
//                         index=0 index=1 index=2 index=3 index=4
std::string fecString = R"(
         1         0   819     108       0     107       0     106
         1         1   919    1133       0    1134       0       0
)";
// clang-format on

using namespace o2::mch::raw;

BOOST_AUTO_TEST_CASE(DefautCtorCreatesAnEmptyMap)
{
  MapCRU cru("");
  BOOST_CHECK_EQUAL(cru.size(), 0);
}

BOOST_AUTO_TEST_CASE(ReadMapCRUSize)
{
  MapCRU cru(cruString);
  BOOST_CHECK_EQUAL(cru.size(), 2);
}

BOOST_AUTO_TEST_CASE(GetSolarId)
{
  MapCRU cru(cruString);
  FeeLinkId fl(3, 11);
  auto solarId = cru(fl);
  BOOST_CHECK_EQUAL(solarId.has_value(), true);
  BOOST_CHECK_EQUAL(solarId.value(), 20);
}

BOOST_AUTO_TEST_CASE(GetSolarIdIsOKEvenIfLineContainsMoreInformation)
{
  MapCRU cru(cruString2);
  FeeLinkId fl(3, 11);
  auto solarId = cru(fl);
  BOOST_CHECK_EQUAL(solarId.has_value(), true);
  BOOST_CHECK_EQUAL(solarId.value(), 20);
}

BOOST_AUTO_TEST_CASE(ObjectFromStringShouldHaveFiveFEC)
{
  MapFEC fec(fecString);
  BOOST_CHECK_EQUAL(fec.size(), 5);
}

BOOST_AUTO_TEST_CASE(RequestingInvalidLinkMustYieldInvalidDeDs)
{
  MapFEC fec(fecString);
  auto dsDetId = fec(DsElecId(2, 0, 0));
  BOOST_CHECK_EQUAL(dsDetId.has_value(), false);
}

BOOST_AUTO_TEST_CASE(RequestingInvalidDsAddressMustYieldInvalidDeDs)
{
  MapFEC fec(fecString);
  auto dsDetId = fec(DsElecId(1, 6, 0));
  BOOST_CHECK_EQUAL(dsDetId.has_value(), false);
}

BOOST_AUTO_TEST_CASE(GetValidDeDs)
{
  MapFEC fec(fecString);
  uint16_t solarId = 1;
  uint8_t elinkGroupId = 1;
  uint8_t elinkIndex = 2;
  auto dsDetId = fec(DsElecId(solarId, elinkGroupId, elinkIndex));
  BOOST_CHECK_EQUAL(dsDetId.has_value(), true);
  BOOST_CHECK_EQUAL(dsDetId->deId(), 919);
  BOOST_CHECK_EQUAL(dsDetId->dsId(), 1134);
}
