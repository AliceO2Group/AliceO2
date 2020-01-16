// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw DsElecId
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "MCHRawElecMap/DsElecId.h"
#include <fmt/format.h>

using namespace o2::mch::raw;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(dselecid)

BOOST_AUTO_TEST_CASE(DsElecId)
{
  o2::mch::raw::DsElecId eid(448, 6, 2);
  BOOST_CHECK_EQUAL(asString(eid), "S448-J6-DS2");
  auto code = encode(eid);
  auto x = decodeDsElecId(code);
  BOOST_CHECK_EQUAL(code, encode(x.value()));
}

BOOST_AUTO_TEST_CASE(DsElecIdValidString)
{
  o2::mch::raw::DsElecId eid(448, 6, 2);
  auto x = decodeDsElecId("S448-J6-DS2");
  BOOST_CHECK_EQUAL(x.has_value(), true);
  BOOST_CHECK_EQUAL(encode(eid), encode(x.value()));
}

BOOST_AUTO_TEST_CASE(DecodeDsElecIdForTooShortString)
{
  BOOST_CHECK_EQUAL(decodeDsElecId("S448").has_value(), false);
  BOOST_CHECK_EQUAL(decodeDsElecId("S448-XX").has_value(), false);
  BOOST_CHECK_EQUAL(decodeDsElecId("S448-J3-DS2-CH-0").has_value(), true);
}

BOOST_AUTO_TEST_CASE(DecodeDsElecIdForInvalidSolar)
{
  BOOST_CHECK_EQUAL(decodeDsElecId("s448-J6-DS2").has_value(), false);
  BOOST_CHECK_EQUAL(decodeDsElecId("448-J6-DS2").has_value(), false);
  BOOST_CHECK_EQUAL(decodeDsElecId("X448-J6-DS2").has_value(), false);
}

BOOST_AUTO_TEST_CASE(DecodeDsElecIdForInvalidGroup)
{
  BOOST_CHECK_EQUAL(decodeDsElecId("S448-6-DS2").has_value(), false);
  BOOST_CHECK_EQUAL(decodeDsElecId("S448-j6-DS2").has_value(), false);
  BOOST_CHECK_EQUAL(decodeDsElecId("S448-X6-DS2").has_value(), false);
}

BOOST_AUTO_TEST_CASE(DecodeDsElecIdForInvalidDS)
{
  BOOST_CHECK_EQUAL(decodeDsElecId("S448-J6-DS").has_value(), false);
  BOOST_CHECK_EQUAL(decodeDsElecId("S448-J6-Ds2").has_value(), false);
  BOOST_CHECK_EQUAL(decodeDsElecId("S448-J6-D2").has_value(), false);
}

BOOST_AUTO_TEST_CASE(DecodeChannelIdForInvalidString)
{
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2").has_value(), false);
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2-CH-0").has_value(), false);
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2-cH-0").has_value(), false);
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2-XX-0").has_value(), false);
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2-XX0").has_value(), false);
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2-ch0").has_value(), false);
}

BOOST_AUTO_TEST_CASE(DecodeChannelIdForValidString)
{
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2-CH0").has_value(), true);
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2-CH63").has_value(), true);
}

BOOST_AUTO_TEST_CASE(DecodeChannelIdForInvalidChannel)
{
  BOOST_CHECK_EQUAL(decodeChannelId("S448-J6-DS2-CH64").has_value(), false);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
