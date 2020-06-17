// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw FeeLinkId
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "MCHRawElecMap/FeeLinkId.h"
#include <fmt/format.h>

using namespace o2::mch::raw;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(feelinkid)

BOOST_AUTO_TEST_CASE(FeeLinkIdEncodeDecode)
{
  uint16_t feeId{0};
  uint8_t linkId{11};
  FeeLinkId cl(feeId, linkId);
  auto code = encode(cl);
  auto x = decodeFeeLinkId(code);
  BOOST_CHECK_EQUAL(code, encode(x));
}

BOOST_AUTO_TEST_CASE(FeeLinkIdCtorMustThrowIfLinkIdIsInvalid)
{
  uint16_t feeId{0};
  BOOST_CHECK_THROW(FeeLinkId a(feeId, 24), std::logic_error);
  BOOST_CHECK_NO_THROW(FeeLinkId a(feeId, 0));
  BOOST_CHECK_NO_THROW(FeeLinkId a(feeId, 11));
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
