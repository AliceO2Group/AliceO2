// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw CruLinkId
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "MCHRawElecMap/CruLinkId.h"
#include <fmt/format.h>

using namespace o2::mch::raw;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(crulinkid)

BOOST_AUTO_TEST_CASE(CruLinkIdEncodeDecode)
{
  uint16_t cruId{0};
  uint8_t linkId{23};
  uint16_t deId{0};
  CruLinkId cl(cruId, linkId, deId);
  auto code = encode(cl);
  auto x = decodeCruLinkId(code);
  BOOST_CHECK_EQUAL(code, encode(x));
}

BOOST_AUTO_TEST_CASE(CruLinkIdCtorMustThrowIfLinkIdIsInvalid)
{
  uint16_t cruId{0};
  uint16_t solarId{0};
  BOOST_CHECK_THROW(CruLinkId a(cruId, 24, solarId), std::logic_error);
  BOOST_CHECK_NO_THROW(CruLinkId a(cruId, 0, solarId));
  BOOST_CHECK_NO_THROW(CruLinkId a(cruId, 23, solarId));
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
