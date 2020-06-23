// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw ImplHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MoveBuffer.h"

using namespace o2::mch::raw;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(movebuffer)

BOOST_AUTO_TEST_CASE(MoveBuffer)
{
  std::vector<uint64_t> b64;
  b64.emplace_back(0x0706050403020100);
  b64.emplace_back(0x0F0E0D0C0B0A0908);
  std::vector<std::byte> b8;
  impl::moveBuffer(b64, b8);
  std::vector<std::byte> expected;
  for (uint8_t i = 0; i < 16; i++) {
    expected.emplace_back(std::byte{i});
  }
  BOOST_CHECK_EQUAL(b8.size(), expected.size());
  BOOST_CHECK(b8 == expected);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
