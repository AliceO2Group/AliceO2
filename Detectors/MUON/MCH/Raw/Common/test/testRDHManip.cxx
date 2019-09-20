// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw RAWDataHeader
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHRawCommon/RDHManip.h"
#include "Headers/RAWDataHeader.h"

using namespace o2::mch::raw;

std::vector<uint32_t> testBuffer32()
{
  std::vector<uint32_t> buffer(16);
  int n{0};
  for (int i = 0; i < 16; i++) {
    buffer[i] = n | ((n + 1) << 8) | ((n + 2) << 16) | ((n + 3) << 24);
    n += 4;
  }
  return buffer;
}

std::vector<uint8_t> testBuffer8()
{
  std::vector<uint8_t> buffer(64);
  for (int i = 0; i < 64; i++) {
    buffer[i] = i;
  }
  return buffer;
}

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(rawdataheader)

o2::header::RAWDataHeaderV4 getTestRDH()
{
  o2::header::RAWDataHeaderV4 rdh;

  rdh.word0 = 0x0706050403020100;
  rdh.word1 = 0x0F0E0D0C0B0A0908;
  rdh.word2 = 0x1716151413121110;
  rdh.word3 = 0x1F1E1D1C1B1A1918;
  rdh.word4 = 0x2726252423222120;
  rdh.word5 = 0x2F2E2D2C2B2A2928;
  rdh.word6 = 0x3736353433323130;
  rdh.word7 = 0x3F3E3D3C3B3A3938;
  return rdh;
}

BOOST_AUTO_TEST_CASE(AppendRDH32)
{
  auto rdh = getTestRDH();
  std::vector<uint32_t> buffer;
  appendRDH<o2::header::RAWDataHeaderV4>(buffer, rdh);
  auto tb = testBuffer32();
  BOOST_CHECK_EQUAL(buffer.size(), tb.size());
  BOOST_CHECK(std::equal(begin(buffer), end(buffer), begin(tb)));
}

BOOST_AUTO_TEST_CASE(AppendRDH8)
{
  auto rdh = getTestRDH();
  std::vector<uint8_t> buffer;
  appendRDH<o2::header::RAWDataHeaderV4>(buffer, rdh);
  auto tb = testBuffer8();
  BOOST_CHECK_EQUAL(buffer.size(), tb.size());
  BOOST_CHECK(std::equal(begin(buffer), end(buffer), begin(tb)));
}

BOOST_AUTO_TEST_CASE(CreateRDHFromBuffer32)
{
  auto buffer = testBuffer32();
  auto rdh = createRDH<o2::header::RAWDataHeaderV4>(buffer);
  BOOST_CHECK_EQUAL(rdh.word0, 0x0706050403020100);
  BOOST_CHECK_EQUAL(rdh.word1, 0x0F0E0D0C0B0A0908);
  BOOST_CHECK_EQUAL(rdh.word2, 0x1716151413121110);
  BOOST_CHECK_EQUAL(rdh.word3, 0x1F1E1D1C1B1A1918);
  BOOST_CHECK_EQUAL(rdh.word4, 0x2726252423222120);
  BOOST_CHECK_EQUAL(rdh.word5, 0x2F2E2D2C2B2A2928);
  BOOST_CHECK_EQUAL(rdh.word6, 0x3736353433323130);
  BOOST_CHECK_EQUAL(rdh.word7, 0x3F3E3D3C3B3A3938);
}

BOOST_AUTO_TEST_CASE(CreateRDHFromBuffer8)
{
  auto buffer = testBuffer8();
  auto rdh = createRDH<o2::header::RAWDataHeaderV4>(buffer);
  BOOST_CHECK_EQUAL(rdh.word0, 0x0706050403020100);
  BOOST_CHECK_EQUAL(rdh.word1, 0x0F0E0D0C0B0A0908);
  BOOST_CHECK_EQUAL(rdh.word2, 0x1716151413121110);
  BOOST_CHECK_EQUAL(rdh.word3, 0x1F1E1D1C1B1A1918);
  BOOST_CHECK_EQUAL(rdh.word4, 0x2726252423222120);
  BOOST_CHECK_EQUAL(rdh.word5, 0x2F2E2D2C2B2A2928);
  BOOST_CHECK_EQUAL(rdh.word6, 0x3736353433323130);
  BOOST_CHECK_EQUAL(rdh.word7, 0x3F3E3D3C3B3A3938);
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
