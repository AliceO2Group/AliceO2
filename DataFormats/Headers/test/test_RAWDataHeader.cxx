// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test RAWDataHeader
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <array>
#include "Headers/RAWDataHeader.h"

using RDH = o2::header::RAWDataHeader;

BOOST_AUTO_TEST_CASE(test_rdh)
{
  static_assert(sizeof(RDH) == 32, "the RAWDataHeader is supposed to be 256 bit");

  // check the defaults
  RDH defaultRDH;
  BOOST_CHECK(defaultRDH.version == 2);
  BOOST_CHECK(defaultRDH.blockLength == 0);
  BOOST_CHECK(defaultRDH.feeId == 0xffff);
  BOOST_CHECK(defaultRDH.linkId == 0xff);
  BOOST_CHECK(defaultRDH.headerSize == 4); // header size in 64 bit words

  using byte = unsigned char;
  std::array<byte, sizeof(RDH)> buffer;
  buffer.fill(0);

  buffer[0] = 0x2; // set version 2
  buffer[1] = 0x6;
  buffer[2] = 0x1;
  buffer[3] = 0xad;
  buffer[4] = 0xde;
  buffer[5] = 0xaf;
  buffer[6] = 0x04;
  buffer[7] = 0;

  buffer[16] = 0x23;
  buffer[17] = 0x71;
  buffer[18] = 0x98;
  buffer[19] = 0xba;
  buffer[20] = 0xdc;
  buffer[21] = 0x3e;
  buffer[22] = 0x12;
  buffer[23] = 0;

  buffer[24] = 0x0f;
  buffer[25] = 0xd0;
  buffer[26] = 0;
  buffer[27] = 0xad;
  buffer[28] = 0xde;
  buffer[29] = 0;

  auto rdh = reinterpret_cast<RDH*>(buffer.data());

  // if the tests fail, we are probbaly on a big endian architecture or
  // there are alignment issues
  BOOST_CHECK(rdh->version == 2);
  BOOST_CHECK(rdh->blockLength == 262);
  BOOST_CHECK(rdh->feeId == 0xdead);
  BOOST_CHECK(rdh->linkId == 0xaf);
  BOOST_CHECK(rdh->headerSize == 4);
  BOOST_CHECK(rdh->zero0 == 0);
  BOOST_CHECK(rdh->triggerOrbit == 0);
  BOOST_CHECK(rdh->heartbeatOrbit == 0);
  BOOST_CHECK(rdh->triggerBC == 0x123);
  BOOST_CHECK(rdh->triggerType == 0xedcba987);
  BOOST_CHECK(rdh->heartbeatBC == 0x123);
  BOOST_CHECK(rdh->zero2 == 0);
  BOOST_CHECK(rdh->pageCnt == 0xd00f);
  BOOST_CHECK(rdh->stop == 0);
  BOOST_CHECK(rdh->detectorField == 0xdead);
  BOOST_CHECK(rdh->par == 0);
  BOOST_CHECK(rdh->zero3 == 0);
}
