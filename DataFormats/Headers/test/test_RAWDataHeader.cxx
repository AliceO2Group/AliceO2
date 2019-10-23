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

BOOST_AUTO_TEST_CASE(test_rdh_v4)
{
  using RDH = o2::header::RAWDataHeader;
  static_assert(sizeof(RDH) == 64, "RAWDataHeader v5 is supposed to be 512 bit");

  // check the defaults
  RDH defaultRDH;
  BOOST_CHECK(defaultRDH.version == 4);
  BOOST_CHECK(defaultRDH.blockLength == 0);
  BOOST_CHECK(defaultRDH.feeId == 0xffff);
  //BOOST_CHECK(defaultRDH.linkId == 0xff);
  BOOST_CHECK(defaultRDH.headerSize == 64); // header size in 64 bytes = 8*64 bit words

  using byte = unsigned char;
  std::array<byte, sizeof(RDH)> buffer;
  buffer.fill(0);

  buffer[0] = 0x3;  // set version 3
  buffer[1] = 0x08; // header size 8 64 bit words
  buffer[2] = 0x6;
  buffer[3] = 0x1;
  buffer[4] = 0xad;
  buffer[5] = 0xde;
  buffer[6] = 0;
  buffer[7] = 0;

  // do not set anything in words 1,2, and 3

  buffer[32] = 0xdc;
  buffer[33] = 0x0e;
  buffer[34] = 0x12;
  buffer[35] = 0x0a;
  buffer[36] = 0x23;
  buffer[37] = 0x71;
  buffer[38] = 0x98;
  buffer[39] = 0xba;

  // do not set anything in word 5

  buffer[48] = 0xad;
  buffer[49] = 0xde;
  buffer[50] = 0;
  buffer[51] = 0;
  buffer[52] = 0x00;
  buffer[53] = 0x0f;
  buffer[54] = 0xd0;
  buffer[55] = 0x00;

  // do not set anything in word 7

  auto rdh = reinterpret_cast<RDH*>(buffer.data());

  // if the tests fail, we are probbaly on a big endian architecture or
  // there are alignment issues
  BOOST_CHECK(rdh->version == 3);
  BOOST_CHECK(rdh->blockLength == 262);
  BOOST_CHECK(rdh->feeId == 0xdead);
  //BOOST_CHECK(rdh->linkId == 0xaf);
  BOOST_CHECK(rdh->headerSize == 8);
  BOOST_CHECK(rdh->priority == 0);
  BOOST_CHECK(rdh->zero0 == 0);
  BOOST_CHECK(rdh->word1 == 0);
  BOOST_CHECK(rdh->word2 == 0);
  BOOST_CHECK(rdh->word3 == 0);
  BOOST_CHECK(rdh->triggerOrbit == 0);
  BOOST_CHECK(rdh->heartbeatOrbit == 0);
  BOOST_CHECK(rdh->triggerType == 0xba987123);
  BOOST_CHECK(rdh->triggerBC == 0xedc);
  BOOST_CHECK(rdh->heartbeatBC == 0xa12);
  BOOST_CHECK(rdh->zero41 == 0);
  BOOST_CHECK(rdh->zero42 == 0);
  BOOST_CHECK(rdh->word5 == 0);
  BOOST_CHECK(rdh->pageCnt == 0xd00f);
  BOOST_CHECK(rdh->stop == 0);
  BOOST_CHECK(rdh->detectorField == 0xdead);
  BOOST_CHECK(rdh->par == 0);
  BOOST_CHECK(rdh->zero6 == 0);
  BOOST_CHECK(rdh->word7 == 0);
}

BOOST_AUTO_TEST_CASE(test_rdh_v5)
{
  using RDH = o2::header::RAWDataHeaderV5;
  static_assert(sizeof(RDH) == 64, "RAWDataHeader v5 is supposed to be 512 bit");

  // check the defaults
  RDH defaultRDH;
  BOOST_CHECK(defaultRDH.version == 5);
  BOOST_CHECK(defaultRDH.feeId == 0xffff);
  BOOST_CHECK(defaultRDH.headerSize == 64); // header size in 64 bytes = 8*64 bit words
}
