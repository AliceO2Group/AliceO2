// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTRDDataFormats.cxx
/// \brief This task tests the data format structs
/// \author Sean Murray, murrays@cern.ch

#define BOOST_TEST_MODULE Test TRD_RawDataHeader
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"

namespace o2
{
namespace trd
{

/// \brief Test the data header struct sizes
//
/// check the bit manipulations
BOOST_AUTO_TEST_CASE(TRDRawDataHeaderSizes)
{
  //check the sizes of header structs due to packing
  BOOST_CHECK_EQUAL(sizeof(o2::trd::TrackletMCMData), 4);
  BOOST_CHECK_EQUAL(sizeof(o2::trd::TrackletHCHeader), 4);
  BOOST_CHECK_EQUAL(sizeof(o2::trd::TrackletMCMHeader), 4);
  BOOST_CHECK_EQUAL(sizeof(o2::trd::HalfCRUHeader), 64);
}

BOOST_AUTO_TEST_CASE(TRDRawDataHeaderInternals)
{
  o2::trd::TrackletMCMData tracklet;
  o2::trd::HalfCRUHeader halfcruheader;
  // changed as now nothing spans a 32 or 16 bit boundary
  halfcruheader.word0 = 0x22200; //bc is at 0x0000000fff00
  BOOST_CHECK_EQUAL(halfcruheader.BunchCrossing, 0x222);
  halfcruheader.word0 = 0x00000077; //headerversion is at 0x000000000
  BOOST_CHECK_EQUAL(halfcruheader.HeaderVersion, 119);
  //error flags
  halfcruheader.word12[0] = 0xa02;
  halfcruheader.word12[1] = 0xab0000;
  BOOST_CHECK_EQUAL(halfcruheader.errorflags[0].errorflag, 2);
  BOOST_CHECK_EQUAL(halfcruheader.errorflags[1].errorflag, 0xa);
  halfcruheader.word12[1] = 0x00ed000000000000; // should link 14 error flags.
  BOOST_CHECK_EQUAL(halfcruheader.errorflags[14].errorflag, 0xed);
  BOOST_CHECK_EQUAL(halfcruheader.errorflags[14].errorflag, o2::trd::getHalfCRULinkErrorFlag(halfcruheader, 14));
  //datasizes
  halfcruheader.word47[0] = 0xbdbd;
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[0].size, 0xbdbd);
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[0].size, o2::trd::getHalfCRULinkDataSize(halfcruheader, 0));
  halfcruheader.word47[1] = 0xabcd;
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[4].size, 0xabcd);
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[4].size, o2::trd::getHalfCRULinkDataSize(halfcruheader, 4));
  halfcruheader.word47[2] = 0xaaade127;
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[8].size, 0xe127);
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[8].size, o2::trd::getHalfCRULinkDataSize(halfcruheader, 8));
  halfcruheader.word47[3] = 0xefaadebc0000;
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[14].size, 0xefaa);
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[14].size, o2::trd::getHalfCRULinkDataSize(halfcruheader, 14));
  o2::trd::TrackletMCMHeader mcmrawdataheader;
  mcmrawdataheader.word = 0x78000000;
  BOOST_CHECK_EQUAL(mcmrawdataheader.padrow, 15);
  mcmrawdataheader.word = 0x06000000;
  BOOST_CHECK_EQUAL(mcmrawdataheader.col, 3);
  mcmrawdataheader.word = 0x01fe0000;
  BOOST_CHECK_EQUAL(mcmrawdataheader.pid2, 0xff); // 8 bits
  mcmrawdataheader.word = 0x01fe;
  BOOST_CHECK_EQUAL(mcmrawdataheader.pid0, 0xff); // 8 bits
  //check tracklet
  tracklet.word = 0xffe00000;
  BOOST_CHECK_EQUAL((uint32_t)tracklet.pos, 0x7ff);

  // This will get expanded, for now just check the current changes to charges.
  // build a tracklet package of ff00ff
  TrackletHCHeader hcheader;
  hcheader.word = 0;
  constructTrackletHCHeader(hcheader, 42, 42, 3); //only thing important is the format of 3
  // hcid=42, clock=42
  TrackletMCMHeader header;
  header.word = 0;
  header.onea = 1;
  header.oneb = 1;
  header.padrow = 2;
  header.col = 1;
  header.pid0 = 0xff;
  header.pid1 = 0x02;
  header.pid2 = 0xff;
  std::array<TrackletMCMData, 3> tracklets;
  tracklets[0].word = 0;
  tracklets[0].pos = 42;
  tracklets[0].slope = 24;
  tracklets[0].checkbit = 1;
  tracklets[0].pid = 0xefe;
  tracklets[1].word = 0;
  tracklets[2].word = 0;
  std::array<uint8_t, 3> charges;
  // auto invalidheader = getChargesFromRawHeaders(hcheader, &header, tracklets, charges, 0);
  // auto trackletcount = getNumberOfTrackletsFromHeader(&header);

  // BOOST_CHECK(trackletcount == 1);
  // BOOST_CHECK(invalidheader == 0);
  // std::cout << std::hex << " charges[0]:0x"<<(int)charges[0];
  // std::cout << std::hex << " charges[1]:0x"<<(int)charges[1];
  // std::cout << std::hex << " charges[2]:0x"<<(int)charges[2];
  // BOOST_CHECK((int)charges[2] == 0x02);
}
} // namespace trd
} // namespace o2
