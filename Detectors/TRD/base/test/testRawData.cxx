// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  o2::trd::TrackletHCHeader halfchamberheader;
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
  BOOST_CHECK_EQUAL(halfcruheader.errorflags[14].errorflag, o2::trd::getlinkerrorflag(halfcruheader, 14));
  //datasizes
  halfcruheader.word47[0] = 0xbdbd;
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[0].size, 0xbdbd);
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[0].size, o2::trd::getlinkdatasize(halfcruheader, 0));
  halfcruheader.word47[1] = 0xabcd;
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[4].size, 0xabcd);
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[4].size, o2::trd::getlinkdatasize(halfcruheader, 4));
  halfcruheader.word47[2] = 0xaaade127;
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[8].size, 0xe127);
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[8].size, o2::trd::getlinkdatasize(halfcruheader, 8));
  halfcruheader.word47[3] = 0xefaadebc0000;
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[14].size, 0xefaa);
  BOOST_CHECK_EQUAL(halfcruheader.datasizes[14].size, o2::trd::getlinkdatasize(halfcruheader, 14));
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
  tracklet.word = 0xffc00000;
  BOOST_CHECK_EQUAL((uint32_t)tracklet.pos, 0x3ff);
}
} // namespace trd
} // namespace o2
