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
  BOOST_CHECK_EQUAL(sizeof(o2::trd::TrapRawTracklet), 4);
  BOOST_CHECK_EQUAL(sizeof(o2::trd::HalfChamberHeader), 4);
  BOOST_CHECK_EQUAL(sizeof(o2::trd::MCMRawDataHeader), 4);
  BOOST_CHECK_EQUAL(sizeof(o2::trd::HalfCRUHeader), 64);
}

BOOST_AUTO_TEST_CASE(TRDRawDataHeaderInternals)
{
  o2::trd::TrapRawTracklet tracklet;
  o2::trd::HalfChamberHeader halfchamberheader;
  o2::trd::HalfCRUHeader halfcruheader;
  halfcruheader.word02[0] = 0x102;
  BOOST_CHECK_EQUAL(halfcruheader.linksA[0].errorflags, 2);
  BOOST_CHECK_EQUAL(halfcruheader.linksA[0].size, 1);
  BOOST_CHECK_EQUAL(halfcruheader.linksA[0].errorflags, o2::trd::getlinkerrorflag(halfcruheader, 0));
  BOOST_CHECK_EQUAL(halfcruheader.linksA[0].size, o2::trd::getlinkdatasize(halfcruheader, 0));
  //ab is size
  halfcruheader.word02[2] = 0xffffaa0000000000;
  BOOST_CHECK_EQUAL(halfcruheader.linksA[7].size, 0xffff);
  BOOST_CHECK_EQUAL(halfcruheader.linksA[7].errorflags, 0xaa);
  BOOST_CHECK_EQUAL(halfcruheader.linksA[7].errorflags, o2::trd::getlinkerrorflag(halfcruheader, 7));
  BOOST_CHECK_EQUAL(halfcruheader.linksA[7].size, o2::trd::getlinkdatasize(halfcruheader, 7));
  halfcruheader.word3 = (0x222LL) << 40; //bc is at 0x000ffff000000000
  BOOST_CHECK_EQUAL(halfcruheader.BunchCrossing, 0x222);
  halfcruheader.word3 = 0x7700000000; //headerversion is at 0x00ff0000000
  BOOST_CHECK_EQUAL(halfcruheader.HeaderVersion, 119);
  halfcruheader.word57[0] = 0x345869faba;
  halfcruheader.word57[2] = 0xaaade1277;
  //linkb[0] zero is undefined.
  BOOST_CHECK_EQUAL(halfcruheader.linksB[1].errorflags, 0x58);
  BOOST_CHECK_EQUAL(halfcruheader.linksB[1].size, 0x34);
  BOOST_CHECK_EQUAL(halfcruheader.linksB[1].errorflags, o2::trd::getlinkerrorflag(halfcruheader, 8));
  BOOST_CHECK_EQUAL(halfcruheader.linksB[1].size, o2::trd::getlinkdatasize(halfcruheader, 8));
  //ab is size
  BOOST_CHECK_EQUAL(halfcruheader.linksB[5].size, 0x1277);
  BOOST_CHECK_EQUAL(halfcruheader.linksB[6].size, 0xaaa);
  BOOST_CHECK_EQUAL(halfcruheader.linksB[6].errorflags, 0xde);
  BOOST_CHECK_EQUAL(halfcruheader.linksB[6].errorflags, o2::trd::getlinkerrorflag(halfcruheader, 13));
  BOOST_CHECK_EQUAL(halfcruheader.linksB[6].size, o2::trd::getlinkdatasize(halfcruheader, 13));
  //now test the boundary crossing of the int64_t
  halfcruheader.word02[0] = (uint64_t)0x1000000000000000;
  halfcruheader.word02[1] = (uint64_t)0xa00b000c000d0ebf;
  BOOST_CHECK_EQUAL(halfcruheader.linksA[2].size, 0xbf10); // check a size that spans a 64bit word.
  o2::trd::MCMRawDataHeader mcmrawdataheader;
  mcmrawdataheader.word = 0x78000000;
  BOOST_CHECK_EQUAL(mcmrawdataheader.padrow, 15);
  mcmrawdataheader.word = 0x06000000;
  BOOST_CHECK_EQUAL(mcmrawdataheader.col, 3);
  mcmrawdataheader.word = 0x01fe0000;
  BOOST_CHECK_EQUAL(mcmrawdataheader.pid2, 0xff); // 8 bits  // gave up seperating pid, so its a flat 24 bit needs to be masked 0xfff 0xfff000 and 0xfff000000
  mcmrawdataheader.word = 0x01fe;
  BOOST_CHECK_EQUAL(mcmrawdataheader.pid0, 0xff); // 8 bits  // gave up seperating pid, so its a flat 24 bit needs to be masked 0xfff 0xfff000 and 0xfff000000
  //check tracklet
  tracklet.word = 0xffc00000;
  BOOST_CHECK_EQUAL((uint32_t)tracklet.pos, 0x3ff);
}
} // namespace trd
} // namespace o2
