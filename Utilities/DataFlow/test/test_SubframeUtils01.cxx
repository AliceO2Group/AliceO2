// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Utilities DataFlowTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "DataFlow/SubframeUtils.h"
#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE(SubframeUtils01)
{
  o2::dataflow::SubframeId a;
  a.timeframeId = 0;
  a.socketId = 1;
  o2::dataflow::SubframeId b;
  b.timeframeId = 1;
  b.socketId = 0;
  BOOST_CHECK(a < b);
  char* buf = new char[1000];
  memset(buf, 126, 1000);
  for (size_t i = sizeof(o2::header::HeartbeatHeader); i < 1000 - sizeof(o2::header::HeartbeatHeader); ++i) {
    buf[i] = 0;
  }
  BOOST_CHECK(buf[0] == 126);
  BOOST_CHECK(buf[sizeof(o2::header::HeartbeatHeader)] == 0);
  BOOST_CHECK(buf[sizeof(o2::header::HeartbeatHeader) - 1] == 126);
  char* realPayload = nullptr;
  size_t realSize = o2::dataflow::extractDetectorPayloadStrip(&realPayload, buf, 1000);
  BOOST_CHECK(realPayload != nullptr);
  BOOST_CHECK(realSize == 1000 - sizeof(o2::header::HeartbeatHeader) - sizeof(o2::header::HeartbeatTrailer));
  BOOST_CHECK(realPayload == buf + sizeof(o2::header::HeartbeatHeader));
  BOOST_CHECK(realPayload[0] == 0);

  o2::header::HeartbeatHeader header1;
  header1.orbit = 0;
  o2::header::HeartbeatHeader header2;
  header2.orbit = 255;
  o2::header::HeartbeatHeader header3;
  header3.orbit = 256;

  auto id1 = o2::dataflow::makeIdFromHeartbeatHeader(header1, 1, 256);
  auto id2 = o2::dataflow::makeIdFromHeartbeatHeader(header2, 1, 256);
  auto id3 = o2::dataflow::makeIdFromHeartbeatHeader(header3, 1, 256);
  BOOST_CHECK(!(id1 < id2)); // Maybe we should provide an == operator
  BOOST_CHECK(!(id2 < id1));
  BOOST_CHECK(id1 < id3);
  BOOST_CHECK(id2 < id3);
  BOOST_CHECK(id1.timeframeId == 0);
  BOOST_CHECK(id3.timeframeId == 1);
}
