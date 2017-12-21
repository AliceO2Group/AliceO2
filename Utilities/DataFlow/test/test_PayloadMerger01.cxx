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

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>
#include "DataFlow/PayloadMerger.h"
#include "DataFlow/FakeTimeframeBuilder.h"
#include "DataFlow/TimeframeParser.h"
#include "DataFlow/SubframeUtils.h"
#include "fairmq/FairMQTransportFactory.h"
#include "fairmq/FairMQParts.h"

using SubframeId = o2::dataflow::SubframeId;
using HeartbeatHeader = o2::header::HeartbeatHeader;
using HeartbeatTrailer = o2::header::HeartbeatTrailer;

SubframeId fakeAddition(o2::dataflow::PayloadMerger<SubframeId> &merger,
                  std::shared_ptr<FairMQTransportFactory> &transport,
                  int64_t orbit) {
  // Create a message
  //
  // We set orbit to be always the same and the actual contents to be 127
  static size_t dummyMessageSize = 1000;
  auto msg = transport->CreateMessage(dummyMessageSize);
  char *b = reinterpret_cast<char*>(msg->GetData()) + sizeof(HeartbeatHeader);
  for (size_t i = 0; i < (dummyMessageSize - sizeof(HeartbeatHeader)); ++i) {
    b[i] = orbit;
  }
  b[0] = 127;
  HeartbeatHeader *header = reinterpret_cast<HeartbeatHeader*>(msg->GetData());
  header->orbit = orbit;
  return merger.aggregate(msg);
}

BOOST_AUTO_TEST_CASE(PayloadMergerTest) {
  auto zmq = FairMQTransportFactory::CreateTransportFactory("zeromq");

  // Needs three subtimeframes to merge them
  auto checkIfComplete = [](SubframeId id, o2::dataflow::PayloadMerger<SubframeId>::MessageMap &m) -> bool {
    return m.count(id) >= 3;
  };

  // Id is given by the orbit, 2 orbits per timeframe
  auto makeId = [](std::unique_ptr<FairMQMessage> &msg) {
    auto header = reinterpret_cast<o2::header::HeartbeatHeader const*>(msg->GetData());
    return o2::dataflow::makeIdFromHeartbeatHeader(*header, 0, 2);
  };

  o2::dataflow::PayloadMerger<SubframeId> merger(makeId, checkIfComplete, o2::dataflow::extractDetectorPayloadStrip);
  char *finalBuf = new char[3000];
  size_t finalSize = 0;
  auto id = fakeAddition(merger, zmq, 1);
  finalSize = merger.finalise(&finalBuf, id);
  BOOST_CHECK(finalSize == 0); // Not enough parts, not merging yet.
  id = fakeAddition(merger, zmq, 1);
  finalSize = merger.finalise(&finalBuf, id);
  BOOST_CHECK(finalSize == 0); // Not enough parts, not merging yet.
  id = fakeAddition(merger, zmq, 2);
  finalSize = merger.finalise(&finalBuf, id);
  BOOST_CHECK(finalSize == 0); // Different ID, not merging yet.
  id = fakeAddition(merger, zmq, 1);
  finalSize = merger.finalise(&finalBuf, id);
  BOOST_CHECK(finalSize); // Now we merge!
  size_t partSize = (1000-sizeof(HeartbeatHeader) - sizeof(HeartbeatTrailer));
  BOOST_CHECK(finalSize == 3*partSize); // This should be the calculated size
  for (size_t i = 0; i < finalSize; ++i) {
    BOOST_CHECK(finalBuf[i] == ((i % partSize) == 0 ? 127 : 1));
  }
}
