// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataModelViews.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include <fairmq/TransportFactory.h>
#include <cstring>
#include <catch_amalgamated.hpp>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

namespace
{
// Build a header message containing a DataHeader with the given split-payload fields.
fair::mq::MessagePtr makeHeader(fair::mq::TransportFactory& transport,
                                uint32_t splitPayloadParts, uint32_t splitPayloadIndex)
{
  DataHeader dh;
  dh.dataDescription = "TEST";
  dh.dataOrigin = "TST";
  dh.subSpecification = 0;
  dh.splitPayloadParts = splitPayloadParts;
  dh.splitPayloadIndex = splitPayloadIndex;
  DataProcessingHeader dph{0, 1};
  Stack stack{dh, dph};
  auto msg = transport.CreateMessage(stack.size());
  memcpy(msg->GetData(), stack.data(), stack.size());
  return msg;
}

fair::mq::MessagePtr makePayload(fair::mq::TransportFactory& transport)
{
  return transport.CreateMessage(4);
}
} // namespace

// ---------------------------------------------------------------------------
// Single [header, payload] pair (splitPayloadParts == 0)
// ---------------------------------------------------------------------------
TEST_CASE("SinglePair")
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");

  std::vector<fair::mq::MessagePtr> msgs;
  msgs.emplace_back(makeHeader(*transport, 0, 0));
  msgs.emplace_back(makePayload(*transport));

  REQUIRE((msgs | count_parts{}) == 1);
  REQUIRE((msgs | count_payloads{}) == 1);
  REQUIRE((msgs | get_num_payloads{0}) == 1);

  auto idx = msgs | get_pair{0};
  REQUIRE(idx.headerIdx == 0);
  REQUIRE(idx.payloadIdx == 1);

  // Advancing past the only pair goes out of range.
  auto next = msgs | get_next_pair{idx};
  REQUIRE(next.headerIdx >= msgs.size());
}

// ---------------------------------------------------------------------------
// Old-style multipart: N [header, payload] pairs, each with splitPayloadParts=N
// and splitPayloadIndex running 0..N-1 (0-indexed).
// The new-style sentinel is splitPayloadIndex == splitPayloadParts, which is
// never true for old-style (max index is N-1 < N).
// Layout: [h0,p0, h1,p1, h2,p2]
// count_parts returns N because each [h,p] pair is a separate logical part.
// ---------------------------------------------------------------------------
TEST_CASE("OldStyleMultipart")
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  constexpr uint32_t N = 3;

  std::vector<fair::mq::MessagePtr> msgs;
  for (uint32_t i = 0; i < N; ++i) {
    msgs.emplace_back(makeHeader(*transport, N, i)); // 0-indexed
    msgs.emplace_back(makePayload(*transport));
  }

  REQUIRE((msgs | count_parts{}) == N);    // N separate logical parts
  REQUIRE((msgs | count_payloads{}) == N); // one payload each
  for (uint32_t i = 0; i < N; ++i) {
    REQUIRE((msgs | get_num_payloads{i}) == 1);
  }

  // get_pair reaches each sub-part directly.
  for (uint32_t i = 0; i < N; ++i) {
    auto idx = msgs | get_pair{i};
    REQUIRE(idx.headerIdx == 2 * i);
    REQUIRE(idx.payloadIdx == 2 * i + 1);
  }

  // get_next_pair advances sequentially through all pairs.
  DataRefIndices idx{0, 1};
  for (uint32_t i = 1; i < N; ++i) {
    idx = msgs | get_next_pair{idx};
    REQUIRE(idx.headerIdx == 2 * i);
    REQUIRE(idx.payloadIdx == 2 * i + 1);
  }
  // One more step goes out of range.
  idx = msgs | get_next_pair{idx};
  REQUIRE(idx.headerIdx >= msgs.size());
}

// ---------------------------------------------------------------------------
// New-style multipart: one header followed by N contiguous payloads.
// splitPayloadParts == splitPayloadIndex == N  (the sentinel for new style).
// Layout: [h, p0, p1, p2]
// ---------------------------------------------------------------------------
TEST_CASE("NewStyleMultiPayload")
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  constexpr uint32_t N = 3;

  std::vector<fair::mq::MessagePtr> msgs;
  msgs.emplace_back(makeHeader(*transport, N, N));
  for (uint32_t i = 0; i < N; ++i) {
    msgs.emplace_back(makePayload(*transport));
  }

  REQUIRE((msgs | count_parts{}) == 1);
  REQUIRE((msgs | count_payloads{}) == N);
  REQUIRE((msgs | get_num_payloads{0}) == N); // all payloads belong to part 0

  // get_pair returns the same header for every sub-part, advancing payloadIdx.
  for (uint32_t i = 0; i < N; ++i) {
    auto idx = msgs | get_pair{i};
    REQUIRE(idx.headerIdx == 0);
    REQUIRE(idx.payloadIdx == 1 + i);
  }

  // get_next_pair advances payloadIdx within the block, then moves to next block.
  DataRefIndices idx{0, 1};
  for (uint32_t i = 1; i < N; ++i) {
    idx = msgs | get_next_pair{idx};
    REQUIRE(idx.headerIdx == 0);
    REQUIRE(idx.payloadIdx == 1 + i);
  }
  // One more step exits the block.
  idx = msgs | get_next_pair{idx};
  REQUIRE(idx.headerIdx >= msgs.size());
}

// ---------------------------------------------------------------------------
// Mixed message set: two routes, one single-pair and one new-style block.
// Layout: [h0, p0,  h1, p1_0, p1_1]
// ---------------------------------------------------------------------------
TEST_CASE("MixedLayout")
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");

  std::vector<fair::mq::MessagePtr> msgs;
  // Route 0: single pair
  msgs.emplace_back(makeHeader(*transport, 0, 0));
  msgs.emplace_back(makePayload(*transport));
  // Route 1: new-style 2-payload block
  msgs.emplace_back(makeHeader(*transport, 2, 2));
  msgs.emplace_back(makePayload(*transport));
  msgs.emplace_back(makePayload(*transport));

  REQUIRE((msgs | count_parts{}) == 2);
  REQUIRE((msgs | count_payloads{}) == 3);

  // get_pair across routes
  auto idx0 = msgs | get_pair{0};
  REQUIRE(idx0.headerIdx == 0);
  REQUIRE(idx0.payloadIdx == 1);

  auto idx1 = msgs | get_pair{1};
  REQUIRE(idx1.headerIdx == 2);
  REQUIRE(idx1.payloadIdx == 3);

  auto idx2 = msgs | get_pair{2};
  REQUIRE(idx2.headerIdx == 2);
  REQUIRE(idx2.payloadIdx == 4);

  // get_next_pair traversal from the first element
  DataRefIndices idx{0, 1};
  idx = msgs | get_next_pair{idx};
  REQUIRE(idx.headerIdx == 2);
  REQUIRE(idx.payloadIdx == 3);
  idx = msgs | get_next_pair{idx};
  REQUIRE(idx.headerIdx == 2);
  REQUIRE(idx.payloadIdx == 4);
  idx = msgs | get_next_pair{idx};
  REQUIRE(idx.headerIdx >= msgs.size());
}
