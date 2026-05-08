// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <fairmq/Message.h>
#include <fairmq/TransportFactory.h>
#include "Framework/DataModelViews.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/PartRef.h"
#include "Headers/Stack.h"
#include "Headers/DataHeader.h"
#include "MemoryResources/MemoryResources.h"
#include <catch_amalgamated.hpp>

using namespace o2::framework;

TEST_CASE("MessageSet")
{
  std::vector<fair::mq::MessagePtr> messages;
  o2::header::DataHeader dh{};
  dh.splitPayloadParts = 0;
  dh.splitPayloadIndex = 0;
  o2::framework::DataProcessingHeader dph{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::MessagePtr header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  std::vector<fair::mq::MessagePtr> ptrs;
  ptrs.emplace_back(std::move(header));
  ptrs.emplace_back(std::move(msg2));
  for (size_t i = 0; i < 2; ++i) {
    messages.emplace_back(std::move(ptrs[i]));
  }

  REQUIRE(messages.size() == 2);
  REQUIRE((messages | count_payloads{}) == 1);
  REQUIRE((messages | get_dataref_indices{0, 0}).headerIdx == 0);
  REQUIRE((messages | get_dataref_indices{0, 0}).payloadIdx == 1);
  REQUIRE((messages | get_pair{0}).headerIdx == 0);
  REQUIRE((messages | get_pair{0}).payloadIdx == 1);
  CHECK_THROWS((messages | get_pair{1}));
  REQUIRE((messages | get_num_payloads{0}) == 1);
  REQUIRE((messages | count_parts{}) == 1);
}

TEST_CASE("MessageSetWithFunction")
{
  std::vector<fair::mq::MessagePtr> ptrs;
  o2::header::DataHeader dh{};
  dh.splitPayloadParts = 0;
  dh.splitPayloadIndex = 0;
  o2::framework::DataProcessingHeader dph{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::MessagePtr header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  ptrs.emplace_back(std::move(header));
  ptrs.emplace_back(std::move(msg2));
  std::vector<fair::mq::MessagePtr> messages;
  for (size_t i = 0; i < 2; ++i) {
    messages.emplace_back(std::move(ptrs[i]));
  }

  REQUIRE(messages.size() == 2);
  REQUIRE((messages | count_payloads{}) == 1);
  REQUIRE((messages | get_dataref_indices{0, 0}).headerIdx == 0);
  REQUIRE((messages | get_dataref_indices{0, 0}).payloadIdx == 1);
  REQUIRE((messages | get_pair{0}).headerIdx == 0);
  REQUIRE((messages | get_pair{0}).payloadIdx == 1);
  CHECK_THROWS((messages | get_pair{1}));
  REQUIRE((messages | get_num_payloads{0}) == 1);
  REQUIRE((messages | count_parts{}) == 1);
}

TEST_CASE("MessageSetWithMultipart")
{
  std::vector<fair::mq::MessagePtr> ptrs;
  o2::header::DataHeader dh{};
  dh.splitPayloadParts = 2;
  dh.splitPayloadIndex = 2;
  o2::framework::DataProcessingHeader dph{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::MessagePtr header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  std::unique_ptr<fair::mq::Message> msg3(nullptr);
  ptrs.emplace_back(std::move(header));
  ptrs.emplace_back(std::move(msg2));
  ptrs.emplace_back(std::move(msg3));
  std::vector<fair::mq::MessagePtr> messages;
  for (size_t i = 0; i < 3; ++i) {
    messages.emplace_back(std::move(ptrs[i]));
  }

  REQUIRE(messages.size() == 3);
  REQUIRE((messages | count_payloads{}) == 2);
  REQUIRE((messages | get_dataref_indices{0, 0}).headerIdx == 0);
  REQUIRE((messages | get_dataref_indices{0, 0}).payloadIdx == 1);
  REQUIRE((messages | get_dataref_indices{0, 1}).headerIdx == 0);
  REQUIRE((messages | get_dataref_indices{0, 1}).payloadIdx == 2);
  REQUIRE((messages | get_pair{0}).headerIdx == 0);
  REQUIRE((messages | get_pair{0}).payloadIdx == 1);
  REQUIRE((messages | get_pair{1}).headerIdx == 0);
  REQUIRE((messages | get_pair{1}).payloadIdx == 2);
  CHECK_THROWS((messages | get_pair{2}));
  REQUIRE((messages | get_num_payloads{0}) == 2);
  REQUIRE((messages | count_parts{}) == 1);
}

TEST_CASE("MessageSetAddPartRef")
{
  std::unique_ptr<fair::mq::Message> msg(nullptr);
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  PartRef ref{std::move(msg), std::move(msg2)};
  std::vector<fair::mq::MessagePtr> messages;
  messages.emplace_back(std::move(ref.header));
  messages.emplace_back(std::move(ref.payload));

  REQUIRE(messages.size() == 2);
}

TEST_CASE("MessageSetAddMultiple")
{
  o2::header::DataHeader dh1{};
  dh1.splitPayloadParts = 0;
  dh1.splitPayloadIndex = 0;
  o2::header::DataHeader dh2{};
  dh2.splitPayloadParts = 1;
  dh2.splitPayloadIndex = 0;
  o2::header::DataHeader dh3{};
  dh3.splitPayloadParts = 2;
  dh3.splitPayloadIndex = 2;
  o2::framework::DataProcessingHeader dph{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::MessagePtr header1 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh1, dph});
  fair::mq::MessagePtr header2 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh2, dph});
  fair::mq::MessagePtr header3 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh3, dph});

  std::vector<fair::mq::MessagePtr> messages;
  // part 0: dh1 (splitPayloadParts=0) — standard pair
  messages.emplace_back(std::move(header1));
  messages.emplace_back(std::unique_ptr<fair::mq::Message>(nullptr));
  // part 1: dh2 (splitPayloadParts=1) — traditional split, one pair
  messages.emplace_back(std::move(header2));
  messages.emplace_back(std::unique_ptr<fair::mq::Message>(nullptr));
  // part 2: dh3 (splitPayloadParts=2, splitPayloadIndex=2) — multi-payload, two payloads
  messages.emplace_back(std::move(header3));
  messages.emplace_back(std::unique_ptr<fair::mq::Message>(nullptr));
  messages.emplace_back(std::unique_ptr<fair::mq::Message>(nullptr));

  REQUIRE(messages.size() == 7);

  REQUIRE((messages | count_payloads{}) == 4);
  REQUIRE((messages | get_dataref_indices{0, 0}).headerIdx == 0);
  REQUIRE((messages | get_dataref_indices{0, 0}).payloadIdx == 1);
  REQUIRE((messages | get_dataref_indices{1, 0}).headerIdx == 2);
  REQUIRE((messages | get_dataref_indices{1, 0}).payloadIdx == 3);
  REQUIRE((messages | get_dataref_indices{2, 0}).headerIdx == 4);
  REQUIRE((messages | get_dataref_indices{2, 0}).payloadIdx == 5);
  REQUIRE((messages | get_dataref_indices{2, 1}).headerIdx == 4);
  REQUIRE((messages | get_dataref_indices{2, 1}).payloadIdx == 6);
  REQUIRE((messages | get_pair{0}).headerIdx == 0);
  REQUIRE((messages | get_pair{0}).payloadIdx == 1);
  REQUIRE((messages | get_pair{1}).headerIdx == 2);
  REQUIRE((messages | get_pair{1}).payloadIdx == 3);
  REQUIRE((messages | get_pair{2}).headerIdx == 4);
  REQUIRE((messages | get_pair{2}).payloadIdx == 5);
  REQUIRE((messages | get_pair{3}).headerIdx == 4);
  REQUIRE((messages | get_pair{3}).payloadIdx == 6);
  REQUIRE((messages | get_num_payloads{0}) == 1);
  REQUIRE((messages | get_num_payloads{1}) == 1);
  REQUIRE((messages | get_num_payloads{2}) == 2);
  REQUIRE((messages | count_parts{}) == 3);
  REQUIRE((messages | count_payloads{}) == 4);
}

TEST_CASE("GetHeaderPayloadOperators")
{
  // Validates that get_header{part} / get_payload{part, 0} pipe operators
  // correctly return the right messages, including access to parts at index > 0.
  o2::framework::DataProcessingHeader dph{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());

  std::vector<fair::mq::MessagePtr> messages;

  // Add two separate header-payload pairs
  for (size_t part = 0; part < 2; ++part) {
    o2::header::DataHeader dh{};
    dh.dataDescription = "CLUSTERS";
    dh.dataOrigin = "TPC";
    dh.subSpecification = part;
    dh.splitPayloadParts = 1;
    dh.splitPayloadIndex = 0;
    messages.emplace_back(o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph}));
    messages.emplace_back(transport->CreateMessage(100 + part * 100));
  }

  REQUIRE(messages.size() == 4);

  // Validate part 0
  auto& hdr0 = messages | get_header{0};
  REQUIRE(hdr0.get() != nullptr);
  auto* dh0 = o2::header::get<o2::header::DataHeader*>(hdr0->GetData());
  REQUIRE(dh0 != nullptr);
  REQUIRE(dh0->subSpecification == 0);
  auto& pl0 = messages | get_payload{0, 0};
  REQUIRE(pl0.get() != nullptr);
  REQUIRE(pl0->GetSize() == 100);

  // Validate part 1
  auto& hdr1 = messages | get_header{1};
  REQUIRE(hdr1.get() != nullptr);
  auto* dh1 = o2::header::get<o2::header::DataHeader*>(hdr1->GetData());
  REQUIRE(dh1 != nullptr);
  REQUIRE(dh1->subSpecification == 1);
  auto& pl1 = messages | get_payload{1, 0};
  REQUIRE(pl1.get() != nullptr);
  REQUIRE(pl1->GetSize() == 200);

  REQUIRE((messages | count_parts{}) == 2);
  REQUIRE((messages | count_payloads{}) == 2);
  REQUIRE((messages | get_pair{0}).headerIdx == 0);
  REQUIRE((messages | get_pair{0}).payloadIdx == 1);
  REQUIRE((messages | get_pair{1}).headerIdx == 2);
  REQUIRE((messages | get_pair{1}).payloadIdx == 3);
}

TEST_CASE("GetHeaderPayloadMultiPayload")
{
  // Validates get_header{part} / get_payload{part, subpart} where both
  // part and subpart can be non-zero.
  // Layout:
  //   part 0: standard (1 header + 1 payload)  → splitPayloadParts=1
  //   part 1: multi-payload (1 header + 3 payloads) → splitPayloadParts=3, splitPayloadIndex=3
  o2::framework::DataProcessingHeader dph{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());

  std::vector<fair::mq::MessagePtr> messages;

  // Part 0: standard header-payload pair
  {
    o2::header::DataHeader dh{};
    dh.dataDescription = "CLUSTERS";
    dh.dataOrigin = "TPC";
    dh.subSpecification = 0;
    dh.splitPayloadParts = 1;
    dh.splitPayloadIndex = 0;
    messages.emplace_back(o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph}));
    messages.emplace_back(transport->CreateMessage(100));
  }

  // Part 1: one header with 3 payloads (splitPayloadIndex == splitPayloadParts)
  {
    o2::header::DataHeader dh{};
    dh.dataDescription = "TRACKS";
    dh.dataOrigin = "TPC";
    dh.subSpecification = 1;
    dh.splitPayloadParts = 3;
    dh.splitPayloadIndex = 3;
    messages.emplace_back(o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph}));
    messages.emplace_back(transport->CreateMessage(200));
    messages.emplace_back(transport->CreateMessage(300));
    messages.emplace_back(transport->CreateMessage(400));
  }

  // messages: [hdr0, pl0, hdr1, pl1_0, pl1_1, pl1_2]
  REQUIRE(messages.size() == 6);

  // Part 0
  auto& hdr0 = messages | get_header{0};
  REQUIRE(hdr0.get() != nullptr);
  auto* dh0 = o2::header::get<o2::header::DataHeader*>(hdr0->GetData());
  REQUIRE(dh0->subSpecification == 0);
  auto& pl0 = messages | get_payload{0, 0};
  REQUIRE(pl0.get() != nullptr);
  REQUIRE(pl0->GetSize() == 100);

  // Part 1: multi-payload header
  auto& hdr1 = messages | get_header{1};
  REQUIRE(hdr1.get() != nullptr);
  auto* dh1 = o2::header::get<o2::header::DataHeader*>(hdr1->GetData());
  REQUIRE(dh1->subSpecification == 1);

  auto& pl1_0 = messages | get_payload{1, 0};
  REQUIRE(pl1_0.get() != nullptr);
  REQUIRE(pl1_0->GetSize() == 200);

  auto& pl1_1 = messages | get_payload{1, 1};
  REQUIRE(pl1_1.get() != nullptr);
  REQUIRE(pl1_1->GetSize() == 300);

  auto& pl1_2 = messages | get_payload{1, 2};
  REQUIRE(pl1_2.get() != nullptr);
  REQUIRE(pl1_2->GetSize() == 400);

  REQUIRE((messages | get_num_payloads{0}) == 1);
  REQUIRE((messages | get_num_payloads{1}) == 3);
  REQUIRE((messages | count_parts{}) == 2);
  REQUIRE((messages | count_payloads{}) == 4);
  REQUIRE((messages | get_pair{0}).headerIdx == 0);
  REQUIRE((messages | get_pair{0}).payloadIdx == 1);
  REQUIRE((messages | get_pair{1}).headerIdx == 2);
  REQUIRE((messages | get_pair{1}).payloadIdx == 3);
  REQUIRE((messages | get_pair{2}).headerIdx == 2);
  REQUIRE((messages | get_pair{2}).payloadIdx == 4);
  REQUIRE((messages | get_pair{3}).headerIdx == 2);
  REQUIRE((messages | get_pair{3}).payloadIdx == 5);
}

TEST_CASE("TraditionalSplitParts")
{
  // Validates operators with traditional split parts layout:
  // 3 (header, payload) pairs where splitPayloadParts=3, splitPayloadIndex=0,1,2
  // Memory layout: [hdr0, pl0, hdr1, pl1, hdr2, pl2]
  o2::framework::DataProcessingHeader dph{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());

  std::vector<fair::mq::MessagePtr> messages;

  for (size_t i = 0; i < 3; ++i) {
    o2::header::DataHeader dh{};
    dh.dataDescription = "CLUSTERS";
    dh.dataOrigin = "TPC";
    dh.subSpecification = 0;
    dh.splitPayloadParts = 3;
    dh.splitPayloadIndex = i;
    messages.emplace_back(o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph}));
    messages.emplace_back(transport->CreateMessage(100 * (i + 1)));
  }

  REQUIRE(messages.size() == 6);

  REQUIRE((messages | count_payloads{}) == 3);
  REQUIRE((messages | count_parts{}) == 3);

  for (size_t i = 0; i < 3; ++i) {
    auto& hdr = messages | get_header{i};
    REQUIRE(hdr.get() != nullptr);
    auto* dh = o2::header::get<o2::header::DataHeader*>(hdr->GetData());
    REQUIRE(dh != nullptr);
    REQUIRE(dh->splitPayloadIndex == i);

    auto& pl = messages | get_payload{i, 0};
    REQUIRE(pl.get() != nullptr);
    REQUIRE(pl->GetSize() == 100 * (i + 1));
  }

  for (size_t i = 0; i < 3; ++i) {
    auto indices = messages | get_dataref_indices{i, 0};
    REQUIRE(indices.headerIdx == 2 * i);
    REQUIRE(indices.payloadIdx == 2 * i + 1);
  }

  for (size_t i = 0; i < 3; ++i) {
    auto indices = messages | get_pair{i};
    REQUIRE(indices.headerIdx == 2 * i);
    REQUIRE(indices.payloadIdx == 2 * i + 1);
  }

  for (size_t i = 0; i < 3; ++i) {
    REQUIRE((messages | get_num_payloads{i}) == 1);
  }
  REQUIRE((messages | count_parts{}) == 3);
  REQUIRE((messages | count_payloads{}) == 3);
}
