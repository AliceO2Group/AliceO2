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
#include "Framework/MessageSet.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/Stack.h"
#include "MemoryResources/MemoryResources.h"
#include <catch_amalgamated.hpp>

using namespace o2::framework;

TEST_CASE("MessageSet")
{
  o2::framework::MessageSet msgSet;
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
  msgSet.add([&ptrs](size_t i) -> fair::mq::MessagePtr& { return ptrs[i]; }, 2);

  REQUIRE(msgSet.messages.size() == 2);
  REQUIRE((msgSet.messages | count_payloads{}) == 1);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 0}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 0}).payloadIdx == 1);
  REQUIRE((msgSet.messages | get_pair{0}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_pair{0}).payloadIdx == 1);
  CHECK_THROWS((msgSet.messages | get_pair{1}));
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
  o2::framework::MessageSet msgSet([&ptrs](size_t i) -> fair::mq::MessagePtr& { return ptrs[i]; }, 2);

  REQUIRE(msgSet.messages.size() == 2);
  REQUIRE((msgSet.messages | count_payloads{}) == 1);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 0}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 0}).payloadIdx == 1);
  REQUIRE((msgSet.messages | get_pair{0}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_pair{0}).payloadIdx == 1);
  CHECK_THROWS((msgSet.messages | get_pair{1}));
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
  o2::framework::MessageSet msgSet([&ptrs](size_t i) -> fair::mq::MessagePtr& { return ptrs[i]; }, 3);

  REQUIRE(msgSet.messages.size() == 3);
  REQUIRE((msgSet.messages | count_payloads{}) == 2);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 0}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 0}).payloadIdx == 1);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 1}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 1}).payloadIdx == 2);
  REQUIRE((msgSet.messages | get_pair{0}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_pair{0}).payloadIdx == 1);
  REQUIRE((msgSet.messages | get_pair{1}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_pair{1}).payloadIdx == 2);
  CHECK_THROWS((msgSet.messages | get_pair{2}));
}

TEST_CASE("MessageSetAddPartRef")
{
  std::vector<fair::mq::MessagePtr> ptrs;
  std::unique_ptr<fair::mq::Message> msg(nullptr);
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  ptrs.emplace_back(std::move(msg));
  ptrs.emplace_back(std::move(msg2));
  PartRef ref{std::move(msg), std::move(msg2)};
  o2::framework::MessageSet msgSet;
  msgSet.add(std::move(ref));

  REQUIRE(msgSet.messages.size() == 2);
}

TEST_CASE("MessageSetAddMultiple")
{
  std::vector<fair::mq::MessagePtr> ptrs;
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
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::MessagePtr header1 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh1, dph});
  fair::mq::MessagePtr header2 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh2, dph});
  fair::mq::MessagePtr header3 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh3, dph});

  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  std::unique_ptr<fair::mq::Message> msg3(nullptr);
  PartRef ref{std::move(header1), std::move(msg2)};
  o2::framework::MessageSet msgSet;
  msgSet.add(std::move(ref));
  PartRef ref2{std::move(header2), std::move(msg2)};
  msgSet.add(std::move(ref2));
  std::vector<fair::mq::MessagePtr> msgs;
  msgs.push_back(std::move(header3));
  msgs.push_back(std::unique_ptr<fair::mq::Message>(nullptr));
  msgs.push_back(std::unique_ptr<fair::mq::Message>(nullptr));
  msgSet.add([&msgs](size_t i) {
    return std::move(msgs[i]);
  },
             3);

  REQUIRE(msgSet.messages.size() == 7);

  REQUIRE((msgSet.messages | count_payloads{}) == 4);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 0}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_dataref_indices{0, 0}).payloadIdx == 1);
  REQUIRE((msgSet.messages | get_dataref_indices{1, 0}).headerIdx == 2);
  REQUIRE((msgSet.messages | get_dataref_indices{1, 0}).payloadIdx == 3);
  REQUIRE((msgSet.messages | get_dataref_indices{2, 0}).headerIdx == 4);
  REQUIRE((msgSet.messages | get_dataref_indices{2, 0}).payloadIdx == 5);
  REQUIRE((msgSet.messages | get_dataref_indices{2, 1}).headerIdx == 4);
  REQUIRE((msgSet.messages | get_dataref_indices{2, 1}).payloadIdx == 6);
  REQUIRE((msgSet.messages | get_pair{0}).headerIdx == 0);
  REQUIRE((msgSet.messages | get_pair{0}).payloadIdx == 1);
  REQUIRE((msgSet.messages | get_pair{1}).headerIdx == 2);
  REQUIRE((msgSet.messages | get_pair{1}).payloadIdx == 3);
  REQUIRE((msgSet.messages | get_pair{2}).headerIdx == 4);
  REQUIRE((msgSet.messages | get_pair{2}).payloadIdx == 5);
  REQUIRE((msgSet.messages | get_pair{3}).headerIdx == 4);
  REQUIRE((msgSet.messages | get_pair{3}).payloadIdx == 6);
}
