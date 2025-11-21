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
#include "Framework/MessageSet.h"
#include <catch_amalgamated.hpp>

using namespace o2::framework;

TEST_CASE("MessageSet") {
  o2::framework::MessageSet msgSet;
  std::vector<fair::mq::MessagePtr> ptrs;
  std::unique_ptr<fair::mq::Message> msg(nullptr);
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  ptrs.emplace_back(std::move(msg));
  ptrs.emplace_back(std::move(msg2));
  msgSet.add([&ptrs](size_t i) -> fair::mq::MessagePtr& { return ptrs[i]; }, 2);

  REQUIRE(msgSet.messages.size() == 2);
  REQUIRE(msgSet.messageMap.size() == 1);
  REQUIRE(msgSet.pairMap.size() == 1);
  REQUIRE(msgSet.messageMap[0].position == 0);
  REQUIRE(msgSet.messageMap[0].size == 1);

  REQUIRE(msgSet.pairMap[0].partIndex == 0);
  REQUIRE(msgSet.pairMap[0].payloadIndex == 0);
}

TEST_CASE("MessageSetWithFunction") {
  std::vector<fair::mq::MessagePtr> ptrs;
  std::unique_ptr<fair::mq::Message> msg(nullptr);
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  ptrs.emplace_back(std::move(msg));
  ptrs.emplace_back(std::move(msg2));
  o2::framework::MessageSet msgSet([&ptrs](size_t i) -> fair::mq::MessagePtr& { return ptrs[i]; }, 2);

  REQUIRE(msgSet.messages.size() == 2);
  REQUIRE(msgSet.messageMap.size() == 1);
  REQUIRE(msgSet.pairMap.size() == 1);
  REQUIRE(msgSet.messageMap[0].position == 0);
  REQUIRE(msgSet.messageMap[0].size == 1);

  REQUIRE(msgSet.pairMap[0].partIndex == 0);
  REQUIRE(msgSet.pairMap[0].payloadIndex == 0);
}

TEST_CASE("MessageSetWithMultipart") {
  std::vector<fair::mq::MessagePtr> ptrs;
  std::unique_ptr<fair::mq::Message> msg(nullptr);
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  std::unique_ptr<fair::mq::Message> msg3(nullptr);
  ptrs.emplace_back(std::move(msg));
  ptrs.emplace_back(std::move(msg2));
  ptrs.emplace_back(std::move(msg3));
  o2::framework::MessageSet msgSet([&ptrs](size_t i) -> fair::mq::MessagePtr& { return ptrs[i]; }, 3);

  REQUIRE(msgSet.messages.size() == 3);
  REQUIRE(msgSet.messageMap.size() == 1);
  REQUIRE(msgSet.pairMap.size() == 2);
  REQUIRE(msgSet.messageMap[0].position == 0);
  REQUIRE(msgSet.messageMap[0].size == 2);

  REQUIRE(msgSet.pairMap[0].partIndex == 0);
  REQUIRE(msgSet.pairMap[0].payloadIndex == 0);
  REQUIRE(msgSet.pairMap[1].partIndex == 0);
  REQUIRE(msgSet.pairMap[1].payloadIndex == 1);
}

TEST_CASE("MessageSetAddPartRef") {
  std::vector<fair::mq::MessagePtr> ptrs;
  std::unique_ptr<fair::mq::Message> msg(nullptr);
  std::unique_ptr<fair::mq::Message> msg2(nullptr);
  ptrs.emplace_back(std::move(msg));
  ptrs.emplace_back(std::move(msg2));
  PartRef ref {std::move(msg), std::move(msg2)};
  o2::framework::MessageSet msgSet;
  msgSet.add(std::move(ref));

  REQUIRE(msgSet.messages.size() == 2);
  REQUIRE(msgSet.messageMap.size() == 1);
  REQUIRE(msgSet.pairMap.size() == 1);
  REQUIRE(msgSet.messageMap[0].position == 0);
  REQUIRE(msgSet.messageMap[0].size == 1);

  REQUIRE(msgSet.pairMap[0].partIndex == 0);
  REQUIRE(msgSet.pairMap[0].payloadIndex == 0);
}
