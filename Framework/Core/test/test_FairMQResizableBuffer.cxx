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

#include <catch_amalgamated.hpp>
#include "Framework/TableBuilder.h"
#include "Framework/FairMQResizableBuffer.h"
#include <fairmq/TransportFactory.h>
#include <cstring>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/util/config.h>

using namespace o2::framework;

template class std::unique_ptr<fair::mq::Message>;

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
TEST_CASE("TestCreation")
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  FairMQResizableBuffer buffer{[&transport](size_t size) -> std::unique_ptr<fair::mq::Message> {
    return std::move(transport->CreateMessage(size));
  }};
}

// Check a few invariants for Resize and Reserve operations
TEST_CASE("TestInvariants")
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  FairMQResizableBuffer buffer{[&transport](size_t size) -> std::unique_ptr<fair::mq::Message> {
    return std::move(transport->CreateMessage(size));
  }};

  REQUIRE(buffer.size() == 0);
  REQUIRE(buffer.size() <= buffer.capacity());

  auto status = buffer.Reserve(10000);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 10000);
  REQUIRE(buffer.size() == 0);
  auto old_ptr = buffer.data();

  status = buffer.Resize(9000, false);
  REQUIRE(status.ok());
  REQUIRE(old_ptr == buffer.data());
  REQUIRE(buffer.capacity() == 10000);
  REQUIRE(buffer.size() == 9000);

  status = buffer.Resize(11000, false);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 11000);
  REQUIRE(buffer.size() == 11000);

  status = buffer.Resize(10000, false);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 11000);
  REQUIRE(buffer.size() == 10000);

  status = buffer.Resize(9000, true);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 11000);
  REQUIRE(buffer.size() == 9000);

  status = buffer.Resize(19000, true);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 19000);
  REQUIRE(buffer.size() == 19000);
}

// Check a few invariants for Resize and Reserve operations
TEST_CASE("TestContents")
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  FairMQResizableBuffer buffer{[&transport](size_t size) -> std::unique_ptr<fair::mq::Message> {
    return std::move(transport->CreateMessage(size));
  }};

  REQUIRE(buffer.size() == 0);
  REQUIRE(buffer.size() <= buffer.capacity());

  auto status = buffer.Resize(10, true);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 10);
  REQUIRE(buffer.size() == 10);
  auto old_ptr = buffer.data();

  strcpy((char*)buffer.mutable_data(), "foo");

  status = buffer.Resize(9000, false);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 9000);
  REQUIRE(buffer.size() == 9000);
  REQUIRE(strncmp((const char*)buffer.data(), "foo", 3) == 0);

  status = buffer.Resize(4000, false);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 9000);
  REQUIRE(buffer.size() == 4000);
  REQUIRE(strncmp((const char*)buffer.data(), "foo", 3) == 0);

  status = buffer.Resize(40, true);
  REQUIRE(status.ok());
  REQUIRE(buffer.capacity() == 9000);
  REQUIRE(buffer.size() == 40);
  REQUIRE(strncmp((const char*)buffer.data(), "foo", 3) == 0);
}
