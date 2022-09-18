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

#define BOOST_TEST_MODULE Test Framework DataRelayer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
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
BOOST_AUTO_TEST_CASE(TestCreation)
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  FairMQResizableBuffer buffer{[&transport](size_t size) -> std::unique_ptr<fair::mq::Message> {
    return std::move(transport->CreateMessage(size));
  }};
}

// Check a few invariants for Resize and Reserve operations
BOOST_AUTO_TEST_CASE(TestInvariants)
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  FairMQResizableBuffer buffer{[&transport](size_t size) -> std::unique_ptr<fair::mq::Message> {
    return std::move(transport->CreateMessage(size));
  }};

  BOOST_REQUIRE_EQUAL(buffer.size(), 0);
  BOOST_REQUIRE(buffer.size() <= buffer.capacity());

  auto status = buffer.Reserve(10000);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 10000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 0);
  auto old_ptr = buffer.data();

  status = buffer.Resize(9000, false);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(old_ptr, buffer.data());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 10000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 9000);

  status = buffer.Resize(11000, false);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 11000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 11000);

  status = buffer.Resize(10000, false);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 11000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 10000);

  status = buffer.Resize(9000, true);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 11000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 9000);

  status = buffer.Resize(19000, true);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 19000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 19000);
}

// Check a few invariants for Resize and Reserve operations
BOOST_AUTO_TEST_CASE(TestContents)
{
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  FairMQResizableBuffer buffer{[&transport](size_t size) -> std::unique_ptr<fair::mq::Message> {
    return std::move(transport->CreateMessage(size));
  }};

  BOOST_REQUIRE_EQUAL(buffer.size(), 0);
  BOOST_REQUIRE(buffer.size() <= buffer.capacity());

  auto status = buffer.Resize(10, true);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 10);
  BOOST_REQUIRE_EQUAL(buffer.size(), 10);
  auto old_ptr = buffer.data();

  strcpy((char*)buffer.mutable_data(), "foo");

  status = buffer.Resize(9000, false);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 9000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 9000);
  BOOST_REQUIRE(strncmp((const char*)buffer.data(), "foo", 3) == 0);

  status = buffer.Resize(4000, false);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 9000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 4000);
  BOOST_REQUIRE(strncmp((const char*)buffer.data(), "foo", 3) == 0);

  status = buffer.Resize(40, true);
  BOOST_REQUIRE(status.ok());
  BOOST_REQUIRE_EQUAL(buffer.capacity(), 9000);
  BOOST_REQUIRE_EQUAL(buffer.size(), 40);
  BOOST_REQUIRE(strncmp((const char*)buffer.data(), "foo", 3) == 0);
}
