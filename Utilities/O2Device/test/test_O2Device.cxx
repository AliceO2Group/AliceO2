// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test O2Device
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "O2Device/O2Device.h"
#include "Headers/NameHeader.h"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

using namespace o2::Base;
using namespace o2::header;
using namespace o2::memoryResources;

auto factoryZMQ = FairMQTransportFactory::CreateTransportFactory("zeromq");
auto factorySHM = FairMQTransportFactory::CreateTransportFactory("shmem");
auto allocZMQ = getTransportAllocator(factoryZMQ.get());
auto allocSHM = getTransportAllocator(factorySHM.get());

BOOST_AUTO_TEST_CASE(getMessage_Stack)
{
  {
    //check that a message is constructed properly with the default new_delete_resource
    Stack s1{ DataHeader{ gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{ 0 }, 0 },
              NameHeader<9>{ "somename" } };

    auto message = o2::memoryResources::getMessage(std::move(s1), allocZMQ);

    BOOST_REQUIRE(s1.data() == nullptr);
    BOOST_REQUIRE(message != nullptr);
    auto* h3 = get<NameHeader<0>*>(message->GetData());
    BOOST_REQUIRE(h3 != nullptr);
    BOOST_CHECK(h3->getNameLength() == 9);
    BOOST_CHECK(0 == std::strcmp(h3->getName(), "somename"));
    BOOST_CHECK(message->GetType() == fair::mq::Transport::ZMQ);
  }
  {
    //check that a message is constructed properly, cross resource
    Stack s1{ allocZMQ, DataHeader{ gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{ 0 }, 0 },
              NameHeader<9>{ "somename" } };
    BOOST_TEST(allocZMQ->getNumberOfMessages() == 1);

    auto message = o2::memoryResources::getMessage(std::move(s1), allocSHM);

    BOOST_TEST(allocZMQ->getNumberOfMessages() == 0);
    BOOST_TEST(allocSHM->getNumberOfMessages() == 0);
    BOOST_REQUIRE(s1.data() == nullptr);
    BOOST_REQUIRE(message != nullptr);
    auto* h3 = get<NameHeader<0>*>(message->GetData());
    BOOST_REQUIRE(h3 != nullptr);
    BOOST_CHECK(h3->getNameLength() == 9);
    BOOST_CHECK(0 == std::strcmp(h3->getName(), "somename"));
    BOOST_CHECK(message->GetType() == fair::mq::Transport::SHM);
  }
}
