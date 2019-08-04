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

using namespace o2::base;
using namespace o2::header;
using namespace o2::pmr;

auto factoryZMQ = FairMQTransportFactory::CreateTransportFactory("zeromq");
auto factorySHM = FairMQTransportFactory::CreateTransportFactory("shmem");
auto allocZMQ = getTransportAllocator(factoryZMQ.get());
auto allocSHM = getTransportAllocator(factorySHM.get());

BOOST_AUTO_TEST_CASE(getMessage_Stack)
{
  {
    //check that a message is constructed properly with the default new_delete_resource
    Stack s1{DataHeader{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}},
             NameHeader<9>{"somename"}};

    auto message = o2::pmr::getMessage(std::move(s1), allocZMQ);

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
    Stack s1{allocZMQ, DataHeader{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}},
             NameHeader<9>{"somename"}};
    BOOST_TEST(allocZMQ->getNumberOfMessages() == 1);

    auto message = o2::pmr::getMessage(std::move(s1), allocSHM);

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

BOOST_AUTO_TEST_CASE(addDataBlockForEach_test)
{
  {
    //simple addition of a data block from an exisiting message
    O2Message message;
    auto simpleMessage = factoryZMQ->CreateMessage(10);
    addDataBlock(message,
                 Stack{allocZMQ, DataHeader{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}}},
                 std::move(simpleMessage));
    BOOST_CHECK(message.Size() == 2);
  }

  {
    struct elem {
      int i;
      int j;
    };
    using namespace boost::container::pmr;
    O2Message message;
    std::vector<elem, polymorphic_allocator<elem>> vec(polymorphic_allocator<elem>{allocZMQ});
    vec.reserve(100);
    vec.push_back({1, 2});
    vec.push_back({3, 4});

    addDataBlock(message,
                 Stack{allocZMQ, DataHeader{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}}},
                 std::move(vec));
    BOOST_CHECK(message.Size() == 2);
    BOOST_CHECK(vec.size() == 0);
    BOOST_CHECK(message[0].GetSize() == 80);
    BOOST_CHECK(message[1].GetSize() == 2 * sizeof(elem)); //check the size of the buffer is set correctly

    //check contents
    int sum{0};
    forEach(message, [&](auto header, auto data) {
      const int* numbers = reinterpret_cast<const int*>(data.data());
      sum = numbers[0] + numbers[1] + numbers[2] + numbers[3];
    });
    BOOST_CHECK(sum == 10);

    //add one more data block and check total size using forEach;
    addDataBlock(message,
                 Stack{allocZMQ, DataHeader{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}}},
                 factoryZMQ->CreateMessage(10));
    int size{0};
    forEach(message, [&](auto header, auto data) { size += header.size() + data.size(); });
    BOOST_CHECK(size == 80 + 2 * sizeof(elem) + 80 + 10);

    //check contents (headers)
    int checkOK{0};
    forEach(message, [&](auto header, auto data) {
      auto dh = get<DataHeader*>(header.data());
      if (dh->dataDescription == gDataDescriptionInvalid && dh->dataOrigin == gDataOriginInvalid) {
        checkOK++;
      };
    });
    BOOST_CHECK(checkOK == 2);
  }
}
