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

#define BOOST_TEST_MODULE Test MemoryResources
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "MemoryResources/MemoryResources.h"
#include <fairmq/TransportFactory.h>
#include <fairmq/Tools.h>
#include <fairmq/ProgOptions.h>
#include <vector>
#include <cstring>

namespace o2::pmr
{
struct testData {
  int i{1};
  static int nconstructions;
  testData()
  {
    ++nconstructions;
  }
  testData(const testData& in) : i{in.i}
  {
    ++nconstructions;
  }
  testData(const testData&& in) : i{in.i}
  {
    ++nconstructions;
  }
  testData(int in) : i{in}
  {
    ++nconstructions;
  }
};

int testData::nconstructions = 0;

BOOST_AUTO_TEST_CASE(transportallocatormap_test)
{
  size_t session{fair::mq::tools::UuidHash()};
  fair::mq::ProgOptions config;
  config.SetProperty<std::string>("session", std::to_string(session));

  auto factoryZMQ = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto factorySHM = fair::mq::TransportFactory::CreateTransportFactory("shmem");
  auto allocZMQ = getTransportAllocator(factoryZMQ.get());
  auto allocSHM = getTransportAllocator(factorySHM.get());
  BOOST_CHECK(allocZMQ != nullptr && allocSHM != allocZMQ);
  auto _tmp = getTransportAllocator(factoryZMQ.get());
  BOOST_CHECK(_tmp == allocZMQ);
}

using namespace boost::container::pmr;

BOOST_AUTO_TEST_CASE(allocator_test)
{
  size_t session{fair::mq::tools::UuidHash()};
  fair::mq::ProgOptions config;
  config.SetProperty<std::string>("session", std::to_string(session));

  auto factoryZMQ = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto factorySHM = fair::mq::TransportFactory::CreateTransportFactory("shmem");
  auto allocZMQ = getTransportAllocator(factoryZMQ.get());
  auto allocSHM = getTransportAllocator(factorySHM.get());

  testData::nconstructions = 0;

  {
    std::vector<testData, polymorphic_allocator<testData>> v(polymorphic_allocator<testData>{allocZMQ});
    v.reserve(3);
    BOOST_CHECK(v.capacity() == 3);
    BOOST_CHECK(allocZMQ->getNumberOfMessages() == 1);
    v.emplace_back(1);
    v.emplace_back(2);
    v.emplace_back(3);
    BOOST_CHECK((std::byte*)&(*v.end()) - (std::byte*)&(*v.begin()) == 3 * sizeof(testData));
    BOOST_CHECK(testData::nconstructions == 3);
  }

  testData::nconstructions = 0;
  {
    std::vector<testData, SpectatorAllocator<testData>> v(SpectatorAllocator<testData>{allocZMQ});
    v.reserve(3);
    BOOST_CHECK(allocZMQ->getNumberOfMessages() == 1);
    v.emplace_back(1);
    v.emplace_back(2);
    v.emplace_back(3);
    BOOST_CHECK(testData::nconstructions == 3);
  }
  BOOST_CHECK(allocZMQ->getNumberOfMessages() == 0);
}

BOOST_AUTO_TEST_CASE(getMessage_test)
{
  size_t session{fair::mq::tools::UuidHash()};
  fair::mq::ProgOptions config;
  config.SetProperty<std::string>("session", std::to_string(session));

  auto factoryZMQ = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto factorySHM = fair::mq::TransportFactory::CreateTransportFactory("shmem");
  auto allocZMQ = getTransportAllocator(factoryZMQ.get());
  auto allocSHM = getTransportAllocator(factorySHM.get());

  testData::nconstructions = 0;

  fair::mq::MessagePtr message{nullptr};

  int* messageArray{nullptr};

  // test message creation on the same channel it was allocated with
  {
    std::vector<testData, polymorphic_allocator<testData>> v(polymorphic_allocator<testData>{allocZMQ});
    v.emplace_back(1);
    v.emplace_back(2);
    v.emplace_back(3);
    void* vectorBeginPtr = &v[0];
    message = o2::pmr::getMessage(std::move(v));
    BOOST_CHECK(message != nullptr);
    BOOST_CHECK(message->GetData() == vectorBeginPtr);
  }
  BOOST_CHECK(message->GetSize() == 3 * sizeof(testData));
  messageArray = static_cast<int*>(message->GetData());
  BOOST_CHECK(messageArray[0] == 1 && messageArray[1] == 2 && messageArray[2] == 3);

  // test message creation on a different channel than it was allocated with
  {
    std::vector<testData, polymorphic_allocator<testData>> v(polymorphic_allocator<testData>{allocZMQ});
    v.emplace_back(4);
    v.emplace_back(5);
    v.emplace_back(6);
    void* vectorBeginPtr = &v[0];
    message = o2::pmr::getMessage(std::move(v), allocSHM);
    BOOST_CHECK(message != nullptr);
    BOOST_CHECK(message->GetData() != vectorBeginPtr);
  }
  BOOST_CHECK(message->GetSize() == 3 * sizeof(testData));
  messageArray = static_cast<int*>(message->GetData());
  BOOST_CHECK(messageArray[0] == 4 && messageArray[1] == 5 && messageArray[2] == 6);

  {
    std::vector<testData, SpectatorAllocator<testData>> v(SpectatorAllocator<testData>{allocSHM});
  }
}

BOOST_AUTO_TEST_CASE(adoptVector_test)
{
  size_t session{fair::mq::tools::UuidHash()};
  fair::mq::ProgOptions config;
  config.SetProperty<std::string>("session", std::to_string(session));

  auto factoryZMQ = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto factorySHM = fair::mq::TransportFactory::CreateTransportFactory("shmem");
  auto allocZMQ = getTransportAllocator(factoryZMQ.get());
  auto allocSHM = getTransportAllocator(factorySHM.get());

  testData::nconstructions = 0;

  //Create a bogus message
  auto message = factoryZMQ->CreateMessage(3 * sizeof(testData));
  auto messageAddr = message.get();
  testData tmpBuf[3] = {3, 2, 1};
  std::memcpy(message->GetData(), tmpBuf, 3 * sizeof(testData));

  auto adoptedOwner = adoptVector<testData>(3, std::move(message));
  BOOST_CHECK(adoptedOwner[0].i == 3);
  BOOST_CHECK(adoptedOwner[1].i == 2);
  BOOST_CHECK(adoptedOwner[2].i == 1);

  auto reclaimedMessage = o2::pmr::getMessage(std::move(adoptedOwner));
  BOOST_CHECK(reclaimedMessage.get() == messageAddr);
  BOOST_CHECK(adoptedOwner.size() == 0);

  auto modified = adoptVector<testData>(3, std::move(reclaimedMessage));
  modified.emplace_back(9);
  BOOST_CHECK(modified[3].i == 9);
  BOOST_CHECK(modified.size() == 4);
  BOOST_CHECK(testData::nconstructions == 7);
  auto modifiedMessage = getMessage(std::move(modified));
  BOOST_CHECK(modifiedMessage != nullptr);
  BOOST_CHECK(modifiedMessage.get() != messageAddr);
}

BOOST_AUTO_TEST_CASE(test_SpectatorMemoryResource)
{
  constexpr int size = 5;
  auto buffer = std::make_unique<int[]>(size);
  auto const* bufferdata = buffer.get();
  SpectatorMemoryResource<decltype(buffer)> resource(std::move(buffer), size * sizeof(int));
  std::vector<int, o2::pmr::SpectatorAllocator<int>> bufferclone(size, o2::pmr::SpectatorAllocator<int>(&resource));
  BOOST_CHECK(bufferclone.data() == bufferdata);
  BOOST_CHECK(bufferclone.size() == size);
  BOOST_CHECK_THROW(bufferclone.resize(2 * size), std::runtime_error);

  auto vecbuf = std::make_unique<std::vector<int>>(size);
  auto const* vectordata = vecbuf->data();
  SpectatorMemoryResource<decltype(vecbuf)> vecresource(std::move(vecbuf));
  std::vector<int, o2::pmr::SpectatorAllocator<int>> vecclone(size, o2::pmr::SpectatorAllocator<int>(&vecresource));
  BOOST_CHECK(vecclone.data() == vectordata);
  BOOST_CHECK(vecclone.size() == size);
  BOOST_CHECK_THROW(vecclone.resize(2 * size), std::runtime_error);

  std::vector<int, o2::pmr::SpectatorAllocator<int>> vecmove;
  vecmove = std::move(vecclone);
  BOOST_CHECK(vecclone.size() == 0);
  BOOST_CHECK(vecmove.data() == vectordata);
  BOOST_CHECK(vecmove.size() == size);
}

}; // namespace o2::pmr
