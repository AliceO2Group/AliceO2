// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MemoryResources
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "MemoryResources/MemoryResources.h"
#include "FairMQTransportFactory.h"
#include <vector>
#include <cstring>

namespace o2
{
namespace pmr
{
auto factoryZMQ = FairMQTransportFactory::CreateTransportFactory("zeromq");
auto factorySHM = FairMQTransportFactory::CreateTransportFactory("shmem");

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

auto allocZMQ = getTransportAllocator(factoryZMQ.get());
auto allocSHM = getTransportAllocator(factorySHM.get());

BOOST_AUTO_TEST_CASE(transportallocatormap_test)
{
  BOOST_CHECK(allocZMQ != nullptr && allocSHM != allocZMQ);
  auto _tmp = getTransportAllocator(factoryZMQ.get());
  BOOST_CHECK(_tmp == allocZMQ);
}

using namespace boost::container::pmr;

BOOST_AUTO_TEST_CASE(allocator_test)
{
  testData::nconstructions = 0;

  {
    std::vector<testData, polymorphic_allocator<testData>> v(polymorphic_allocator<testData>{allocZMQ});
    v.reserve(3);
    BOOST_CHECK(v.capacity() == 3);
    BOOST_CHECK(allocZMQ->getNumberOfMessages() == 1);
    v.emplace_back(1);
    v.emplace_back(2);
    v.emplace_back(3);
    BOOST_CHECK((byte*)&(*v.end()) - (byte*)&(*v.begin()) == 3 * sizeof(testData));
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
  testData::nconstructions = 0;

  FairMQMessagePtr message{nullptr};

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
}; // namespace pmr
}; // namespace o2
