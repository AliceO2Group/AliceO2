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
namespace memoryResources
{
auto factoryZMQ = FairMQTransportFactory::CreateTransportFactory("zeromq");
auto factorySHM = FairMQTransportFactory::CreateTransportFactory("shmem");

struct testData {
  int i{ 1 };
  static int nallocated;
  static int nallocations;
  static int ndeallocations;
  testData()
  {
    ++nallocated;
    ++nallocations;
  }
  testData(const testData& in) : i{ in.i }
  {
    ++nallocated;
    ++nallocations;
  }
  testData(const testData&& in) : i{ in.i }
  {
    ++nallocated;
    ++nallocations;
  }
  testData(int in) : i{ in }
  {
    ++nallocated;
    ++nallocations;
  }
  ~testData()
  {
    --nallocated;
    ++ndeallocations;
  }
};

int testData::nallocated = 0;
int testData::nallocations = 0;
int testData::ndeallocations = 0;

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
  testData::nallocations = 0;
  testData::ndeallocations = 0;

  {
    std::vector<testData, polymorphic_allocator<testData>> v(polymorphic_allocator<testData>{ allocZMQ });
    v.reserve(3);
    BOOST_CHECK(v.capacity() == 3);
    BOOST_CHECK(allocZMQ->getNumberOfMessages() == 1);
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    BOOST_CHECK((byte*)&(*v.end()) - (byte*)&(*v.begin()) == 3 * sizeof(testData));
    BOOST_CHECK(testData::nallocated == 3);
  }
  BOOST_CHECK(testData::nallocated == 0);
  BOOST_CHECK(testData::nallocations == testData::ndeallocations);

  testData::nallocations = 0;
  testData::ndeallocations = 0;
  {
    std::vector<testData, SpectatorAllocator<testData>> v(SpectatorAllocator<testData>{ allocZMQ });
    v.reserve(3);
    BOOST_CHECK(allocZMQ->getNumberOfMessages() == 1);
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    BOOST_CHECK(testData::nallocated == 3);
  }
  BOOST_CHECK(testData::nallocated == 3); //ByteSpectatorAllocator does not call dtors so nallocated remains at 3;
  BOOST_CHECK(allocZMQ->getNumberOfMessages() == 0);
}

BOOST_AUTO_TEST_CASE(getMessage_test)
{
  testData::nallocations = 0;
  testData::ndeallocations = 0;

  FairMQMessagePtr message{ nullptr };

  int* messageArray{ nullptr };

  // test message creation on the same channel it was allocated with
  {
    std::vector<testData, polymorphic_allocator<testData>> v(polymorphic_allocator<testData>{ allocZMQ });
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    void* vectorBeginPtr = &v[0];
    message = getMessage(std::move(v));
    BOOST_CHECK(message != nullptr);
    BOOST_CHECK(message->GetData() == vectorBeginPtr);
  }
  BOOST_CHECK(message->GetSize() == 3 * sizeof(testData));
  messageArray = static_cast<int*>(message->GetData());
  BOOST_CHECK(messageArray[0] == 1 && messageArray[1] == 2 && messageArray[2] == 3);

  // test message creation on a different channel than it was allocated with
  {
    std::vector<testData, polymorphic_allocator<testData>> v(polymorphic_allocator<testData>{ allocZMQ });
    v.push_back(4);
    v.push_back(5);
    v.push_back(6);
    void* vectorBeginPtr = &v[0];
    message = getMessage(std::move(v), allocSHM);
    BOOST_CHECK(message != nullptr);
    BOOST_CHECK(message->GetData() != vectorBeginPtr);
  }
  BOOST_CHECK(message->GetSize() == 3 * sizeof(testData));
  messageArray = static_cast<int*>(message->GetData());
  BOOST_CHECK(messageArray[0] == 4 && messageArray[1] == 5 && messageArray[2] == 6);

  {
    std::vector<testData, SpectatorAllocator<testData>> v(SpectatorAllocator<testData>{ allocSHM });
  }
}

BOOST_AUTO_TEST_CASE(adoptVector_test)
{
  //Create a bogus message
  auto message = factoryZMQ->CreateMessage(3 * sizeof(testData));
  auto messageAddr = message.get();
  testData tmpBuf[3] = { 3, 2, 1 };
  std::memcpy(message->GetData(), tmpBuf, 3 * sizeof(testData));

  auto adoptedOwner = adoptVector<testData>(3, getTransportAllocator(factoryZMQ.get()), std::move(message));
  BOOST_CHECK(adoptedOwner[0].i == 3);
  BOOST_CHECK(adoptedOwner[1].i == 2);
  BOOST_CHECK(adoptedOwner[2].i == 1);

  auto reclaimedMessage = getMessage(std::move(adoptedOwner));
  BOOST_CHECK(reclaimedMessage.get() == messageAddr);
  BOOST_CHECK(adoptedOwner.size() == 0);
}
};
};
