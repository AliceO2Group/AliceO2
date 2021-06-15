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

#include "Headers/NameHeader.h"

#include "MemoryResources/MemoryResources.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>
#include <fairmq/Tools.h>
#include <fairmq/ProgOptions.h>
#include <gsl/gsl>

using namespace o2::header;
using namespace o2::pmr;

//__________________________________________________________________________________________________
// addDataBlock for generic (compatible) containers, that is contiguous containers using the pmr allocator
template <typename ContainerT, typename std::enable_if<!std::is_same<ContainerT, FairMQMessagePtr>::value, int>::type = 0>
bool addDataBlock(FairMQParts& parts, o2::header::Stack&& inputStack, ContainerT&& inputData, o2::pmr::FairMQMemoryResource* targetResource = nullptr)
{
  using std::forward;
  using std::move;

  auto headerMessage = o2::pmr::getMessage(move(inputStack), targetResource);
  auto dataMessage = o2::pmr::getMessage(forward<ContainerT>(inputData), targetResource);

  parts.AddPart(move(headerMessage));
  parts.AddPart(move(dataMessage));

  return true;
}

//__________________________________________________________________________________________________
// addDataBlock for data already wrapped in FairMQMessagePtr
// note: since we cannot partially specialize function templates, use SFINAE here instead
template <typename ContainerT, typename std::enable_if<std::is_same<ContainerT, FairMQMessagePtr>::value, int>::type = 0>
bool addDataBlock(FairMQParts& parts, o2::header::Stack&& inputStack, ContainerT&& dataMessage, o2::pmr::FairMQMemoryResource* targetResource = nullptr)
{
  using std::move;

  //make sure the payload size in DataHeader corresponds to message size
  using o2::header::DataHeader;
  DataHeader* dataHeader = const_cast<DataHeader*>(o2::header::get<DataHeader*>(inputStack.data()));
  dataHeader->payloadSize = dataMessage->GetSize();

  auto headerMessage = o2::pmr::getMessage(move(inputStack), targetResource);

  parts.AddPart(move(headerMessage));
  parts.AddPart(move(dataMessage));

  return true;
}

template <typename I, typename F>
auto forEach(I begin, I end, F&& function)
{

  using span = gsl::span<const std::byte>;
  using SPAN_SIZE_TYPE = span::size_type;
  using gsl::narrow_cast;
  for (auto it = begin; it != end; ++it) {
    std::byte* headerBuffer{nullptr};
    SPAN_SIZE_TYPE headerBufferSize{0};
    if (*it != nullptr) {
      headerBuffer = reinterpret_cast<std::byte*>((*it)->GetData());
      headerBufferSize = narrow_cast<SPAN_SIZE_TYPE>((*it)->GetSize());
    }
    ++it;
    std::byte* dataBuffer{nullptr};
    SPAN_SIZE_TYPE dataBufferSize{0};
    if (*it != nullptr) {
      dataBuffer = reinterpret_cast<std::byte*>((*it)->GetData());
      dataBufferSize = narrow_cast<SPAN_SIZE_TYPE>((*it)->GetSize());
    }

    // call the user provided function
    function(span{headerBuffer, headerBufferSize}, span{dataBuffer, dataBufferSize});
  }
  return std::move(function);
}

/// Execute user code (e.g. a lambda) on each data block (header-payload pair)
/// returns the function (same as std::for_each)
template <typename F>
auto forEach(FairMQParts& parts, F&& function)
{
  if ((parts.Size() % 2) != 0) {
    throw std::invalid_argument(
      "number of parts in message not even (n%2 != 0), cannot be considered an O2 compliant message");
  }

  return forEach(parts.begin(), parts.end(), std::forward<F>(function));
}

BOOST_AUTO_TEST_CASE(getMessage_Stack)
{
  size_t session{fair::mq::tools::UuidHash()};
  fair::mq::ProgOptions config;
  config.SetProperty<std::string>("session", std::to_string(session));

  auto factoryZMQ = FairMQTransportFactory::CreateTransportFactory("zeromq");
  auto factorySHM = FairMQTransportFactory::CreateTransportFactory("shmem");
  BOOST_REQUIRE(factorySHM != nullptr);
  BOOST_REQUIRE(factoryZMQ != nullptr);
  auto allocZMQ = getTransportAllocator(factoryZMQ.get());
  BOOST_REQUIRE(allocZMQ != nullptr);
  auto allocSHM = getTransportAllocator(factorySHM.get());
  BOOST_REQUIRE(allocSHM != nullptr);
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
  size_t session{fair::mq::tools::UuidHash()};
  fair::mq::ProgOptions config;
  config.SetProperty<std::string>("session", std::to_string(session));

  auto factoryZMQ = FairMQTransportFactory::CreateTransportFactory("zeromq");
  BOOST_REQUIRE(factoryZMQ);
  auto allocZMQ = getTransportAllocator(factoryZMQ.get());
  BOOST_REQUIRE(allocZMQ);

  {
    //simple addition of a data block from an exisiting message
    FairMQParts message;
    auto simpleMessage = factoryZMQ->CreateMessage(10);
    addDataBlock(message,
                 Stack{allocZMQ, DataHeader{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}}},
                 std::move(simpleMessage));
    BOOST_CHECK(message.Size() == 2);
  }

  {
    int sizeofDataHeader = sizeof(o2::header::DataHeader);
    struct elem {
      int i;
      int j;
    };
    using namespace boost::container::pmr;
    FairMQParts message;
    std::vector<elem, polymorphic_allocator<elem>> vec(polymorphic_allocator<elem>{allocZMQ});
    vec.reserve(100);
    vec.push_back({1, 2});
    vec.push_back({3, 4});

    addDataBlock(message,
                 Stack{allocZMQ, DataHeader{gDataDescriptionInvalid, gDataOriginInvalid, DataHeader::SubSpecificationType{0}}},
                 std::move(vec));
    BOOST_CHECK(message.Size() == 2);
    BOOST_CHECK(vec.size() == 0);
    BOOST_CHECK(message[0].GetSize() == sizeofDataHeader);
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
    BOOST_CHECK(size == sizeofDataHeader + 2 * sizeof(elem) + sizeofDataHeader + 10);

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
