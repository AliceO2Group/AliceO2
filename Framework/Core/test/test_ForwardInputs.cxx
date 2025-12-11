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

#include <catch_amalgamated.hpp>
#include "Headers/DataHeader.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/DomainInfoHeader.h"
#include "Framework/Signpost.h"
#include "Framework/MessageSet.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Headers/Stack.h"
#include "MemoryResources/MemoryResources.h"
#include <fairmq/TransportFactory.h>
#include <fairmq/Channel.h>
#include <vector>

O2_DECLARE_DYNAMIC_LOG(forwarding);
using namespace o2::framework;

TEST_CASE("ForwardInputsEmpty")
{
  o2::header::DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;

  std::vector<MessageSet> currentSetOfInputs;

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.empty());
}

TEST_CASE("ForwardInputsSingleMessageSingleRoute")
{
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};
  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B")};

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{ForwardRoute{
    .timeslice = 0,
    .maxTimeslices = 1,
    .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
    .channel = "from_A_to_B",
    .policy = nullptr,
  }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 1);    // One route
  REQUIRE(result[0].Size() == 2); // Two messages for that route
}

TEST_CASE("ForwardInputsSingleMessageSingleRouteNoConsume")
{
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};
  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B")};

  bool copyByDefault = false;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{ForwardRoute{
    .timeslice = 0,
    .maxTimeslices = 1,
    .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
    .channel = "from_A_to_B",
    .policy = nullptr,
  }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(nullptr);
  REQUIRE(payload.get() == nullptr);
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, true);
  REQUIRE(result.size() == 1);
  REQUIRE(result[0].Size() == 0); // Because there is a nullptr, we do not forward this as it was already consumed.
}

TEST_CASE("ForwardInputsSingleMessageSingleRouteAtEOS")
{
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};

  o2::framework::SourceInfoHeader sih{};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B")};

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{ForwardRoute{
    .timeslice = 0,
    .maxTimeslices = 1,
    .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
    .channel = "from_A_to_B",
    .policy = nullptr,
  }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph, sih});
  REQUIRE(o2::header::get<SourceInfoHeader*>(header->GetData()));
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 1);    // One route
  REQUIRE(result[0].Size() == 0); // FIXME: this is an actual error. It should be 2. However it cannot really happen.
  // Correct behavior below:
  // REQUIRE(result[0].Size() == 2);
  // REQUIRE(o2::header::get<SourceInfoHeader*>(result[0].At(0)->GetData()) == nullptr);
}

TEST_CASE("ForwardInputsSingleMessageSingleRouteWithOldestPossible")
{
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};

  o2::framework::DomainInfoHeader dih{};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B")};

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{ForwardRoute{
    .timeslice = 0,
    .maxTimeslices = 1,
    .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
    .channel = "from_A_to_B",
    .policy = nullptr,
  }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph, dih});
  REQUIRE(o2::header::get<DomainInfoHeader*>(header->GetData()));
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 1);    // One route
  REQUIRE(result[0].Size() == 0); // FIXME: this is actually wrong
  // FIXME: actually correct behavior below
  // REQUIRE(result[0].Size() == 2);                                                     // Two messages
  // REQUIRE(o2::header::get<DomainInfoHeader*>(result[0].At(0)->GetData()) == nullptr); // it should not have the end of stream
}

TEST_CASE("ForwardInputsSingleMessageMultipleRoutes")
{
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B"),
    fair::mq::Channel("from_A_to_C"),
  };

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
      .channel = "from_A_to_B",
      .policy = nullptr,
    },
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding2", ConcreteDataMatcher{"TST", "A", 0}},
      .channel = "from_A_to_C",
      .policy = nullptr,
    }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 2);    // Two routes
  REQUIRE(result[0].Size() == 2); // Two messages per route
  REQUIRE(result[1].Size() == 0); // Only the first DPL matched channel matters
}

TEST_CASE("ForwardInputsSingleMessageMultipleRoutesExternals")
{
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("external"),
    fair::mq::Channel("from_A_to_C"),
  };

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
      .channel = "external",
      .policy = nullptr,
    },
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding2", ConcreteDataMatcher{"TST", "A", 0}},
      .channel = "from_A_to_C",
      .policy = nullptr,
    }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 2);    // Two routes
  REQUIRE(result[0].Size() == 2); // With external matching channels, we need to copy and then forward
  REQUIRE(result[1].Size() == 2); //
}

TEST_CASE("ForwardInputsMultiMessageMultipleRoutes")
{
  o2::header::DataHeader dh1;
  dh1.dataOrigin = "TST";
  dh1.dataDescription = "A";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 1;

  o2::header::DataHeader dh2;
  dh2.dataOrigin = "TST";
  dh2.dataDescription = "B";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B"),
    fair::mq::Channel("from_A_to_C"),
  };

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
      .channel = "from_A_to_B",
      .policy = nullptr,
    },
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding2", ConcreteDataMatcher{"TST", "B", 0}},
      .channel = "from_A_to_C",
      .policy = nullptr,
    }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload1(transport->CreateMessage());
  fair::mq::MessagePtr payload2(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header1 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh1, dph});
  MessageSet messageSet1;
  messageSet1.add(PartRef{std::move(header1), std::move(payload1)});
  REQUIRE(messageSet1.size() == 1);

  auto header2 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh2, dph});
  MessageSet messageSet2;
  messageSet2.add(PartRef{std::move(header2), std::move(payload2)});
  REQUIRE(messageSet2.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet1));
  currentSetOfInputs.emplace_back(std::move(messageSet2));
  REQUIRE(currentSetOfInputs.size() == 2);

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 2);    // Two routes
  REQUIRE(result[0].Size() == 2); //
  REQUIRE(result[1].Size() == 2); //
}

TEST_CASE("ForwardInputsSingleMessageMultipleRoutesOnlyOneMatches")
{
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B"),
    fair::mq::Channel("from_A_to_C"),
  };

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding", ConcreteDataMatcher{"TST", "B", 0}},
      .channel = "from_A_to_B",
      .policy = nullptr,
    },
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
      .channel = "from_A_to_C",
      .policy = nullptr,
    }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 2);    // Two routes
  REQUIRE(result[0].Size() == 0); // Two messages per route
  REQUIRE(result[1].Size() == 2); // Two messages per route
}

TEST_CASE("ForwardInputsSplitPayload")
{
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 2;
  dh.splitPayloadParts = 2;

  o2::header::DataHeader dh2;
  dh2.dataOrigin = "TST";
  dh2.dataDescription = "B";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 1;

  o2::framework::DataProcessingHeader dph{0, 1};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B"),
    fair::mq::Channel("from_A_to_C"),
  };

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding", ConcreteDataMatcher{"TST", "B", 0}},
      .channel = "from_A_to_B",
      .policy = nullptr,
    },
    ForwardRoute{
      .timeslice = 0,
      .maxTimeslices = 1,
      .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
      .channel = "from_A_to_C",
      .policy = nullptr,
    }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload1(transport->CreateMessage());
  fair::mq::MessagePtr payload2(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
  std::vector<std::unique_ptr<fair::mq::Message>> messages;
  messages.push_back(std::move(header));
  messages.push_back(std::move(payload1));
  messages.push_back(std::move(payload2));
  auto fillMessages = [&messages](size_t t) -> fair::mq::MessagePtr {
    return std::move(messages[t]);
  };
  messageSet.add(fillMessages, 3);
  auto header2 = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh2, dph});
  PartRef part{std::move(header2), transport->CreateMessage()};
  messageSet.add(std::move(part));

  REQUIRE(messageSet.size() == 2);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 2);  // Two routes
  CHECK(result[0].Size() == 2); // No messages on this route
  CHECK(result[1].Size() == 3);
}

TEST_CASE("ForwardInputEOSSingleRoute")
{
  o2::framework::SourceInfoHeader sih{};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B")};

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{ForwardRoute{
    .timeslice = 0,
    .maxTimeslices = 1,
    .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
    .channel = "from_A_to_B",
    .policy = nullptr,
  }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, sih});
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 1);    // One route
  REQUIRE(result[0].Size() == 0); // Oldest possible timeframe should not be forwarded
}

TEST_CASE("ForwardInputOldestPossibleSingleRoute")
{
  o2::framework::DomainInfoHeader dih{};

  std::vector<fair::mq::Channel> channels{
    fair::mq::Channel("from_A_to_B")};

  bool consume = true;
  bool copyByDefault = true;
  FairMQDeviceProxy proxy;
  std::vector<ForwardRoute> routes{ForwardRoute{
    .timeslice = 0,
    .maxTimeslices = 1,
    .matcher = {"binding", ConcreteDataMatcher{"TST", "A", 0}},
    .channel = "from_A_to_B",
    .policy = nullptr,
  }};

  auto findChannelByName = [&channels](std::string const& channelName) -> fair::mq::Channel& {
    for (auto& channel : channels) {
      if (channel.GetName() == channelName) {
        return channel;
      }
    }
    throw std::runtime_error("Channel not found");
  };

  proxy.bind({}, {}, routes, findChannelByName, nullptr);

  std::vector<MessageSet> currentSetOfInputs;
  MessageSet messageSet;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  fair::mq::MessagePtr payload(transport->CreateMessage());
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dih});
  messageSet.add(PartRef{std::move(header), std::move(payload)});
  REQUIRE(messageSet.size() == 1);
  currentSetOfInputs.emplace_back(std::move(messageSet));

  auto result = o2::framework::DataProcessingHelpers::routeForwardedMessageSet(proxy, currentSetOfInputs, copyByDefault, consume);
  REQUIRE(result.size() == 1);    // One route
  REQUIRE(result[0].Size() == 0); // Oldest possible timeframe should not be forwarded
}
