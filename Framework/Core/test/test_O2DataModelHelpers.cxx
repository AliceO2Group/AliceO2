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

#define BOOST_TEST_MODULE Test O2DataModelHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/O2DataModelHelpers.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include <fairmq/TransportFactory.h>
#include <boost/test/unit_test.hpp>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestNoWait)
{
  o2::header::DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 0;

  o2::header::DataHeader dh2;
  dh2.dataDescription = "CLUSTERS";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 0;

  DataProcessingHeader dph1{0, 1};
  DataProcessingHeader dph2{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  std::array<fair::mq::MessagePtr, 2> messages;
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::Parts inputs{
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh1, dph1}),
    transport->CreateMessage(1000),
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh2, dph2}),
    transport->CreateMessage(1000)};
  // Check if any header has dataDescription == "CLUSTERS"
  BOOST_CHECK(O2DataModelHelpers::all_headers_matching(inputs, [](auto const& header) {
    return header != nullptr && header->dataDescription == o2::header::DataDescription("CLUSTERS");
  }));

  BOOST_CHECK(O2DataModelHelpers::any_header_matching(inputs, [](auto const& header) {
    return header != nullptr && header->dataOrigin == o2::header::DataOrigin("ITS");
  }));

  BOOST_CHECK(O2DataModelHelpers::any_header_matching(inputs, [](auto const& header) {
    return header != nullptr && header->dataOrigin == o2::header::DataOrigin("TPC");
  }));

  dh2.splitPayloadParts = 2;
  o2::header::DataHeader dh3;
  dh3.dataDescription = "TRACKS";
  dh3.dataOrigin = "ITS";
  dh3.subSpecification = 0;
  dh3.splitPayloadIndex = 0;
  dh3.splitPayloadParts = 0;
  DataProcessingHeader dph3{0, 1};

  fair::mq::Parts inputs2{
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh1, dph1}),
    transport->CreateMessage(1000),
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh2, dph2}),
    transport->CreateMessage(1000),
    transport->CreateMessage(1000),
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh3, dph3}),
    transport->CreateMessage(1000),
  };

  BOOST_CHECK(O2DataModelHelpers::all_headers_matching(inputs, [](auto const& header) {
    return header != nullptr && header->dataDescription == o2::header::DataDescription("CLUSTERS");
  }));

  BOOST_CHECK(O2DataModelHelpers::any_header_matching(inputs, [](auto const& header) {
    return header != nullptr && header->dataOrigin == o2::header::DataOrigin("ITS");
  }));

  BOOST_CHECK(O2DataModelHelpers::any_header_matching(inputs, [](auto const& header) {
                return header != nullptr && header->dataDescription == o2::header::DataDescription("TRACKS");
              }) == false);

  BOOST_CHECK(O2DataModelHelpers::any_header_matching(inputs2, [](auto const& header) {
    return header != nullptr && header->dataDescription == o2::header::DataDescription("TRACKS");
  }));

  BOOST_CHECK(O2DataModelHelpers::all_headers_matching(inputs2, [](auto const& header) {
    return header != nullptr && header->subSpecification == 0;
  }));
}

// Add a test to check that all the Lifetime::Timeframe messages are
// actually there in the parts.
BOOST_AUTO_TEST_CASE(TestTimeframePresent)
{
  o2::header::DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 0;

  o2::header::DataHeader dh2;
  dh2.dataDescription = "CLUSTERS";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 0;

  DataProcessingHeader dph1{0, 1};
  DataProcessingHeader dph2{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  std::array<fair::mq::MessagePtr, 2> messages;
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::Parts inputs{
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh1, dph1}),
    transport->CreateMessage(1000),
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh2, dph2}),
    transport->CreateMessage(1000)};

  std::vector<OutputSpec> outputs{
    OutputSpec{{"TPC"}, "CLUSTERS", 0, Lifetime::Timeframe},
    OutputSpec{{"ITS"}, "CLUSTERS", 0, Lifetime::Timeframe},
  };
  std::vector<bool> present;
  present.resize(outputs.size());
  BOOST_CHECK(O2DataModelHelpers::checkForMissingSporadic(inputs, outputs, present));
}

BOOST_AUTO_TEST_CASE(TestTimeframeMissing)
{
  o2::header::DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 0;

  DataProcessingHeader dph1{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  std::array<fair::mq::MessagePtr, 2> messages;
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::Parts inputs{
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh1, dph1}),
    transport->CreateMessage(1000),
  };

  std::vector<OutputSpec> outputs{
    OutputSpec{{"TPC"}, "CLUSTERS", 0, Lifetime::Timeframe},
    OutputSpec{{"ITS"}, "CLUSTERS", 0, Lifetime::Timeframe},
  };
  std::vector<bool> present;
  present.resize(outputs.size());
  BOOST_CHECK(O2DataModelHelpers::checkForMissingSporadic(inputs, outputs, present) == false);
  BOOST_CHECK_EQUAL(O2DataModelHelpers::describeMissingOutputs(outputs, present),
                    "This timeframe has a missing output of lifetime timeframe: ITS/CLUSTERS/0. If this is expected, please change its lifetime to Sporadic / QA.");
}

BOOST_AUTO_TEST_CASE(TestTimeframeSporadic)
{
  o2::header::DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 0;

  DataProcessingHeader dph1{0, 1};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  std::array<fair::mq::MessagePtr, 2> messages;
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  fair::mq::Parts inputs{
    o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh1, dph1}),
    transport->CreateMessage(1000),
  };

  std::vector<OutputSpec> outputs{
    OutputSpec{{"TPC"}, "CLUSTERS", 0, Lifetime::Timeframe},
    OutputSpec{{"ITS"}, "QA", 0, Lifetime::Sporadic},
  };
  std::vector<bool> present;
  present.resize(outputs.size());
  BOOST_CHECK(O2DataModelHelpers::checkForMissingSporadic(inputs, outputs, present) == true);
}
