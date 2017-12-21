// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Headers DataHeaderTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include "aliceHLTwrapper/MessageFormat.h"
#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"

namespace o2 { 
namespace alice_hlt {
  template<typename... Targs>
  void hexDump(Targs... Fargs) {
    //o2::header::hexDump(Fargs...);
  }

  using DataHeader = o2::header::DataHeader;

  BOOST_AUTO_TEST_CASE(test_createMessagesModeMultiPart)
  {
    std::cout << "Testing kOutputModeMultiPart" << std::endl;
    MessageFormat handler;
    handler.setOutputMode(MessageFormat::kOutputModeMultiPart);

    std::vector<std::string> dataFields = {
      "data1",
      "anotherDataSet"
    };

    std::vector<BlockDescriptor> dataDescriptors;
    for (auto & dataField : dataFields) {
      dataDescriptors.emplace_back((void*)dataField.c_str(), dataField.size() + 1, AliHLTComponentDataTypeInitializer("TESTDATA", "TEST"), 0);
    }

    unsigned totalPayloadSize = 0;
    for (auto & desc : dataDescriptors) {
      totalPayloadSize += desc.fSize;
      hexDump("HLT data descriptor", &desc, sizeof(desc));
      if (!desc.fPtr) continue;
      hexDump("  data payload", (char*)desc.fPtr + desc.fOffset, desc.fSize);
    }

    // testing without event info
    auto outputs = handler.createMessages(&dataDescriptors[0], dataDescriptors.size(), totalPayloadSize);
    BOOST_REQUIRE(outputs.size() == dataDescriptors.size());
    unsigned dataidx = 0;
    for (auto & output : outputs) {
      hexDump("Output block (w/o event info)", output.mP, output.mSize);
      const char* data = (char*)output.mP + sizeof(AliHLTComponentBlockData);
      BOOST_CHECK(dataFields[dataidx++] == data);
    }

    // testing with dummy event info
    AliHLTComponentEventData evtData;
    handler.clear();
    outputs.clear();
    outputs = handler.createMessages(&dataDescriptors[0], dataDescriptors.size(), totalPayloadSize, &evtData);
    dataidx = 0;
    for (auto & output : outputs) {
      std::string debugMessage = "Output block (with";
      debugMessage += (dataidx == 0?"":"out");
      debugMessage += " event info)";
      hexDump(debugMessage.c_str(), output.mP, output.mSize);
      const char* data = (char*)output.mP + sizeof(AliHLTComponentBlockData) + (dataidx == 0?sizeof(AliHLTComponentEventData):0);
      BOOST_CHECK(dataFields[dataidx++] == data);
    }
  }

  BOOST_AUTO_TEST_CASE(test_createMessagesModeSequence)
  {
    std::cout << "Testing kOutputModeSequence" << std::endl;
    MessageFormat handler;
    handler.setOutputMode(MessageFormat::kOutputModeSequence);

    std::vector<std::string> dataFields = {
      "data1",
      "anotherDataSet"
    };

    std::vector<BlockDescriptor> dataDescriptors;
    for (auto & dataField : dataFields) {
      dataDescriptors.emplace_back((void*)dataField.c_str(), dataField.size() + 1, AliHLTComponentDataTypeInitializer("TESTDATA", "TEST"), 0);
    }

    unsigned totalPayloadSize = 0;
    for (auto & desc : dataDescriptors) {
      totalPayloadSize += desc.fSize;
      hexDump("HLT data descriptor", &desc, sizeof(desc));
      if (!desc.fPtr) continue;
      hexDump("  data payload", (char*)desc.fPtr + desc.fOffset, desc.fSize);
    }

    // testing without event info
    auto outputs = handler.createMessages(&dataDescriptors[0], dataDescriptors.size(), totalPayloadSize);
    BOOST_REQUIRE(outputs.size() == 1);
    hexDump("Sequential collection", outputs[0].mP, outputs[0].mSize);
    unsigned capacity = outputs[0].mSize;
    BOOST_REQUIRE(capacity >= sizeof(AliHLTComponentBlockData));
    const AliHLTComponentBlockData* desc = reinterpret_cast<const AliHLTComponentBlockData*>(outputs[0].mP);
    for (auto & dataField : dataFields) {
      BOOST_REQUIRE(desc->fSize + sizeof(AliHLTComponentBlockData) <= capacity);
      const char* data = (const char*)desc + sizeof(AliHLTComponentBlockData);
      hexDump("Output block", data, desc->fSize);
      BOOST_CHECK(dataField == data);
      capacity -= desc->fSize + sizeof(AliHLTComponentBlockData);
      data += desc->fSize;
      desc = reinterpret_cast<const AliHLTComponentBlockData*>(data);
    }
  }

  BOOST_AUTO_TEST_CASE(test_createMessagesModeO2)
  {
    std::cout << "Testing kOutputModeO2" << std::endl;
    MessageFormat handler;
    handler.setOutputMode(MessageFormat::kOutputModeO2);

    std::vector<std::string> dataFields = {
      "data1",
      "anotherDataSet"
    };

    std::vector<BlockDescriptor> dataDescriptors;
    for (auto & dataField : dataFields) {
      dataDescriptors.emplace_back((void*)dataField.c_str(), dataField.size() + 1, AliHLTComponentDataTypeInitializer("TESTDATA", "TEST"), 0);
    }

    unsigned totalPayloadSize = 0;
    for (auto & desc : dataDescriptors) {
      totalPayloadSize += desc.fSize;
      hexDump("HLT data descriptor", &desc, sizeof(desc));
      if (!desc.fPtr) continue;
      hexDump("  data payload", (char*)desc.fPtr + desc.fOffset, desc.fSize);
    }

    // testing without event info
    auto outputs = handler.createMessages(&dataDescriptors[0], dataDescriptors.size(), totalPayloadSize);
    BOOST_REQUIRE(outputs.size() % 2 == 0);
    unsigned dataidx = 0;
    for (auto & output : outputs) {
      if (dataidx % 2 == 0) {
        hexDump("Header block", output.mP, output.mSize);
        BOOST_CHECK(output.mSize == sizeof(DataHeader));
      } else {
        hexDump("Payload block", output.mP, output.mSize);
        hexDump("  Data string", dataFields[dataidx/2].c_str(), dataFields[dataidx/2].size() + 1);
        const char* data = (char*)output.mP;
        BOOST_CHECK(dataFields[dataidx/2] == data);
      }
      ++dataidx;
    }

    std::cout << "reading O2 format" << std::endl;
    MessageFormat inputHandler;
    int result = inputHandler.addMessages(outputs);
    BOOST_REQUIRE(result == dataFields.size());

    const std::vector<BlockDescriptor>& descriptors = inputHandler.getBlockDescriptors();
    dataidx = 0;
    for (auto & desc : descriptors) {
      hexDump("Readback: HLT data descriptor", &desc, sizeof(desc));
      if (!desc.fPtr) continue;
      const char* data = (char*)desc.fPtr + desc.fOffset;
      hexDump("  data payload", data, desc.fSize);
      BOOST_CHECK(dataFields[dataidx++] == data);
    }
  }

  BOOST_AUTO_TEST_CASE(test_createHeartbeatFrame)
  {
    using HeartbeatFrameEnvelope = o2::header::HeartbeatFrameEnvelope;
    using HeartbeatHeader = o2::header::HeartbeatHeader;
    using HeartbeatTrailer = o2::header::HeartbeatTrailer;
    using HeartbeatStatistics = o2::header::HeartbeatStatistics;
    std::cout << "Testing HearbeatFrame propagation" << std::endl;
    MessageFormat handler;
    handler.setOutputMode(MessageFormat::kOutputModeO2);

    // data is wrapped into heartbeat frame if the
    // HeartbeatFrameEnvelope header is found in the incoming
    // header stack
    HeartbeatStatistics hbfPayload;
    DataHeader dh;
    dh.dataDescription = o2::header::gDataDescriptionHeartbeatFrame;
    dh.dataOrigin = o2::header::DataOrigin("TEST");
    dh.subSpecification = 0;
    dh.payloadSize = sizeof(hbfPayload);

    // create incoming header stack
    HeartbeatFrameEnvelope hbfHeader;
    o2::header::Stack headerMessage(dh, hbfHeader);

    std::vector<MessageFormat::BufferDesc_t> incomingMessages;
    incomingMessages.emplace_back((MessageFormat::BufferDesc_t::PtrT)headerMessage.data(), headerMessage.size());
    incomingMessages.emplace_back((MessageFormat::BufferDesc_t::PtrT)&hbfPayload, sizeof(hbfPayload));
    for (auto& imsg : incomingMessages) {
      hexDump("Incoming message:", imsg.mP, imsg.mSize);
    }
    handler.addMessages(incomingMessages);

    std::vector<std::string> dataFields = {
      "data1",
      "anotherDataSet"
    };

    std::vector<BlockDescriptor> dataDescriptors;
    for (auto & dataField : dataFields) {
      dataDescriptors.emplace_back((void*)dataField.c_str(), dataField.size() + 1, AliHLTComponentDataTypeInitializer("TESTDATA", "TEST"), 0);
    }

    unsigned totalPayloadSize = 0;
    for (auto & desc : dataDescriptors) {
      totalPayloadSize += desc.fSize;
      hexDump("HLT data descriptor", &desc, sizeof(desc));
      if (!desc.fPtr) continue;
      hexDump("  data payload", (char*)desc.fPtr + desc.fOffset, desc.fSize);
    }

    // testing without event info
    std::cout << "... creating messages" << std::endl;
    auto outputs = handler.createMessages(&dataDescriptors[0], dataDescriptors.size(), totalPayloadSize);
    std::cout << "... checking messages" << std::endl;
    BOOST_REQUIRE(outputs.size() % 2 == 0);
    unsigned dataidx = 0;
    unsigned datafieldidx = 0;
    for (auto & output : outputs) {
      if (dataidx % 2 == 0) {
        hexDump("Header block", output.mP, output.mSize);
        BOOST_CHECK(output.mSize >= sizeof(o2::header::DataHeader));
      } else {
        hexDump("Payload block", output.mP, output.mSize);
        hexDump("  Data string", dataFields[datafieldidx].c_str(), dataFields[datafieldidx].size() + 1);
        if (dataidx >= 2) {
          const HeartbeatHeader* hbh = reinterpret_cast<const HeartbeatHeader*>(output.mP);
          const HeartbeatTrailer* hbt = reinterpret_cast<const HeartbeatTrailer*>(output.mP + output.mSize - sizeof(HeartbeatTrailer));
          BOOST_CHECK(hbh->blockType == 1 && hbh->headerLength == 1);
          BOOST_CHECK(hbt->blockType == 5 && hbt->trailerLength == 1);
          const char* data = (char*)(output.mP + sizeof(HeartbeatHeader));
          BOOST_CHECK(dataFields[datafieldidx] == data);
          ++datafieldidx;
        }
      }
      ++dataidx;
    }

    std::cout << "... reading back messages" << std::endl;
    MessageFormat readhandler;
    readhandler.addMessages(outputs);
    const auto readbackdescriptors = readhandler.getBlockDescriptors();
    BOOST_CHECK(readbackdescriptors.size() == dataFields.size());
    datafieldidx = 0;
    for (auto readbackdesc : readbackdescriptors) {
      auto data = reinterpret_cast<const char*>(readbackdesc.fPtr);
      data += readbackdesc.fOffset;
      hexDump("Payload block", data, readbackdesc.fSize);
      BOOST_CHECK(dataFields[datafieldidx] == data);
      ++datafieldidx;
    }
  }
} // namespace alice_hlt
} // namespace o2
