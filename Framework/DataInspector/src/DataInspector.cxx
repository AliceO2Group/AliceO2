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
#include "DataInspector.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/RawDeviceService.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include "boost/asio.hpp"
#include <TBufferJSON.h>
#include <boost/algorithm/string/join.hpp>
#include <arrow/table.h>
#include "Framework/TableConsumer.h"
#include "boost/archive/iterators/base64_from_binary.hpp"
#include "boost/archive/iterators/transform_width.hpp"
#include "boost/predef/other/endian.h"

using namespace rapidjson;

namespace o2::framework::data_inspector
{
#if BOOST_ENDIAN_BIG_BYTE
static const auto endianness = "BIG";
#elif BOOST_ENDIAN_LITTLE_BYTE
static const auto endianness = "LITTLE";
#else
static const auto endianness = "UNKNOWN";
#endif

inline size_t base64PaddingSize(uint64_t dataSize)
{
  return (3 - dataSize % 3) % 3;
}

std::string encode64(const char* data, uint64_t size)
{
  auto* begin = data;
  auto* end = data + size;

  using namespace boost::archive::iterators;
  using EncodingIt = base64_from_binary<transform_width<const char*, 6, 8>>;
  return std::string(EncodingIt(begin), EncodingIt(end)).append(base64PaddingSize(size), '=');
}

void addPayload(Document& message,
                uint64_t payloadSize,
                const char* payload,
                Document::AllocatorType& alloc)
{
  message.AddMember("payload", Value(encode64(payload, payloadSize).c_str(), alloc), alloc);
  message.AddMember("payloadEndianness", Value(endianness, alloc), alloc);
}

void addBasicDataHeaderInfo(Document& message, const header::DataHeader* header, Document::AllocatorType& alloc)
{
  std::string origin = header->dataOrigin.as<std::string>();
  std::string description = header->dataDescription.as<std::string>();
  std::string method = header->payloadSerializationMethod.as<std::string>();

  message.AddMember("origin", Value(origin.c_str(), alloc), alloc);
  message.AddMember("description", Value(description.c_str(), alloc), alloc);
  message.AddMember("subSpecification", Value(header->subSpecification), alloc);
  message.AddMember("firstTForbit", Value(header->firstTForbit), alloc);
  message.AddMember("tfCounter", Value(header->tfCounter), alloc);
  message.AddMember("runNumber", Value(header->runNumber), alloc);
  message.AddMember("payloadSize", Value(header->payloadSize), alloc);
  message.AddMember("splitPayloadParts", Value(header->splitPayloadParts), alloc);
  message.AddMember("payloadSerialization", Value(method.c_str(), alloc), alloc);
  message.AddMember("payloadSplitIndex", Value(header->splitPayloadIndex), alloc);
}

void addBasicDataProcessingHeaderInfo(Document& message, const DataProcessingHeader* header, Document::AllocatorType& alloc)
{
  message.AddMember("startTime", Value(header->startTime), alloc);
  message.AddMember("duration", Value(header->duration), alloc);
  message.AddMember("creationTimer", Value(header->creation), alloc);
}

void addBasicOutputObjHeaderInfo(Document& message, const OutputObjHeader* header, Document::AllocatorType& alloc)
{
  message.AddMember("taskHash", Value(header->mTaskHash), alloc);
}

void buildDocument(Document& message, std::string sender, const DataRef& ref)
{
  message.SetObject();
  Document::AllocatorType& alloc = message.GetAllocator();
  message.AddMember("sender", Value(sender.c_str(), alloc), alloc);

  const header::BaseHeader* baseHeader = header::BaseHeader::get(reinterpret_cast<const std::byte*>(ref.header));
  for (; baseHeader != nullptr; baseHeader = baseHeader->next()) {
    if (baseHeader->description == header::DataHeader::sHeaderType) {
      const auto* header = header::get<header::DataHeader*>(baseHeader->data());
      addBasicDataHeaderInfo(message, header, alloc);
      addPayload(message, header->payloadSize, ref.payload, alloc);
    } else if (baseHeader->description == DataProcessingHeader::sHeaderType) {
      const auto* header = header::get<DataProcessingHeader*>(baseHeader->data());
      addBasicDataProcessingHeaderInfo(message, header, alloc);
    } else if (baseHeader->description == OutputObjHeader::sHeaderType) {
      const auto* header = header::get<OutputObjHeader*>(baseHeader->data());
      addBasicOutputObjHeaderInfo(message, header, alloc);
    }
  }
}

/* Callback which transforms each `DataRef` to a JSON object*/
std::vector<DIMessage> serializeO2Messages(const std::vector<DataRef>& refs, const std::string& deviceName)
{
  std::vector<DIMessage> messages{};

  for (auto& ref : refs) {
    Document message;
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    buildDocument(message, deviceName, ref);
    message.Accept(writer);

    messages.emplace_back(DIMessage{DIMessage::Header::Type::DATA, std::string{buffer.GetString(), buffer.GetSize()}});
  }

  return messages;
}
} // namespace o2::framework::data_inspector
