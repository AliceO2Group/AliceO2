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

/// @file   RawPageTestData.cxx
/// @author Matthias Richter
/// @since  2021-06-21
/// @brief  Raw page test data generator

#include "RawPageTestData.h"
#include "Headers/Stack.h"
#include <random>

namespace o2::framework::test
{

DataSet createData(std::vector<InputSpec> const& inputspecs, std::vector<DataHeader> const& dataheaders, AmendRawDataHeader amendRdh)
{
  // Create the routes we want for the InputRecord
  size_t i = 0;
  auto createRoute = [&i](std::string const& source, InputSpec const& spec) {
    return InputRoute{
      spec,
      i++,
      source};
  };

  std::vector<InputRoute> schema;
  for (auto const& spec : inputspecs) {
    auto routename = spec.binding + "_source";
    schema.emplace_back(createRoute(routename, spec));
  }

  std::random_device rd;
  std::uniform_int_distribution<> testvals(0, 42);
  auto randval = [&rd, &testvals]() {
    return testvals(rd);
  };
  std::vector<int> checkValues;
  DataSet::Messages messages;

  auto initRawPage = [&checkValues, &amendRdh](char* buffer, size_t size, auto value) {
    char* wrtptr = buffer;
    while (wrtptr < buffer + size) {
      auto* header = reinterpret_cast<RAWDataHeader*>(wrtptr);
      *header = RAWDataHeader();
      if (amendRdh) {
        amendRdh(*header);
      }
      header->offsetToNext = PAGESIZE;
      *reinterpret_cast<decltype(value)*>(wrtptr + header->headerSize) = value;
      wrtptr += PAGESIZE;
      checkValues.emplace_back(value);
      ++value;
    }
  };

  auto createMessage = [&messages, &initRawPage, &randval](DataHeader dh) {
    DataProcessingHeader dph{0, 1};
    Stack stack{dh, dph};
    if (dh.splitPayloadParts == 0 || dh.splitPayloadIndex == 0) {
      // add new message collection
      messages.emplace_back();
    }
    messages.back().emplace_back(std::make_unique<std::vector<char>>(stack.size()));
    memcpy(messages.back().back()->data(), stack.data(), messages.back().back()->size());
    messages.back().emplace_back(std::make_unique<std::vector<char>>(dh.payloadSize));
    int value = randval();
    initRawPage(messages.back().back()->data(), messages.back().back()->size(), value);
  };

  // create messages for the provided dataheaders
  for (auto header : dataheaders) {
    for (DataHeader::SplitPayloadIndexType index = 0; index == 0 || index < header.splitPayloadParts; index++) {
      header.splitPayloadIndex = index;
      createMessage(header);
    }
  }

  static ServiceRegistry registry;
  return {std::move(schema), std::move(messages), std::move(checkValues), {registry}};
}

} // namespace o2::framework
