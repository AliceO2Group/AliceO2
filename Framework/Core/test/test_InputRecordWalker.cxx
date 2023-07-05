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

#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/WorkflowSpec.h" // o2::framework::select
#include "Framework/DataRefUtils.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "Headers/Stack.h"
#include <catch_amalgamated.hpp>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

// simple helper struct to keep the InputRecord and ownership of messages
struct DataSet {
  // not nice with the double vector but for quick unit test ok
  using MessageSet = std::vector<std::unique_ptr<std::vector<char>>>;
  using TaggedSet = std::pair<o2::header::DataOrigin, MessageSet>;
  using Messages = std::vector<TaggedSet>;
  using CheckType = std::vector<std::string>;
  DataSet(std::vector<InputRoute>&& s, Messages&& m, CheckType&& v, ServiceRegistryRef registry)
    : schema{std::move(s)}, messages{std::move(m)}, span{[this](size_t i, size_t part) {
                                                           REQUIRE(i < this->messages.size());
                                                           REQUIRE(part < this->messages[i].second.size() / 2);
                                                           auto header = static_cast<char const*>(this->messages[i].second.at(2 * part)->data());
                                                           auto payload = static_cast<char const*>(this->messages[i].second.at(2 * part + 1)->data());
                                                           return DataRef{nullptr, header, payload};
                                                         },
                                                         [this](size_t i) { return i < this->messages.size() ? messages[i].second.size() / 2 : 0; }, this->messages.size()},
      record{schema, span, registry},
      values{std::move(v)}
  {
    REQUIRE(messages.size() == schema.size());
  }

  std::vector<InputRoute> schema;
  Messages messages;
  InputSpan span;
  InputRecord record;
  CheckType values;
};

DataSet createData()
{
  static ServiceRegistry registry;
  // Create the routes we want for the InputRecord
  std::vector<InputSpec> inputspecs = {
    InputSpec{"tpc", "TPC", "SOMEDATA", 0, Lifetime::Timeframe},
    InputSpec{"its", ConcreteDataTypeMatcher{"ITS", "SOMEDATA"}, Lifetime::Timeframe},
    InputSpec{"tof", "TOF", "SOMEDATA", 1, Lifetime::Timeframe}};

  size_t i = 0;
  auto createRoute = [&i](const char* source, InputSpec& spec) {
    return InputRoute{
      spec,
      i++,
      source};
  };

  std::vector<InputRoute> schema = {
    createRoute("tpc_source", inputspecs[0]),
    createRoute("its_source", inputspecs[1]),
    createRoute("tof_source", inputspecs[2])};

  decltype(DataSet::values) checkValues;
  DataSet::Messages messages;

  auto createMessage = [&messages, &checkValues](DataHeader dh) {
    checkValues.emplace_back(fmt::format("{}_{}_{}", dh.dataOrigin, dh.dataDescription, dh.subSpecification));
    std::string const& data = checkValues.back();
    dh.payloadSize = data.size();
    DataProcessingHeader dph{0, 1};
    Stack stack{dh, dph};
    auto it = messages.begin(), end = messages.end();
    for (; it != end; ++it) {
      if (it->first == dh.dataOrigin) {
        break;
      }
    }
    if (it == end) {
      messages.resize(messages.size() + 1);
      it = messages.end() - 1;
      it->first = dh.dataOrigin;
    }
    auto& routemessages = it->second;
    routemessages.emplace_back(std::make_unique<std::vector<char>>(stack.size()));
    memcpy(routemessages.back()->data(), stack.data(), routemessages.back()->size());
    routemessages.emplace_back(std::make_unique<std::vector<char>>(dh.payloadSize));
    memcpy(routemessages.back()->data(), data.data(), routemessages.back()->size());
  };

  // we create message for the 3 input routes, the messages have different page size
  // and the second messages has 3 parts, each with the same page size
  // the test value is written as payload after the RDH and all values are cached for
  // later checking when parsing the data set
  DataHeader dh1;
  dh1.dataDescription = "SOMEDATA";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  DataHeader dh2;
  dh2.dataDescription = "SOMEDATA";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  dh2.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  DataHeader dh3;
  dh3.dataDescription = "SOMEDATA";
  dh3.dataOrigin = "ITS";
  dh3.subSpecification = 1;
  dh3.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  DataHeader dh4;
  dh4.dataDescription = "SOMEDATA";
  dh4.dataOrigin = "TOF";
  dh4.subSpecification = 255;
  dh4.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  createMessage(dh1);
  createMessage(dh2);
  createMessage(dh3);
  createMessage(dh4);

  return {std::move(schema), std::move(messages), std::move(checkValues), registry};
}

TEST_CASE("test_DPLRawParser")
{
  auto dataset = createData();
  InputRecord& inputs = dataset.record;
  REQUIRE(dataset.messages.size() > 0);
  REQUIRE(dataset.messages[0].second.at(0) != nullptr);
  REQUIRE(inputs.size() == 3);
  REQUIRE((*inputs.begin()).header == dataset.messages[0].second.at(0)->data());

  int count = 0;
  for (auto const& ref : InputRecordWalker(inputs)) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto const data = inputs.get<std::string>(ref);
    REQUIRE(data == dataset.values[count]);
    count++;
  }
  REQUIRE(count == 4);

  std::vector<InputSpec> filter{
    {"tpc", "TPC", "SOMEDATA", 0, Lifetime::Timeframe},
    {"its", ConcreteDataTypeMatcher{"ITS", "SOMEDATA"}, Lifetime::Timeframe},
  };

  count = 0;
  for (auto const& ref : InputRecordWalker(inputs, filter)) {
    auto const data = inputs.get<std::string>(ref);
    REQUIRE(data == dataset.values[count]);
    count++;
  }
  REQUIRE(count == 3);
}
