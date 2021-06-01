// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Utils DPLRawParser
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DPLUtils/DPLRawParser.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Framework/WorkflowSpec.h" // o2::framework::select
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include <vector>
#include <memory>
#include <iostream>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;
using RAWDataHeaderV4 = o2::header::RAWDataHeaderV4;

static const size_t PAGESIZE = 8192;

// simple helper struct to keep the InputRecord and ownership of messages
struct DataSet {
  // not nice with the double vector but for quick unit test ok
  using Messages = std::vector<std::vector<std::unique_ptr<std::vector<char>>>>;
  DataSet(std::vector<InputRoute>&& s, Messages&& m, std::vector<int>&& v)
    : schema{std::move(s)},
      messages{std::move(m)},
      span{[this](size_t i, size_t part) {
             BOOST_REQUIRE(i < this->messages.size());
             BOOST_REQUIRE(part < this->messages[i].size() / 2);
             auto header = static_cast<char const*>(this->messages[i].at(2 * part)->data());
             auto payload = static_cast<char const*>(this->messages[i].at(2 * part + 1)->data());
             return DataRef{nullptr, header, payload};
           },
           [this](size_t i) { return i < this->messages.size() ? messages[i].size() / 2 : 0; }, this->messages.size()},
      record{schema, span},
      values{std::move(v)}
  {
    BOOST_REQUIRE(messages.size() == schema.size());
  }

  std::vector<InputRoute> schema;
  Messages messages;
  InputSpan span;
  InputRecord record;
  std::vector<int> values;
};

DataSet createData()
{
  // Create the routes we want for the InputRecord
  std::vector<InputSpec> inputspecs = {
    InputSpec{"tpc0", "TPC", "RAWDATA", 0, Lifetime::Timeframe},
    InputSpec{"its1", "ITS", "RAWDATA", 0, Lifetime::Timeframe},
    InputSpec{"its1", "ITS", "RAWDATA", 1, Lifetime::Timeframe}};

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

  std::vector<int> checkValues;
  DataSet::Messages messages;

  auto initRawPage = [&checkValues](char* buffer, size_t size, int value) {
    char* wrtptr = buffer;
    while (wrtptr < buffer + size) {
      auto* header = reinterpret_cast<RAWDataHeaderV4*>(wrtptr);
      *header = RAWDataHeaderV4();
      header->offsetToNext = PAGESIZE;
      *reinterpret_cast<decltype(value)*>(wrtptr + header->headerSize) = value;
      wrtptr += PAGESIZE;
      checkValues.emplace_back(value);
      ++value;
    }
  };

  auto createMessage = [&messages, &initRawPage](DataHeader dh, int value) {
    DataProcessingHeader dph{0, 1};
    Stack stack{dh, dph};
    if (dh.splitPayloadParts == 0 || dh.splitPayloadIndex == 0) {
      // add new message collection
      messages.emplace_back();
    }
    messages.back().emplace_back(std::make_unique<std::vector<char>>(stack.size()));
    memcpy(messages.back().back()->data(), stack.data(), messages.back().back()->size());
    messages.back().emplace_back(std::make_unique<std::vector<char>>(dh.payloadSize));
    initRawPage(messages.back().back()->data(), messages.back().back()->size(), value);
  };

  // we create message for the 3 input routes, the messages have different page size
  // and the second messages has 3 parts, each with the same page size
  // the test value is written as payload after the RDH and all values are cached for
  // later checking when parsing the data set
  DataHeader dh1;
  dh1.dataDescription = "RAWDATA";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  dh1.payloadSize = 5 * PAGESIZE;
  DataHeader dh2;
  dh2.dataDescription = "RAWDATA";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  dh2.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  dh2.payloadSize = 3 * PAGESIZE;
  dh2.splitPayloadParts = 3;
  dh2.splitPayloadIndex = 0;
  DataHeader dh3;
  dh3.dataDescription = "RAWDATA";
  dh3.dataOrigin = "ITS";
  dh3.subSpecification = 1;
  dh3.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  dh3.payloadSize = 4 * PAGESIZE;
  createMessage(dh1, 10);
  createMessage(dh2, 20);
  dh2.splitPayloadIndex++;
  createMessage(dh2, 23);
  dh2.splitPayloadIndex++;
  createMessage(dh2, 26);
  createMessage(dh3, 30);

  return {std::move(schema), std::move(messages), std::move(checkValues)};
}

BOOST_AUTO_TEST_CASE(test_DPLRawParser)
{
  auto dataset = createData();
  InputRecord& inputs = dataset.record;
  BOOST_REQUIRE(dataset.messages.size() > 0);
  BOOST_REQUIRE(dataset.messages[0].at(0) != nullptr);
  BOOST_REQUIRE(inputs.size() > 0);
  BOOST_CHECK((*inputs.begin()).header == dataset.messages[0].at(0)->data());
  DPLRawParser parser(inputs);
  int count = 0;
  o2::header::DataHeader const* last = nullptr;
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it, ++count) {
    LOG(INFO) << "data " << count << " " << *((int*)it.data());
    // now check the iterator API
    // retrieving RDH v4
    auto const* rdh = it.get_if<o2::header::RAWDataHeaderV4>();
    // retrieving the raw pointer of the page
    auto const* raw = it.raw();
    // retrieving payload pointer of the page
    auto const* payload = it.data();
    // size of payload
    size_t payloadSize = it.size();
    // offset of payload in the raw page
    size_t offset = it.offset();
    BOOST_REQUIRE(rdh != nullptr);
    BOOST_REQUIRE(offset == sizeof(o2::header::RAWDataHeaderV4));
    BOOST_REQUIRE(payload == raw + offset);
    BOOST_REQUIRE(*reinterpret_cast<int const*>(payload) == dataset.values[count]);
    BOOST_REQUIRE(payloadSize == PAGESIZE - sizeof(o2::header::RAWDataHeaderV4));
    auto const* dh = it.o2DataHeader();
    if (last != dh) {
      // this is a special wrapper to print the RDU info and table header, this will
      // be extended
      std::cout << DPLRawParser::RDHInfo(it) << std::endl;
      last = dh;
    }
    std::cout << it << " payload size " << it.size() << std::endl;
  }

  // test the parser with filter on data specs, this will filter out the first input
  // route with 5 raw pages in the payload, so we start checking at count 5
  DPLRawParser filteredparser(inputs, o2::framework::select("its:ITS/RAWDATA"));
  count = 5;
  for (auto it = filteredparser.begin(), end = filteredparser.end(); it != end; ++it, ++count) {
    LOG(INFO) << "data " << count << " " << *((int*)it.data());
    BOOST_REQUIRE(*reinterpret_cast<int const*>(it.data()) == dataset.values[count]);
  }

  // test with filter not matching any input route
  DPLRawParser nomatchingparser(inputs, o2::framework::select("nmatch:NO/MATCH"));
  count = 0;
  for (auto it = nomatchingparser.begin(), end = nomatchingparser.end(); it != end; ++it, ++count) {
    LOG(INFO) << "data " << count << " " << *((int*)it.data());
  }
  BOOST_CHECK(count == 0);
}
