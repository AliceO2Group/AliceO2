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

#include <catch_amalgamated.hpp>
#include "DPLUtils/DPLRawParser.h"
#include "RawPageTestData.h"
#include "Framework/InputRecord.h"
#include "Framework/WorkflowSpec.h" // o2::framework::select
#include "Headers/DataHeader.h"
#include <vector>
#include <memory>
#include <iostream>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
auto const PAGESIZE = test::PAGESIZE;
using DataSet = test::DataSet;

DataSet createData()
{
  std::vector<InputSpec> inputspecs = {
    InputSpec{"tpc0", "TPC", "RAWDATA", 0, Lifetime::Timeframe},
    InputSpec{"its1", "ITS", "RAWDATA", 0, Lifetime::Timeframe},
    InputSpec{"its1", "ITS", "RAWDATA", 1, Lifetime::Timeframe}};

  // we create message for the 3 input routes, the messages have different page size
  // and the second messages has 3 parts, each with the same page size
  // the test value is written as payload after the RDH and all values are cached for
  // later checking when parsing the data set
  std::vector<DataHeader> dataheaders;
  dataheaders.emplace_back("RAWDATA", "TPC", 0, 5 * PAGESIZE);
  dataheaders.emplace_back("RAWDATA", "ITS", 0, 3 * PAGESIZE, 0, 3);
  dataheaders.emplace_back("RAWDATA", "ITS", 1, 4 * PAGESIZE);

  return test::createData(inputspecs, dataheaders);
}

TEST_CASE("test_DPLRawParser")
{
  auto dataset = createData();
  InputRecord& inputs = dataset.record;
  REQUIRE(dataset.messages.size() > 0);
  REQUIRE(dataset.messages[0].at(0) != nullptr);
  REQUIRE(inputs.size() > 0);
  REQUIRE((*inputs.begin()).header == dataset.messages[0].at(0)->data());
  DPLRawParser parser(inputs);
  int count = 0;
  o2::header::DataHeader const* last = nullptr;
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it, ++count) {
    LOG(info) << "data " << count << " " << *((int*)it.data());
    // now check the iterator API
    // retrieving RDH
    auto const* rdh = it.get_if<test::RAWDataHeader>();
    // retrieving the raw pointer of the page
    auto const* raw = it.raw();
    // retrieving payload pointer of the page
    auto const* payload = it.data();
    // size of payload
    size_t payloadSize = it.size();
    // offset of payload in the raw page
    size_t offset = it.offset();
    REQUIRE(rdh != nullptr);
    REQUIRE(offset == sizeof(test::RAWDataHeader));
    REQUIRE(payload == raw + offset);
    REQUIRE(*reinterpret_cast<int const*>(payload) == dataset.values[count]);
    REQUIRE(payloadSize == PAGESIZE - sizeof(test::RAWDataHeader));
    auto const* dh = it.o2DataHeader();
    if (last != dh) {
      // this is a special wrapper to print the RDU info and table header, this will
      // be extended
      INFO(DPLRawParser::RDHInfo(it));
      last = dh;
    }
    INFO(it << " payload size " << it.size());
  }

  // test the parser with filter on data specs, this will filter out the first input
  // route with 5 raw pages in the payload, so we start checking at count 5
  DPLRawParser filteredparser(inputs, o2::framework::select("its:ITS/RAWDATA"));
  count = 5;
  for (auto it = filteredparser.begin(), end = filteredparser.end(); it != end; ++it, ++count) {
    LOG(info) << "data " << count << " " << *((int*)it.data());
    REQUIRE(*reinterpret_cast<int const*>(it.data()) == dataset.values[count]);
  }

  // test with filter not matching any input route
  DPLRawParser nomatchingparser(inputs, o2::framework::select("nmatch:NO/MATCH"));
  count = 0;
  for (auto it = nomatchingparser.begin(), end = nomatchingparser.end(); it != end; ++it, ++count) {
    LOG(info) << "data " << count << " " << *((int*)it.data());
  }
  REQUIRE(count == 0);
}
