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

/// @file   test_DPLRawPageSequencer.h
/// @author Matthias Richter
/// @since  2021-07-09
/// @brief  Unit test for the DPL raw page sequencer utility

#define BOOST_TEST_MODULE Test Framework Utils DPLRawPageSequencer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DPLUtils/DPLRawPageSequencer.h"
#include "RawPageTestData.h"
#include "Framework/InputRecord.h"
#include "Headers/DataHeader.h"
#include <vector>
#include <memory>
#include <iostream>
#include <random>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
auto const PAGESIZE = test::PAGESIZE;

BOOST_AUTO_TEST_CASE(test_DPLRawPageSequencer)
{
  const int nPages = 64;
  const int nParts = 16;
  std::vector<InputSpec> inputspecs = {
    InputSpec{"tpc", "TPC", "RAWDATA", 0, Lifetime::Timeframe}};

  std::vector<DataHeader> dataheaders;
  dataheaders.emplace_back("RAWDATA", "TPC", 0, nPages * PAGESIZE, 0, nParts);

  std::random_device rd;
  std::uniform_int_distribution<> lengthDist(1, nPages);
  auto randlength = [&rd, &lengthDist]() {
    return lengthDist(rd);
  };

  int rdhCount = 0;
  // whenever a new id is created, it is done from the current counter
  // position, so we also have the possibility to calculate the length
  std::vector<uint16_t> feeids;
  auto nextlength = randlength();
  auto createFEEID = [&rdhCount, &feeids, &nPages, &randlength, &nextlength]() {
    if (rdhCount % nPages == 0 || rdhCount - feeids.back() > nextlength) {
      feeids.emplace_back(rdhCount);
      nextlength = randlength();
    }
    return feeids.back();
  };
  auto amendRdh = [&rdhCount, createFEEID](test::RAWDataHeader& rdh) {
    rdh.feeId = createFEEID();
    rdhCount++;
  };

  auto dataset = test::createData(inputspecs, dataheaders, amendRdh);
  InputRecord& inputs = dataset.record;
  BOOST_REQUIRE(dataset.messages.size() > 0);
  BOOST_REQUIRE(dataset.messages[0].at(0) != nullptr);
  BOOST_REQUIRE(inputs.size() > 0);
  BOOST_CHECK((*inputs.begin()).header == dataset.messages[0].at(0)->data());
  BOOST_REQUIRE(rdhCount == nPages * nParts);
  DPLRawPageSequencer parser(inputs);

  auto isSameRdh = [](const char* left, const char* right) -> bool {
    if (left == right) {
      return true;
    }
    if (left == nullptr || right == nullptr) {
      return true;
    }

    return reinterpret_cast<test::RAWDataHeader const*>(left)->feeId == reinterpret_cast<test::RAWDataHeader const*>(right)->feeId;
  };
  std::vector<std::pair<const char*, size_t>> pages;
  auto insertPages = [&pages](const char* ptr, size_t n, uint32_t subSpec) -> void {
    pages.emplace_back(ptr, n);
  };
  parser(isSameRdh, insertPages);

  // a second parsing step based on forward search
  std::vector<std::pair<const char*, size_t>> pagesByForwardSearch;
  auto insertForwardPages = [&pagesByForwardSearch](const char* ptr, size_t n, uint32_t subSpec) -> void {
    pagesByForwardSearch.emplace_back(ptr, n);
  };
  DPLRawPageSequencer(inputs).forward(isSameRdh, insertForwardPages);

  LOG(info) << "called RDH amend: " << rdhCount;
  LOG(info) << "created " << feeids.size() << " id(s), got " << pages.size() << " page(s)";
  BOOST_REQUIRE(pages.size() == feeids.size());
  BOOST_REQUIRE(pages.size() == pagesByForwardSearch.size());

  feeids.emplace_back(rdhCount);
  auto lastId = feeids.front();
  for (auto i = 0; i < pages.size(); i++) {
    auto length = feeids[i + 1] - feeids[i];
    BOOST_CHECK_MESSAGE(pages[i].second == length, "sequence " << i << " at " << feeids[i] << " length " << length << ": got " << pages[i].second);
    BOOST_CHECK_MESSAGE(pages[i].first == pagesByForwardSearch[i].first && pages[i].second == pagesByForwardSearch[i].second,
                        "mismatch with forward search at sequence " << i
                                                                    << " [" << (void*)pages[i].first << "," << (void*)pagesByForwardSearch[i].first << "]"
                                                                    << " [" << pages[i].second << "," << pagesByForwardSearch[i].second << "]");
  }
}
