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

/// @file   benchmark_DPLRawPageSequencer.h
/// @author Matthias Richter
/// @since  2021-07-21
/// @brief  Unit test for the DPL raw page sequencer utility

#include <benchmark/benchmark.h>

#include "DPLUtils/DPLRawPageSequencer.h"
#include "RawPageTestData.h"
#include <random>
#include <vector>

using namespace o2::framework;
auto const PAGESIZE = test::PAGESIZE;

auto createData(int nPages)
{
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
  std::vector<uint16_t> fees;
  auto nextlength = randlength();
  auto createFEEID = [&rdhCount, &fees, &nPages, &randlength, &nextlength]() {
    if (rdhCount % nPages == 0 || rdhCount - fees.back() > nextlength) {
      fees.emplace_back(rdhCount);
      nextlength = randlength();
    }
    return fees.back();
  };
  auto amendRdh = [&rdhCount, createFEEID](test::RAWDataHeader& rdh) {
    rdh.feeId = createFEEID();
    rdhCount++;
  };

  return test::createData(inputspecs, dataheaders, amendRdh);
}

static void BM_DPLRawPageSequencerBinary(benchmark::State& state)
{
  auto isSameRdh = [](const char* left, const char* right) -> bool {
    if (left == right) {
      return true;
    }

    return reinterpret_cast<test::RAWDataHeader const*>(left)->feeId == reinterpret_cast<test::RAWDataHeader const*>(right)->feeId;
  };
  std::vector<std::pair<const char*, size_t>> pages;
  auto insertPages = [&pages](const char* ptr, size_t n, uint32_t subSpec) -> void {
    pages.emplace_back(ptr, n);
  };
  auto dataset = createData(state.range(0));
  for (auto _ : state) {
    DPLRawPageSequencer(dataset.record).binary(isSameRdh, insertPages);
  }
}

static void BM_DPLRawPageSequencerForward(benchmark::State& state)
{
  auto isSameRdh = [](const char* left, const char* right) -> bool {
    if (left == right) {
      return true;
    }

    return reinterpret_cast<test::RAWDataHeader const*>(left)->feeId == reinterpret_cast<test::RAWDataHeader const*>(right)->feeId;
  };
  std::vector<std::pair<const char*, size_t>> pages;
  auto insertPages = [&pages](const char* ptr, size_t n, uint32_t subSpec) -> void {
    pages.emplace_back(ptr, n);
  };
  auto dataset = createData(state.range(0));
  for (auto _ : state) {
    DPLRawPageSequencer(dataset.record).forward(isSameRdh, insertPages);
  }
}

BENCHMARK(BM_DPLRawPageSequencerBinary)->Arg(64)->Arg(512)->Arg(1024);
BENCHMARK(BM_DPLRawPageSequencerForward)->Arg(64)->Arg(512)->Arg(1024);

BENCHMARK_MAIN();
