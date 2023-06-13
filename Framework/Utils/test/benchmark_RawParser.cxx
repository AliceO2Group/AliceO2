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
#include <benchmark/benchmark.h>

#include "DPLUtils/RawParser.h"

using namespace o2::framework;

class TestPages
{
 public:
  using V4 = o2::header::RAWDataHeaderV4;
  static constexpr size_t MaxNPages = 256 * 1024;
  static constexpr size_t PageSize = 8192;
  static constexpr size_t PageDataSize = PageSize - sizeof(V4);
  struct RawPage {
    V4 rdh;
    char data[PageDataSize];
  };
  static_assert(sizeof(RawPage) == PageSize);

  TestPages()
    : mPages(MaxNPages)
  {
    for (int pageNo = 0; pageNo < mPages.size(); pageNo++) {
      mPages[pageNo].rdh.version = 4;
      mPages[pageNo].rdh.headerSize = sizeof(V4);
      mPages[pageNo].rdh.offsetToNext = PageSize;
      auto* data = reinterpret_cast<size_t*>(&mPages[pageNo].data);
      *data = pageNo;
    }
  }

  const char* data() const
  {
    return reinterpret_cast<const char*>(mPages.data());
  }

  size_t size() const
  {
    return mPages.size() * sizeof(decltype(mPages)::value_type);
  }

 private:
  std::vector<RawPage> mPages;
};

TestPages gPages;

static void BM_RawParserAuto(benchmark::State& state)
{
  size_t nofPages = state.range(0);
  if (nofPages > TestPages::MaxNPages) {
    return;
  }
  using Parser = RawParser<TestPages::PageSize>;
  Parser parser(reinterpret_cast<const char*>(gPages.data()), nofPages * TestPages::PageSize);
  size_t count = 0;
  auto processor = [&count](auto data, size_t length) {
    count++;
  };
  for (auto _ : state) {
    parser.parse(processor);
  }
}

static void BM_RawParserV4(benchmark::State& state)
{
  size_t nofPages = state.range(0);
  if (nofPages > TestPages::MaxNPages) {
    return;
  }
  using Parser = raw_parser::ConcreteRawParser<TestPages::V4, TestPages::PageSize, true>;
  Parser parser(reinterpret_cast<const char*>(gPages.data()), nofPages * TestPages::PageSize);
  size_t count = 0;
  auto processor = [&count](auto data, size_t length) {
    count++;
  };
  for (auto _ : state) {
    parser.parse(processor);
  }
}

BENCHMARK(BM_RawParserV4)->Arg(1)->Arg(8)->Arg(256)->Arg(1024)->Arg(16 * 1024)->Arg(256 * 1024);
BENCHMARK(BM_RawParserAuto)->Arg(1)->Arg(8)->Arg(256)->Arg(1024)->Arg(16 * 1024)->Arg(256 * 1024);

BENCHMARK_MAIN();
