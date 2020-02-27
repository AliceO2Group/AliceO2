// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Tracking/test/bench_Tracker.cxx
/// \brief  Benchmark tracker device for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 March 2018

#include "benchmark/benchmark.h"
#include <random>
#include <array>
#include <vector>
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/Encoder.h"
#include "MIDRaw/RawUnit.h"

o2::mid::ColumnData getColData(uint8_t deId, uint8_t columnId, uint16_t nbp = 0, uint16_t bp1 = 0, uint16_t bp2 = 0, uint16_t bp3 = 0, uint16_t bp4 = 0)
{
  o2::mid::ColumnData col;
  col.deId = deId;
  col.columnId = columnId;
  col.setNonBendPattern(nbp);
  col.setBendPattern(bp1, 0);
  col.setBendPattern(bp2, 1);
  col.setBendPattern(bp3, 2);
  col.setBendPattern(bp4, 3);
  return col;
}

std::vector<o2::mid::raw::RawUnit> generateTestData(size_t nTF, size_t nDataInTF, size_t nColDataInEvent, o2::mid::Encoder& encoder)
{
  encoder.clear();
  std::vector<o2::mid::ColumnData> colData;
  colData.reserve(nColDataInEvent);
  int maxNcols = 7;
  int nDEs = nColDataInEvent / maxNcols;
  int nColLast = nColDataInEvent % maxNcols;
  if (nColLast > 0) {
    ++nDEs;
  }
  // Generate data
  for (int ide = 0; ide < nDEs; ++ide) {
    int nCol = (ide == nDEs - 1) ? nColLast : maxNcols;
    for (int icol = 0; icol < nCol; ++icol) {
      colData.emplace_back(getColData(ide, icol, 0xFF00, 0xFFFF));
    }
  }
  // Fill TF
  for (size_t itf = 0; itf < nTF; ++itf) {
    colData.clear();
    for (int ilocal = 0; ilocal < nDataInTF; ++ilocal) {
      o2::InteractionRecord ir(ilocal, itf);
      encoder.process(colData, ir, o2::mid::EventType::Standard);
    }
  }

  return encoder.getBuffer();
}

static void BM_Decoder(benchmark::State& state)
{
  o2::mid::Encoder encoder;
  o2::mid::Decoder decoder;

  int nTF = state.range(0);
  int nEventPerTF = state.range(1);
  int nFiredPerEvent = state.range(2);
  double num{0};

  auto inputData = generateTestData(nTF, nEventPerTF, nFiredPerEvent, encoder);

  for (auto _ : state) {
    decoder.process(inputData);

    ++num;
  }

  state.counters["num"] = benchmark::Counter(num, benchmark::Counter::kIsRate);
}

static void CustomArguments(benchmark::internal::Benchmark* bench)
{
  // Empty headers
  bench->Args({1, 0, 0});
  bench->Args({10, 0, 0});
  // One per event
  bench->Args({1, 1, 1});
  bench->Args({10, 1, 1});
  // One large data
  bench->Args({1, 1, 70 * 4});
  // Many small data
  bench->Args({1, 100, 4});
}

BENCHMARK(BM_Decoder)->Apply(CustomArguments)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
