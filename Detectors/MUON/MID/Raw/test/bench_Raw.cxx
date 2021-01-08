// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/test/bench_Raw.cxx
/// \brief  Benchmark MID raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 March 2018

#include "benchmark/benchmark.h"
#include <vector>
#include "Framework/Logger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/RawFileReader.h"
#include "DPLUtils/RawParser.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/Encoder.h"
#include "MIDRaw/GBTDecoder.h"

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

std::vector<uint8_t> generateTestData(size_t nTF, size_t nDataInTF, size_t nColDataInEvent, size_t nLinks = o2::mid::crateparams::sNGBTs)
{
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
    auto rpcLine = o2::mid::detparams::getRPCLine(ide);
    int firstCol = (rpcLine < 3 || rpcLine > 5) ? 0 : 1;
    for (int icol = firstCol; icol < nCol; ++icol) {
      colData.emplace_back(getColData(ide, icol, 0xFF00, 0xFF0));
    }
  }

  auto severity = fair::Logger::GetConsoleSeverity();
  fair::Logger::SetConsoleSeverity(fair::Severity::WARNING);
  std::string tmpFilename = "tmp_mid_raw.raw";
  o2::mid::Encoder encoder;
  encoder.init(tmpFilename.c_str());
  std::string tmpConfigFilename = "tmp_MIDConfig.cfg";
  encoder.getWriter().writeConfFile("MID", "RAWDATA", tmpConfigFilename.c_str(), false);
  // Fill TF
  for (size_t itf = 0; itf < nTF; ++itf) {
    for (int ilocal = 0; ilocal < nDataInTF; ++ilocal) {
      o2::InteractionRecord ir(ilocal, itf);
      encoder.process(colData, ir, o2::mid::EventType::Standard);
    }
  }
  encoder.finalize();

  o2::raw::RawFileReader rawReader(tmpConfigFilename.c_str());
  rawReader.init();
  size_t nActiveLinks = rawReader.getNLinks() < nLinks ? rawReader.getNLinks() : nLinks;
  std::vector<char> buffer;
  for (size_t itf = 0; itf < rawReader.getNTimeFrames(); ++itf) {
    rawReader.setNextTFToRead(itf);
    for (size_t ilink = 0; ilink < nActiveLinks; ++ilink) {
      auto& link = rawReader.getLink(ilink);
      auto tfsz = link.getNextTFSize();
      if (!tfsz) {
        continue;
      }
      std::vector<char> linkBuffer(tfsz);
      link.readNextTF(linkBuffer.data());
      buffer.insert(buffer.end(), linkBuffer.begin(), linkBuffer.end());
    }
  }
  fair::Logger::SetConsoleSeverity(severity);

  std::remove(tmpFilename.c_str());
  std::remove(tmpConfigFilename.c_str());

  std::vector<uint8_t> data(buffer.size());
  memcpy(data.data(), buffer.data(), buffer.size());

  return data;
}

static void BM_Decoder(benchmark::State& state)
{
  o2::mid::Decoder decoder;

  int nTF = state.range(0);
  int nEventPerTF = state.range(1);
  int nFiredPerEvent = state.range(2);
  double num{0};

  auto inputData = generateTestData(nTF, nEventPerTF, nFiredPerEvent);

  for (auto _ : state) {
    decoder.process(inputData);
    ++num;
  }

  state.counters["num"] = benchmark::Counter(num, benchmark::Counter::kIsRate);
}

static void BM_GBTDecoder(benchmark::State& state)
{
  auto decoder = o2::mid::createGBTDecoder(0);

  int nTF = state.range(0);
  int nEventPerTF = state.range(1);
  int nFiredPerEvent = state.range(2);
  double num{0};

  auto inputData = generateTestData(nTF, nEventPerTF, nFiredPerEvent, 1);
  std::vector<o2::mid::LocalBoardRO> data;
  std::vector<o2::mid::ROFRecord> rofs;

  for (auto _ : state) {
    data.clear();
    rofs.clear();
    o2::framework::RawParser parser(inputData.data(), inputData.size());
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      if (it.size() == 0) {
        continue;
      }
      auto* rdhPtr = it.template get_if<o2::header::RAWDataHeader>();
      gsl::span<const uint8_t> payload(it.data(), it.size());
      decoder->process(payload, *rdhPtr, data, rofs);
    }
    ++num;
  }

  state.counters["num"] = benchmark::Counter(num, benchmark::Counter::kIsRate);
}

static void CustomArguments(benchmark::internal::Benchmark* bench)
{
  // One per event
  bench->Args({1, 1, 1});
  bench->Args({10, 1, 1});
  // One large data
  bench->Args({1, 1, 70 * 4});
  // Many small data
  bench->Args({1, 100, 4});
}

BENCHMARK(BM_GBTDecoder)->Apply(CustomArguments)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Decoder)->Apply(CustomArguments)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
