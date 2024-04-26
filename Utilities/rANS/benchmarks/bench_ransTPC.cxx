// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   bench_ransTPC.h
/// @author Michael Lettrich
/// @brief  main benchmark executable that performs compression/decompression of TPC data while taking detailed metrics.

#include <vector>
#include <cstring>
#include <execution>
#include <iterator>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/mp11.hpp>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <fairlogger/Logger.h>

#include "rANS/factory.h"
#include "rANS/histogram.h"
#include "rANS/serialize.h"

#include "helpers.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

namespace bpo = boost::program_options;
using namespace o2::rans;

using coder_types = boost::mp11::mp_list<std::integral_constant<CoderTag, CoderTag::Compat>
#ifdef RANS_SINGLE_STREAM
                                         ,
                                         std::integral_constant<CoderTag, CoderTag::SingleStream>
#endif /* RANS_SINGLE_STREAM */
#ifdef RANS_SSE
                                         ,
                                         std::integral_constant<CoderTag, CoderTag::SSE>
#endif /* RANS_SSE */
#ifdef RANS_AVX2
                                         ,
                                         std::integral_constant<CoderTag, CoderTag::AVX2>
#endif /* RANS_AVX2 */
                                         >;

// using coder_types = boost::mp11::mp_list<std::integral_constant<CoderTag, CoderTag::SingleStream>>;

std::string toString(CoderTag tag)
{
  switch (tag) {
    case CoderTag::Compat:
      return {"Compat"};
      break;
    case CoderTag::SingleStream:
      return {"SingleStream"};
      break;
    case CoderTag::SSE:
      return {"SSE"};
      break;
    case CoderTag::AVX2:
      return {"AVX2"};
      break;
    default:
      throw Exception("Invalid");
      break;
  };
};

class TimingDecorator
{
 public:
  using jsonWriter_type = rapidjson::Writer<rapidjson::OStreamWrapper>;

  TimingDecorator() = default;
  TimingDecorator(jsonWriter_type& writer) : mWriter{&writer} {};

  template <typename F, std::enable_if_t<!std::is_void_v<std::invoke_result_t<F>>, bool> = true>
  decltype(auto) timeAndLog(const std::string& keyName, const std::string& logMessage, F functor)
  {
    mTimer.start();
    decltype(auto) ret = functor();
    mTimer.stop();
    mWriter->Key(keyName.c_str());
    const double_t msDuration = mTimer.getDurationMS();
    mWriter->Double(msDuration);
    LOGP(info, "{} in {} ms", logMessage, msDuration);

    return ret;
  };

  template <typename F, std::enable_if_t<std::is_void_v<std::invoke_result_t<F>>, bool> = true>
  void timeAndLog(const std::string& keyName, const std::string& logMessage, F functor)
  {
    mTimer.start();
    functor();
    mTimer.stop();
    mWriter->Key(keyName.c_str());
    const double_t msDuration = mTimer.getDurationMS();
    mWriter->Double(msDuration);
    LOGP(info, "{} in {} ms", logMessage, msDuration);
  };

 private:
  jsonWriter_type* mWriter{};
  utils::RANSTimer mTimer{};
};

// std::ofstream ofFrequencies{"frequencies.json"};
// rapidjson::OStreamWrapper streamFrequencies{ofFrequencies};
// rapidjson::Writer<rapidjson::OStreamWrapper> writerFrequencies{streamFrequencies};

// std::ofstream ofRenormed{"renormed.json"};
// rapidjson::OStreamWrapper streamRenormed{ofRenormed};
// rapidjson::Writer<rapidjson::OStreamWrapper> writerRenormed{streamRenormed};

template <typename source_T, CoderTag coderTag_V>
void ransEncodeDecode(const std::string& name, const std::vector<source_T>& inputData, rapidjson::Writer<rapidjson::OStreamWrapper>& writer)
{
  using source_type = source_T;
  utils::RANSTimer timer{};
  TimingDecorator t{writer};

  writer.Key(name.c_str());
  writer.StartObject();

  EncodeBuffer<source_type> encodeBuffer{inputData.size()};
  encodeBuffer.literals.resize(inputData.size(), 0);
  encodeBuffer.literalsEnd = encodeBuffer.literals.data();
  DecodeBuffer<source_type> decodeBuffer{inputData.size()};

  writer.Key("Timing");
  writer.StartObject();

  LOGP(info, "processing: {} (nItems: {}, size: {} MiB)", name, inputData.size(), inputData.size() * sizeof(source_type) / 1024.0 / 1024.0);
  auto histogram = t.timeAndLog(
    "FrequencyTable", "Built Frequency Table", [&]() { return makeDenseHistogram::fromSamples(gsl::span<const source_type>(inputData)); });

  // writerFrequencies.Key(name.c_str());
  // toJSON(histogram, writerFrequencies);

  auto tmpHist = histogram;
  Metrics<source_type> metrics{};
  RenormedDenseHistogram<source_type> renormedHistogram{};
  t.timeAndLog("Renorming", "Renormed Frequency Table", [&]() mutable {
    metrics = Metrics<source_type>{histogram};
    renormedHistogram = renorm(std::move(tmpHist), metrics);
  });
  // writerRenormed.Key(name.c_str());
  // toJSON(renormedFrequencyTable, writerRenormed);

  auto encoder = t.timeAndLog("Encoder", "Built Encoder", [&]() { return makeDenseEncoder<coderTag_V>::fromRenormed(renormedHistogram); });

  t.timeAndLog("Encoding", "Encoded", [&]() mutable {
#ifdef ENABLE_VTUNE_PROFILER
    __itt_resume();
#endif
    if (renormedHistogram.hasIncompressibleSymbol()) {
      std::tie(encodeBuffer.encodeBufferEnd, encodeBuffer.literalsEnd) = encoder.process(inputData.data(), inputData.data() + inputData.size(), encodeBuffer.buffer.data(), encodeBuffer.literalsEnd);
    } else {
      encodeBuffer.encodeBufferEnd = encoder.process(inputData.data(), inputData.data() + inputData.size(), encodeBuffer.buffer.data());
    }
#ifdef ENABLE_VTUNE_PROFILER
    __itt_pause();
#endif
  });
  LOGP(info, "Encoded {} Bytes", inputData.size() * sizeof(source_type));

  std::vector<uint8_t> dict(histogram.size() * sizeof(uint64_t), 0);
  auto dictEnd = t.timeAndLog("WriteDict", "Serialized Dict", [&]() { return compressRenormedDictionary(encoder.getSymbolTable(), dict.data()); });
  LOGP(info, "Serialized Dict of {} Bytes", std::distance(dict.data(), dictEnd));
  auto recoveredHistogram = t.timeAndLog("ReadDict", "Read Dict", [&]() {
    const source_type min =  encoder.getSymbolTable().getOffset();
    const source_type max =  min + std::max<source_type>(static_cast<int64_t>(encoder.getSymbolTable().size())-1,0);
    return readRenormedDictionary(dict.data(), dictEnd,min,max,renormedHistogram.getRenormingBits()); });
  auto decoder = makeDecoder<>::fromRenormed(renormedHistogram);
  auto recoveredDecoder = makeDecoder<>::fromRenormed(recoveredHistogram);

  // if (!(std::equal(decoder.getSymbolTable().begin(), decoder.getSymbolTable().end(), recoveredDecoder.getSymbolTable().begin()) &&
  //       (decoder.getSymbolTable().getEscapeSymbol() == recoveredDecoder.getSymbolTable().getEscapeSymbol()))) {
  //   LOGP(warning, "Missmatch between original and decoded Dictionary");
  // }

  if (encodeBuffer.literalsEnd == encodeBuffer.literals.data()) {
    decoder.process(encodeBuffer.encodeBufferEnd, decodeBuffer.buffer.data(), inputData.size(), encoder.getNStreams());
  } else {
    decoder.process(encodeBuffer.encodeBufferEnd, decodeBuffer.buffer.data(), inputData.size(), encoder.getNStreams(), encodeBuffer.literalsEnd);
  }

  if (!(decodeBuffer == inputData)) {
    LOGP(warning, "Missmatch between original and decoded Message");
  }
  LOG(info) << "finished: " << name;

  writer.EndObject(); // Timing

  const auto& datasetProperties = metrics.getDatasetProperties();

  // Frequency Table
  // ##########################
  writer.Key("FrequencyTable");
  writer.StartObject();
  writer.Key("nSamples");
  writer.Uint64(histogram.getNumSamples());
  writer.Key("Min");
  writer.Int(datasetProperties.min);
  writer.Key("Max");
  writer.Int(datasetProperties.max);
  writer.Key("alphabetRangeBits");
  writer.Int(datasetProperties.alphabetRangeBits);
  writer.Key("nUsedAlphabetSymbols");
  writer.Uint(datasetProperties.nUsedAlphabetSymbols);
  writer.Key("IncompressibleFrequency");
  writer.Uint(0);
  writer.EndObject(); // FrequencyTable

  // RescaledFrequencies
  //##########################
  const Metrics<source_type> renormedMetrics{histogram};
  const auto& renormedDatasetProperties = renormedMetrics.getDatasetProperties();

  writer.Key("RescaledFrequencies");
  writer.StartObject();
  writer.Key("nSamples");
  writer.Uint64(renormedHistogram.getNumSamples());
  writer.Key("Min");
  writer.Int(renormedDatasetProperties.min);
  writer.Key("Max");
  writer.Int(renormedDatasetProperties.max);
  writer.Key("alphabetRangeBits");
  writer.Int(renormedDatasetProperties.alphabetRangeBits);
  writer.Key("nUsedAlphabetSymbols");
  writer.Uint(renormedDatasetProperties.nUsedAlphabetSymbols);
  writer.Key("IncompressibleFrequency");
  writer.Uint(renormedHistogram.getIncompressibleSymbolFrequency());
  writer.Key("RenormingBits");
  writer.Uint(renormedHistogram.getRenormingBits());
  writer.EndObject(); // RescaledFrequencies

  // Message Properties
  //##########################
  writer.Key("Message");
  writer.StartObject();
  writer.Key("Size");
  writer.Uint64(inputData.size());
  writer.Key("SymbolSize");
  writer.Uint(sizeof(source_type));
  writer.Key("Entropy");
  writer.Double(datasetProperties.entropy);
  writer.Key("ExpectedCodewordLength");
  writer.Double(computeExpectedCodewordLength<>(histogram, renormedHistogram));
  writer.EndObject(); // Message

  // Compression Properties
  //##########################
  writer.Key("Compression");
  writer.StartObject();
  writer.Key("EncodeBufferSize");
  writer.Uint64(std::distance(encodeBuffer.buffer.data(), encodeBuffer.encodeBufferEnd) * sizeof(uint32_t));
  writer.Key("LiteralSize");
  writer.Uint64(std::distance(encodeBuffer.literals.data(), encodeBuffer.literalsEnd) * sizeof(source_type));
  writer.Key("DictSize");
  writer.Uint64(std::distance(dict.data(), dictEnd));
  writer.EndObject(); // Compression

  writer.EndObject(); // Encode/Decode Run
};

template <CoderTag coderTag_V>
void encodeTPC(const std::string& name, const TPCCompressedClusters& compressedClusters, bool mergeColumns, rapidjson::Writer<rapidjson::OStreamWrapper>& writer)
{
  writer.Key(name.c_str());
  writer.StartObject();
  ransEncodeDecode<uint16_t, coderTag_V>("qTotA", compressedClusters.qTotA, writer);
  ransEncodeDecode<uint16_t, coderTag_V>("qMaxA", compressedClusters.qMaxA, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("flagsA", compressedClusters.flagsA, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("rowDiffA", compressedClusters.rowDiffA, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("sliceLegDiffA", compressedClusters.sliceLegDiffA, writer);
  ransEncodeDecode<uint16_t, coderTag_V>("padResA", compressedClusters.padResA, writer);
  ransEncodeDecode<uint32_t, coderTag_V>("timeResA", compressedClusters.timeResA, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("sigmaPadA", compressedClusters.sigmaPadA, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("sigmaTimeA", compressedClusters.sigmaTimeA, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("qPtA", compressedClusters.qPtA, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("rowA", compressedClusters.rowA, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("sliceA", compressedClusters.sliceA, writer);
  ransEncodeDecode<uint32_t, coderTag_V>("timeA", compressedClusters.timeA, writer);
  ransEncodeDecode<uint16_t, coderTag_V>("padA", compressedClusters.padA, writer);
  ransEncodeDecode<uint16_t, coderTag_V>("qTotU", compressedClusters.qTotU, writer);
  ransEncodeDecode<uint16_t, coderTag_V>("qMaxU", compressedClusters.qMaxU, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("flagsU", compressedClusters.flagsU, writer);
  ransEncodeDecode<uint16_t, coderTag_V>("padDiffU", compressedClusters.padDiffU, writer);
  ransEncodeDecode<uint32_t, coderTag_V>("timeDiffU", compressedClusters.timeDiffU, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("sigmaPadU", compressedClusters.sigmaPadU, writer);
  ransEncodeDecode<uint8_t, coderTag_V>("sigmaTimeU", compressedClusters.sigmaTimeU, writer);
  ransEncodeDecode<uint16_t, coderTag_V>("nTrackClusters", compressedClusters.nTrackClusters, writer);
  ransEncodeDecode<uint32_t, coderTag_V>("nSliceRowClusters", compressedClusters.nSliceRowClusters, writer);

  writer.EndObject();
};

using encoder_types = boost::mp11::mp_product<boost::mp11::mp_list, coder_types>;

int main(int argc, char* argv[])
{
  bpo::options_description options("Allowed options");
  // clang-format off
  options.add_options()
    ("help,h", "print usage message")
    ("in,i",bpo::value<std::string>(), "file to process")
    ("out,o",bpo::value<std::string>(), "json output file")
    ("mode,m",bpo::value<std::string>(), "compressor processing mode")
    ("log_severity,l",bpo::value<std::string>(), "severity of FairLogger");
  // clang-format on

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(argc, argv, options), vm);
  bpo::notify(vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  const std::string inFile = [&]() {
    if (vm.count("in")) {
      return vm["in"].as<std::string>();
    } else {
      LOG(error) << "missing path to input file";
      exit(1);
    }
  }();

  const std::string outFile = [&]() {
    if (vm.count("out")) {
      return vm["out"].as<std::string>();
    } else {
      return std::string("out.json");
    }
  }();

  if (vm.count("log_severity")) {
    fair::Logger::SetConsoleSeverity(vm["log_severity"].as<std::string>().c_str());
  }

  std::ofstream of{outFile};
  if (!of) {
    std::runtime_error(fmt::format("could not open output file at path {}", inFile));
  }

  // writerFrequencies.StartObject();
  // writerRenormed.StartObject();

  rapidjson::OStreamWrapper stream{of};
  rapidjson::Writer<rapidjson::OStreamWrapper> writer{stream};
  writer.StartObject();

  TPCCompressedClusters compressedClusters = readFile(inFile);
  LOG(info) << "loaded Compressed Clusters from file";
  LOG(info) << "######################################################";
  boost::mp11::mp_for_each<encoder_types>([&](auto L) {
    using coder_type = boost::mp11::mp_at_c<decltype(L), 0>;
    constexpr CoderTag coderTag = coder_type::value;
    const std::string encoderTitle = toString(coderTag);

    LOGP(info, "start rANS {}/Decode", encoderTitle);
    encodeTPC<coderTag>(encoderTitle, compressedClusters, false, writer);
    LOG(info) << "######################################################";
  });
  writer.EndObject();
  stream.Flush();
  of.close();

  // writerFrequencies.EndObject();
  // writerFrequencies.Flush();
  // ofFrequencies.close();
  // writerRenormed.EndObject();
  // writerRenormed.Flush();
  // ofRenormed.close();
};
