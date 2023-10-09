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

/// @file   testCTFEntropyCoder
/// @author Michael Lettrich
/// @brief  Test entropy coding using rANS algorithm

#define BOOST_TEST_MODULE Test CTFEntropyCoder class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <vector>
#include <cstring>
#include <random>
#include <algorithm>
#include <version>

#include <boost/test/unit_test.hpp>
#include <boost/mp11.hpp>
#include <fmt/core.h>

#include "DetectorsCommonDataFormats/internal/Packer.h"
#include "DetectorsCommonDataFormats/internal/ExternalEntropyCoder.h"
#include "DetectorsCommonDataFormats/internal/InplaceEntropyCoder.h"
#include "rANS/histogram.h"
#include "rANS/metrics.h"
#include "rANS/factory.h"
#include "rANS/iterator.h"

using namespace o2;

using buffer_type = uint32_t;
using source_types = boost::mp11::mp_list<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>;

template <typename source_T>
class SourceMessage
{
 public:
  SourceMessage(size_t messageSize, source_T max = std::numeric_limits<source_T>::max(), source_T min = std::numeric_limits<source_T>::min()) : mMin{min}, mMax{max}
  {
    if (mSourceMessage.empty()) {
      std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
      assert(max >= min);
      const size_t draws = (max - min) + 1;
      const double probability = 0.5;
      std::binomial_distribution<int64_t> dist(draws, probability);
      mSourceMessage.resize(messageSize);
      std::generate(mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt, min]() -> source_T { return static_cast<int64_t>(dist(mt)) + min; });
    }
  }

  inline constexpr source_T getMin() const noexcept { return mMin; };
  inline constexpr source_T getMax() const noexcept { return mMax; };
  inline constexpr auto& get() const noexcept { return mSourceMessage; };

 private:
  source_T mMin{};
  source_T mMax{};
  std::vector<source_T> mSourceMessage{};
};

class SourceMessageProxy
{
 public:
  SourceMessageProxy() = default;

  template <typename T>
  const auto& getMessage() const noexcept
  {
    if constexpr (std::is_same_v<uint8_t, T>) {
      return sourceMessage8u.get();
    } else if constexpr (std::is_same_v<int8_t, T>) {
      return sourceMessage8.get();
    } else if constexpr (std::is_same_v<uint16_t, T>) {
      return sourceMessage16u.get();
    } else if constexpr (std::is_same_v<int16_t, T>) {
      return sourceMessage16.get();
    } else if constexpr (std::is_same_v<uint32_t, T>) {
      return sourceMessage32u.get();
    } else if constexpr (std::is_same_v<int32_t, T>) {
      return sourceMessage32.get();
    } else {
      throw std::runtime_error{"unsupported source type"};
    }
  };

 private:
  inline static constexpr size_t MessageSize = rans::utils::pow2(10);
  SourceMessage<uint8_t> sourceMessage8u{MessageSize};
  SourceMessage<int8_t> sourceMessage8{MessageSize};
  SourceMessage<uint16_t> sourceMessage16u{MessageSize};
  SourceMessage<int16_t> sourceMessage16{MessageSize};
  SourceMessage<uint32_t> sourceMessage32u{MessageSize, rans::utils::pow2(27)};
  SourceMessage<int32_t> sourceMessage32{MessageSize, rans::utils::pow2(26), -static_cast<int32_t>(rans::utils::pow2(26))};
};

inline const SourceMessageProxy MessageProxy{};

template <typename source_IT>
void encodeInplace(source_IT begin, source_IT end)
{
  using source_type = typename std::iterator_traits<source_IT>::value_type;

  ctf::internal::InplaceEntropyCoder<source_type> entropyCoder{begin, end};
  // BOOST_CHECK_THROW(entropyCoder.getEncoder(), std::runtime_error);
  entropyCoder.makeEncoder();

  const rans::Metrics<source_type>& metrics = entropyCoder.getMetrics();
  const rans::SizeEstimate sizeEstimate = metrics.getSizeEstimate();

  LOGP(info, "dataset[{},{}], coder[{},{}]", metrics.getDatasetProperties().min, metrics.getDatasetProperties().max, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max);

  std::vector<buffer_type> encodeBuffer(sizeEstimate.getCompressedDatasetSize<buffer_type>(), 0);
  std::vector<buffer_type> literalSymbolsBuffer(sizeEstimate.getIncompressibleSize<buffer_type>(), 0);
  std::vector<buffer_type> dictBuffer(sizeEstimate.getCompressedDictionarySize<buffer_type>(), 0);

  auto encoderEnd = entropyCoder.encode(begin, end, encodeBuffer.data(), encodeBuffer.data() + encodeBuffer.size());
  [[maybe_unused]] auto literalsEnd = entropyCoder.writeIncompressible(literalSymbolsBuffer.data(), literalSymbolsBuffer.data() + literalSymbolsBuffer.size());
  auto dictEnd = entropyCoder.writeDictionary(dictBuffer.data(), dictBuffer.data() + dictBuffer.size());
  // decode
  const auto& coderProperties = metrics.getCoderProperties();
  auto decoder = rans::makeDecoder<>::fromRenormed(rans::readRenormedDictionary(dictBuffer.data(), dictEnd,
                                                                                *coderProperties.min, *coderProperties.max,
                                                                                *coderProperties.renormingPrecisionBits));
  std::vector<source_type> literals(entropyCoder.getNIncompressibleSamples());

  const auto& datasetPropterties = metrics.getDatasetProperties();
  rans::unpack(literalSymbolsBuffer.data(), literals.size(), literals.data(),
               datasetPropterties.alphabetRangeBits, datasetPropterties.min);

  size_t messageLength = std::distance(begin, end);
  std::vector<source_type> sourceBuffer(messageLength, 0);

  decoder.process(encoderEnd, sourceBuffer.data(), messageLength, entropyCoder.getNStreams(), literals.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(sourceBuffer.begin(), sourceBuffer.end(), begin, end);
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testInplaceEncoderEmpty, source_T, source_types)
{
  std::vector<source_T> testMessage{};
  encodeInplace(testMessage.data(), testMessage.data() + testMessage.size());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testInplaceEncoderPTR, source_T, source_types)
{
  const auto& testMessage = MessageProxy.getMessage<source_T>();
  encodeInplace(testMessage.data(), testMessage.data() + testMessage.size());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testInplaceEncoderIter, source_T, source_types)
{
  const auto& testMessage = MessageProxy.getMessage<source_T>();
  encodeInplace(testMessage.begin(), testMessage.end());
};

template <typename value_T, size_t shift>
class ShiftFunctor

{
 public:
  template <typename iterA_T, typename iterB_T>
  inline value_T operator()(iterA_T iterA, iterB_T iterB) const
  {
    return *iterB + (static_cast<value_T>(*iterA) << shift);
  };

  template <typename iterA_T, typename iterB_T>
  inline void operator()(iterA_T iterA, iterB_T iterB, value_T value) const
  {
    *iterA = value >> shift;
    *iterB = value & ((0x1 << shift) - 0x1);
  };
};

template <typename iterA_T, typename iterB_T, typename F>
auto makeInputIterators(iterA_T iterA, iterB_T iterB, size_t nElements, F functor)
{
  using namespace o2::rans::utils;

  return std::make_tuple(rans::CombinedInputIterator{iterA, iterB, functor},
                         rans::CombinedInputIterator{advanceIter(iterA, nElements), advanceIter(iterB, nElements), functor});
};

BOOST_AUTO_TEST_CASE(testInplaceEncoderCombinedIterator)
{

  const auto& testMessage1 = MessageProxy.getMessage<int8_t>();
  const auto& testMessage2 = MessageProxy.getMessage<int8_t>();

  auto [begin, end] = makeInputIterators(testMessage1.data(), testMessage2.data(), testMessage1.size(), ShiftFunctor<uint16_t, rans::utils::toBits<uint8_t>()>{});

  encodeInplace(begin, end);
};

class ExternalEncoderDecoderProxy
{
 public:
  ExternalEncoderDecoderProxy()
  {
    SourceMessageProxy proxy{};

    auto renormed8u = rans::renorm(rans::makeDenseHistogram::fromSamples(proxy.getMessage<uint8_t>().begin(), proxy.getMessage<uint8_t>().end()), rans::RenormingPolicy::ForceIncompressible);
    auto renormed8 = rans::renorm(rans::makeDenseHistogram::fromSamples(proxy.getMessage<int8_t>().begin(), proxy.getMessage<int8_t>().end()), rans::RenormingPolicy::ForceIncompressible);
    auto renormed16u = rans::renorm(rans::makeDenseHistogram::fromSamples(proxy.getMessage<uint16_t>().begin(), proxy.getMessage<uint16_t>().end()), rans::RenormingPolicy::ForceIncompressible);
    auto renormed16 = rans::renorm(rans::makeDenseHistogram::fromSamples(proxy.getMessage<int16_t>().begin(), proxy.getMessage<int16_t>().end()), rans::RenormingPolicy::ForceIncompressible);
    auto renormed32u = rans::renorm(rans::makeDenseHistogram::fromSamples(proxy.getMessage<uint32_t>().begin(), proxy.getMessage<uint32_t>().end()), rans::RenormingPolicy::ForceIncompressible);
    auto renormed32 = rans::renorm(rans::makeDenseHistogram::fromSamples(proxy.getMessage<int32_t>().begin(), proxy.getMessage<int32_t>().end()), rans::RenormingPolicy::ForceIncompressible);

    encoder8u = rans::makeDenseEncoder<>::fromRenormed(renormed8u);
    encoder8 = rans::makeDenseEncoder<>::fromRenormed(renormed8);
    encoder16u = rans::makeDenseEncoder<>::fromRenormed(renormed16u);
    encoder16 = rans::makeDenseEncoder<>::fromRenormed(renormed16);
    encoder32u = rans::makeDenseEncoder<>::fromRenormed(renormed32u);
    encoder32 = rans::makeDenseEncoder<>::fromRenormed(renormed32);

    decoder8u = rans::makeDecoder<>::fromRenormed(renormed8u);
    decoder8 = rans::makeDecoder<>::fromRenormed(renormed8);
    decoder16u = rans::makeDecoder<>::fromRenormed(renormed16u);
    decoder16 = rans::makeDecoder<>::fromRenormed(renormed16);
    decoder32u = rans::makeDecoder<>::fromRenormed(renormed32u);
    decoder32 = rans::makeDecoder<>::fromRenormed(renormed32);
  }

  template <typename T>
  const auto& getEncoder() const noexcept
  {
    if constexpr (std::is_same_v<uint8_t, T>) {
      return encoder8u;
    } else if constexpr (std::is_same_v<int8_t, T>) {
      return encoder8;
    } else if constexpr (std::is_same_v<uint16_t, T>) {
      return encoder16u;
    } else if constexpr (std::is_same_v<int16_t, T>) {
      return encoder16;
    } else if constexpr (std::is_same_v<uint32_t, T>) {
      return encoder32u;
    } else if constexpr (std::is_same_v<int32_t, T>) {
      return encoder32;
    } else {
      throw std::runtime_error{"unsupported encoder type"};
    }
  };

  template <typename T>
  const auto& getDecoder() const noexcept
  {
    if constexpr (std::is_same_v<uint8_t, T>) {
      return decoder8u;
    } else if constexpr (std::is_same_v<int8_t, T>) {
      return decoder8;
    } else if constexpr (std::is_same_v<uint16_t, T>) {
      return decoder16u;
    } else if constexpr (std::is_same_v<int16_t, T>) {
      return decoder16;
    } else if constexpr (std::is_same_v<uint32_t, T>) {
      return decoder32u;
    } else if constexpr (std::is_same_v<int32_t, T>) {
      return decoder32;
    } else {
      throw std::runtime_error{"unsupported encoder type"};
    }
  };

 private:
  rans::denseEncoder_type<uint8_t> encoder8u{};
  rans::denseEncoder_type<int8_t> encoder8{};
  rans::denseEncoder_type<uint16_t> encoder16u{};
  rans::denseEncoder_type<int16_t> encoder16{};
  rans::denseEncoder_type<uint32_t> encoder32u{};
  rans::denseEncoder_type<int32_t> encoder32{};

  rans::defaultDecoder_type<uint8_t> decoder8u{};
  rans::defaultDecoder_type<int8_t> decoder8{};
  rans::defaultDecoder_type<uint16_t> decoder16u{};
  rans::defaultDecoder_type<int16_t> decoder16{};
  rans::defaultDecoder_type<uint32_t> decoder32u{};
  rans::defaultDecoder_type<int32_t> decoder32{};
};

ExternalEncoderDecoderProxy ExternalEncoders{};

template <typename source_IT>
void encodeExternal(source_IT begin, source_IT end)
{
  using source_type = typename std::iterator_traits<source_IT>::value_type;

  ctf::internal::ExternalEntropyCoder<source_type> entropyCoder{ExternalEncoders.getEncoder<source_type>()};

  const size_t sourceExtent = std::distance(begin, end);
  std::vector<buffer_type> encodeBuffer(entropyCoder.template computePayloadSizeEstimate<buffer_type>(sourceExtent), 0);
  auto encoderEnd = entropyCoder.encode(begin, end, encodeBuffer.data(), encodeBuffer.data() + encodeBuffer.size());

  std::vector<buffer_type> literalSymbolsBuffer(entropyCoder.template computePackedIncompressibleSize<buffer_type>(), 0);
  [[maybe_unused]] auto literalsEnd = entropyCoder.writeIncompressible(literalSymbolsBuffer.data(), literalSymbolsBuffer.data() + literalSymbolsBuffer.size());

  // decode
  auto decoder = ExternalEncoders.getDecoder<source_type>();
  std::vector<source_type> literals((entropyCoder.getNIncompressibleSamples()));

  rans::unpack(literalSymbolsBuffer.data(), literals.size(), literals.data(),
               entropyCoder.getIncompressibleSymbolPackingBits(), entropyCoder.getIncompressibleSymbolOffset());

  size_t messageLength = std::distance(begin, end);
  std::vector<source_type> sourceBuffer(messageLength, 0);

  decoder.process(encoderEnd, sourceBuffer.data(), messageLength, entropyCoder.getEncoder().getNStreams(), literals.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(sourceBuffer.begin(), sourceBuffer.end(), begin, end);
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testExternalEncoderEmpty, source_T, source_types)
{
  std::vector<source_T> testMessage{};
  encodeExternal(testMessage.data(), testMessage.data() + testMessage.size());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testExternalEncoderPTR, source_T, source_types)
{
  const auto& testMessage = MessageProxy.getMessage<source_T>();
  encodeExternal(testMessage.data(), testMessage.data() + testMessage.size());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testExternalEncoderIter, source_T, source_types)
{
  const auto& testMessage = MessageProxy.getMessage<source_T>();
  encodeExternal(testMessage.begin(), testMessage.end());
};

BOOST_AUTO_TEST_CASE(testExternalEncoderCombinedIterator)
{

  const auto& testMessage1 = MessageProxy.getMessage<int8_t>();
  const auto& testMessage2 = MessageProxy.getMessage<int8_t>();

  auto [begin, end] = makeInputIterators(testMessage1.data(), testMessage2.data(), testMessage1.size(), ShiftFunctor<uint16_t, rans::utils::toBits<uint8_t>()>{});

  encodeExternal(begin, end);
};