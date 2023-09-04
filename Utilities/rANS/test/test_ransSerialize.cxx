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

/// @file   test_ransSerialize
/// @author Michael Lettrich
/// @brief  Test serialization and deserialization of metadata

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <vector>
#include <cstring>
#include <random>
#include <algorithm>

#include <boost/test/unit_test.hpp>
#include <boost/mp11.hpp>
#include <fmt/core.h>

#include "rANS/serialize.h"
#include "rANS/histogram.h"
#include "rANS/factory.h"
#include "rANS/metrics.h"
#include "rANS/internal/containers/Symbol.h"

using namespace o2::rans;

using buffer_types = boost::mp11::mp_list<uint8_t, uint16_t, uint32_t, uint64_t>;
using source_types = boost::mp11::mp_list<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>;

using test_types = boost::mp11::mp_product<boost::mp11::mp_list, source_types, buffer_types>;

BOOST_AUTO_TEST_CASE_TEMPLATE(testSerializeDeserializeEmptyHistogram, T, test_types)
{
  using source_type = boost::mp11::mp_first<T>;
  using buffer_type = boost::mp11::mp_second<T>;

  DenseHistogram<source_type> h{};
  Metrics<source_type> metrics{h};
  auto srcRenormedHistogram = renorm(h, metrics);
  SizeEstimate sizeEstimate{metrics};
  size_t bufferSize = sizeEstimate.getCompressedDictionarySize<buffer_type>();

  std::vector<buffer_type> serializationBuffer(bufferSize, 0);
  auto begin = serializationBuffer.data();
  auto end = compressRenormedDictionary(srcRenormedHistogram, begin);

  // fill the rest of the serialization buffer with 1s to test that we are not reading bits from the buffer that comes after.
  for (auto iter = end; iter != begin + serializationBuffer.size(); ++iter) {
    *iter = static_cast<buffer_type>(~0);
  }

  auto restoredRenormedHistogram = readRenormedDictionary(begin, end, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max, srcRenormedHistogram.getRenormingBits());

  BOOST_CHECK_EQUAL(srcRenormedHistogram.getIncompressibleSymbolFrequency(), restoredRenormedHistogram.getIncompressibleSymbolFrequency());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getOffset(), restoredRenormedHistogram.getOffset());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getNumSamples(), restoredRenormedHistogram.getNumSamples());

  DenseSymbolTable<source_type, internal::Symbol> srcSymbolTable(srcRenormedHistogram);
  DenseSymbolTable<source_type, internal::Symbol> restoredSymbolTable(restoredRenormedHistogram);

  BOOST_CHECK_EQUAL_COLLECTIONS(srcSymbolTable.begin(), srcSymbolTable.end(), restoredSymbolTable.begin(), restoredSymbolTable.end());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testSerializeDeserializeEmptySymbolTable, T, test_types)
{
  using source_type = boost::mp11::mp_first<T>;
  using buffer_type = boost::mp11::mp_second<T>;

  DenseHistogram<source_type> h{};
  Metrics<source_type> metrics{h};
  auto srcRenormedHistogram = renorm(h, metrics);
  DenseSymbolTable<source_type, internal::Symbol> srcSymbolTable(srcRenormedHistogram);

  SizeEstimate sizeEstimate{metrics};

  std::vector<buffer_type> serializationBuffer(256, 0);
  auto begin = serializationBuffer.data();
  auto end = compressRenormedDictionary(srcSymbolTable, begin);

  // fill the rest of the serialization buffer with 1s to test that we are not reading bits from the buffer that comes after.
  for (auto iter = end; iter != begin + serializationBuffer.size(); ++iter) {
    *iter = static_cast<buffer_type>(~0);
  }

  auto restoredRenormedHistogram = readRenormedDictionary(begin, end, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max, srcRenormedHistogram.getRenormingBits());

  BOOST_CHECK_EQUAL(srcRenormedHistogram.getIncompressibleSymbolFrequency(), restoredRenormedHistogram.getIncompressibleSymbolFrequency());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getOffset(), restoredRenormedHistogram.getOffset());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getNumSamples(), restoredRenormedHistogram.getNumSamples());

  DenseSymbolTable<source_type, internal::Symbol> restoredSymbolTable(restoredRenormedHistogram);

  BOOST_CHECK_EQUAL_COLLECTIONS(srcSymbolTable.begin(), srcSymbolTable.end(), restoredSymbolTable.begin(), restoredSymbolTable.end());
};

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
      throw Exception{"unsupported source type"};
    }
  };

 private:
  inline static constexpr size_t MessageSize = utils::pow2(10);
  inline static const SourceMessage<uint8_t> sourceMessage8u{MessageSize};
  inline static const SourceMessage<int8_t> sourceMessage8{MessageSize};
  inline static const SourceMessage<uint16_t> sourceMessage16u{MessageSize};
  inline static const SourceMessage<int16_t> sourceMessage16{MessageSize};
  inline static const SourceMessage<uint32_t> sourceMessage32u{MessageSize, utils::pow2(27)};
  inline static const SourceMessage<int32_t> sourceMessage32{MessageSize, utils::pow2(26), -static_cast<int32_t>(utils::pow2(26))};
};

inline const SourceMessageProxy MessageProxy{};

BOOST_AUTO_TEST_CASE_TEMPLATE(testSerializeDeserializeHistogram, T, test_types)
{
  using source_type = boost::mp11::mp_first<T>;
  using buffer_type = boost::mp11::mp_second<T>;

  auto message = MessageProxy.getMessage<source_type>();
  auto h = makeDenseHistogram::fromSamples(message.begin(), message.end());
  Metrics<source_type> metrics{h};
  auto srcRenormedHistogram = renorm(h, metrics);
  ;
  SizeEstimate sizeEstimate{metrics};
  size_t bufferSize = sizeEstimate.getCompressedDictionarySize<buffer_type>();

  std::vector<buffer_type> serializationBuffer(bufferSize, 0);
  auto begin = serializationBuffer.data();
  auto end = compressRenormedDictionary(srcRenormedHistogram, begin);

  // fill the rest of the serialization buffer with 1s to test that we are not reading bits from the buffer that comes after.
  for (auto iter = end; iter != begin + serializationBuffer.size(); ++iter) {
    *iter = static_cast<buffer_type>(~0);
  }

  auto restoredRenormedHistogram = readRenormedDictionary(begin, end, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max, srcRenormedHistogram.getRenormingBits());

  BOOST_CHECK_EQUAL(srcRenormedHistogram.getIncompressibleSymbolFrequency(), restoredRenormedHistogram.getIncompressibleSymbolFrequency());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getNumSamples(), restoredRenormedHistogram.getNumSamples());

  DenseSymbolTable<source_type, internal::Symbol> srcSymbolTable(srcRenormedHistogram);
  DenseSymbolTable<source_type, internal::Symbol> restoredSymbolTable(restoredRenormedHistogram);

  BOOST_CHECK_EQUAL(srcSymbolTable.getOffset(), restoredSymbolTable.getOffset());
  BOOST_CHECK_EQUAL_COLLECTIONS(srcSymbolTable.begin(), srcSymbolTable.end(), restoredSymbolTable.begin(), restoredSymbolTable.end());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testSerializeDeserializeSymbolTable, T, test_types)
{
  using source_type = boost::mp11::mp_first<T>;
  using buffer_type = boost::mp11::mp_second<T>;

  auto message = MessageProxy.getMessage<source_type>();
  auto h = makeDenseHistogram::fromSamples(message.begin(), message.end());
  Metrics<source_type> metrics{h};
  auto srcRenormedHistogram = renorm(h, metrics);
  DenseSymbolTable<source_type, internal::Symbol> srcSymbolTable(srcRenormedHistogram);

  SizeEstimate sizeEstimate{metrics};
  size_t bufferSize = sizeEstimate.getCompressedDictionarySize<buffer_type>();

  std::vector<buffer_type> serializationBuffer(bufferSize, 0);
  auto begin = serializationBuffer.data();
  auto end = compressRenormedDictionary(srcSymbolTable, begin);

  // fill the rest of the serialization buffer with 1s to test that we are not reading bits from the buffer that comes after.
  for (auto iter = end; iter != begin + serializationBuffer.size(); ++iter) {
    *iter = static_cast<buffer_type>(~0);
  }

  auto restoredRenormedHistogram = readRenormedDictionary(begin, end, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max, srcRenormedHistogram.getRenormingBits());

  BOOST_CHECK_EQUAL(srcRenormedHistogram.getIncompressibleSymbolFrequency(), restoredRenormedHistogram.getIncompressibleSymbolFrequency());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getNumSamples(), restoredRenormedHistogram.getNumSamples());

  DenseSymbolTable<source_type, internal::Symbol> restoredSymbolTable(restoredRenormedHistogram);

  BOOST_CHECK_EQUAL(srcSymbolTable.getOffset(), restoredSymbolTable.getOffset());
  BOOST_CHECK_EQUAL_COLLECTIONS(srcSymbolTable.begin(), srcSymbolTable.end(), restoredSymbolTable.begin(), restoredSymbolTable.end());
};

using adaptiveSource_types = boost::mp11::mp_list<uint32_t, int32_t>;

using adaptiveTest_types = boost::mp11::mp_product<boost::mp11::mp_list, source_types, buffer_types>;

BOOST_AUTO_TEST_CASE_TEMPLATE(testSerializeDeserializeAdaptiveHistogram, T, adaptiveTest_types)
{
  using source_type = boost::mp11::mp_first<T>;
  using buffer_type = boost::mp11::mp_second<T>;

  auto message = MessageProxy.getMessage<source_type>();
  auto h = makeSparseHistogram::fromSamples(message.begin(), message.end());
  Metrics<source_type> metrics{h};
  auto srcRenormedHistogram = renorm(h, metrics);
  auto h1 = makeDenseHistogram::fromSamples(message.begin(), message.end());
  auto crossCheckRenormedHistogram = renorm(h1, metrics);
  SizeEstimate sizeEstimate{metrics};
  size_t bufferSize = sizeEstimate.getCompressedDictionarySize<buffer_type>();

  std::vector<buffer_type> serializationBuffer(bufferSize, 0);
  auto begin = serializationBuffer.data();
  auto end = compressRenormedDictionary(srcRenormedHistogram, begin);

  // fill the rest of the serialization buffer with 1s to test that we are not reading bits from the buffer that comes after.
  for (auto iter = end; iter != begin + serializationBuffer.size(); ++iter) {
    *iter = static_cast<buffer_type>(~0);
  }

  auto restoredRenormedHistogram = readRenormedDictionary(begin, end, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max, srcRenormedHistogram.getRenormingBits());

  BOOST_CHECK_EQUAL(srcRenormedHistogram.getIncompressibleSymbolFrequency(), restoredRenormedHistogram.getIncompressibleSymbolFrequency());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getNumSamples(), restoredRenormedHistogram.getNumSamples());

  DenseSymbolTable<source_type, internal::Symbol> srcSymbolTable(crossCheckRenormedHistogram);
  DenseSymbolTable<source_type, internal::Symbol> restoredSymbolTable(restoredRenormedHistogram);

  BOOST_CHECK_EQUAL(srcSymbolTable.getOffset(), restoredSymbolTable.getOffset());
  BOOST_CHECK_EQUAL_COLLECTIONS(srcSymbolTable.begin(), srcSymbolTable.end(), restoredSymbolTable.begin(), restoredSymbolTable.end());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testSerializeDeserializeAdaptiveSymbolTable, T, adaptiveTest_types)
{
  using source_type = boost::mp11::mp_first<T>;
  using buffer_type = boost::mp11::mp_second<T>;

  auto message = MessageProxy.getMessage<source_type>();
  auto h = makeSparseHistogram::fromSamples(message.begin(), message.end());
  auto h1 = makeDenseHistogram::fromSamples(message.begin(), message.end());
  Metrics<source_type> metrics{h};
  auto srcRenormedHistogram = renorm(h, metrics);
  AdaptiveSymbolTable<source_type, internal::Symbol> srcSymbolTable(srcRenormedHistogram);
  DenseSymbolTable<source_type, internal::Symbol> srcCrossCheckSymbolTable{renorm(h1, metrics)};

  SizeEstimate sizeEstimate{metrics};
  size_t bufferSize = sizeEstimate.getCompressedDictionarySize<buffer_type>();

  std::vector<buffer_type> serializationBuffer(bufferSize, 0);
  auto begin = serializationBuffer.data();
  auto end = compressRenormedDictionary(srcSymbolTable, begin);

  // fill the rest of the serialization buffer with 1s to test that we are not reading bits from the buffer that comes after.
  for (auto iter = end; iter != begin + serializationBuffer.size(); ++iter) {
    *iter = static_cast<buffer_type>(~0);
  }

  auto restoredRenormedHistogram = readRenormedDictionary(begin, end, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max, srcRenormedHistogram.getRenormingBits());

  BOOST_CHECK_EQUAL(srcRenormedHistogram.getIncompressibleSymbolFrequency(), restoredRenormedHistogram.getIncompressibleSymbolFrequency());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getNumSamples(), restoredRenormedHistogram.getNumSamples());

  DenseSymbolTable<source_type, internal::Symbol> restoredSymbolTable(restoredRenormedHistogram);

  BOOST_CHECK_EQUAL(srcCrossCheckSymbolTable.getOffset(), restoredSymbolTable.getOffset());
  BOOST_CHECK_EQUAL_COLLECTIONS(srcCrossCheckSymbolTable.begin(), srcCrossCheckSymbolTable.end(), restoredSymbolTable.begin(), restoredSymbolTable.end());
};

using sparseTest_types = adaptiveTest_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(testSerializeDeserializeSparseHistogram, T, sparseTest_types)
{
  using source_type = boost::mp11::mp_first<T>;
  using buffer_type = boost::mp11::mp_second<T>;

  auto message = MessageProxy.getMessage<source_type>();
  auto h = makeSparseHistogram::fromSamples(message.begin(), message.end());
  Metrics<source_type> metrics{h};
  auto srcRenormedHistogram = renorm(h, metrics);
  auto h1 = makeDenseHistogram::fromSamples(message.begin(), message.end());
  auto crossCheckRenormedHistogram = renorm(h1, metrics);
  SizeEstimate sizeEstimate{metrics};
  size_t bufferSize = sizeEstimate.getCompressedDictionarySize<buffer_type>();

  std::vector<buffer_type> serializationBuffer(bufferSize, 0);
  auto begin = serializationBuffer.data();
  auto end = compressRenormedDictionary(srcRenormedHistogram, begin);

  // fill the rest of the serialization buffer with 1s to test that we are not reading bits from the buffer that comes after.
  for (auto iter = end; iter != begin + serializationBuffer.size(); ++iter) {
    *iter = static_cast<buffer_type>(~0);
  }

  auto restoredRenormedHistogram = readRenormedDictionary(begin, end, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max, srcRenormedHistogram.getRenormingBits());

  BOOST_CHECK_EQUAL(srcRenormedHistogram.getIncompressibleSymbolFrequency(), restoredRenormedHistogram.getIncompressibleSymbolFrequency());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getNumSamples(), restoredRenormedHistogram.getNumSamples());

  DenseSymbolTable<source_type, internal::Symbol> srcSymbolTable(crossCheckRenormedHistogram);
  DenseSymbolTable<source_type, internal::Symbol> restoredSymbolTable(restoredRenormedHistogram);

  BOOST_CHECK_EQUAL(srcSymbolTable.getOffset(), restoredSymbolTable.getOffset());
  BOOST_CHECK_EQUAL_COLLECTIONS(srcSymbolTable.begin(), srcSymbolTable.end(), restoredSymbolTable.begin(), restoredSymbolTable.end());
};

BOOST_AUTO_TEST_CASE_TEMPLATE(testSerializeDeserializeSparseSymbolTable, T, sparseTest_types)
{
  using source_type = boost::mp11::mp_first<T>;
  using buffer_type = boost::mp11::mp_second<T>;

  auto message = MessageProxy.getMessage<source_type>();
  auto h = makeSparseHistogram::fromSamples(message.begin(), message.end());
  auto h1 = makeDenseHistogram::fromSamples(message.begin(), message.end());
  Metrics<source_type> metrics{h};
  auto srcRenormedHistogram = renorm(h, metrics);
  SparseSymbolTable<source_type, internal::Symbol> srcSymbolTable(srcRenormedHistogram);
  DenseSymbolTable<source_type, internal::Symbol> srcCrossCheckSymbolTable{renorm(h1, metrics)};

  SizeEstimate sizeEstimate{metrics};
  size_t bufferSize = sizeEstimate.getCompressedDictionarySize<buffer_type>();

  std::vector<buffer_type> serializationBuffer(bufferSize, 0);
  auto begin = serializationBuffer.data();
  auto end = compressRenormedDictionary(srcSymbolTable, begin);

  // fill the rest of the serialization buffer with 1s to test that we are not reading bits from the buffer that comes after.
  for (auto iter = end; iter != begin + serializationBuffer.size(); ++iter) {
    *iter = static_cast<buffer_type>(~0);
  }

  auto restoredRenormedHistogram = readRenormedDictionary(begin, end, *metrics.getCoderProperties().min, *metrics.getCoderProperties().max, srcRenormedHistogram.getRenormingBits());

  BOOST_CHECK_EQUAL(srcRenormedHistogram.getIncompressibleSymbolFrequency(), restoredRenormedHistogram.getIncompressibleSymbolFrequency());
  BOOST_CHECK_EQUAL(srcRenormedHistogram.getNumSamples(), restoredRenormedHistogram.getNumSamples());

  DenseSymbolTable<source_type, internal::Symbol> restoredSymbolTable(restoredRenormedHistogram);

  BOOST_CHECK_EQUAL(srcCrossCheckSymbolTable.getOffset(), restoredSymbolTable.getOffset());
  BOOST_CHECK_EQUAL_COLLECTIONS(srcCrossCheckSymbolTable.begin(), srcCrossCheckSymbolTable.end(), restoredSymbolTable.begin(), restoredSymbolTable.end());
};