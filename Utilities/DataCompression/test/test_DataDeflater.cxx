// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   test_DataDeflater.cxx
//  @author Matthias Richter
//  @since  2017-06-21
//  @brief  Test program for the DataDeflater template class

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <bitset>
#include <thread>
#include <stdexcept> // exeptions, runtime_error
#include "../include/DataCompression/DataDeflater.h"
#include "../include/DataCompression/TruncatedPrecisionConverter.h"
#include "DataGenerator.h"
#include "Fifo.h"

namespace o2dc = o2::data_compression;

template <typename DataContainerT, typename DeflatedDataT>
bool compare(const DataContainerT& container, std::size_t bitwidth, const DeflatedDataT& targetBuffer)
{
  unsigned wordcount = 0;
  auto bufferWord = targetBuffer[wordcount];
  using target_type = typename DeflatedDataT::value_type;
  auto targetWidth = 8 * sizeof(target_type);
  int position = targetWidth;
  for (const auto c : container) {
    int length = bitwidth;
    while (length > 0) {
      if (position == 0) {
        ++wordcount;
        BOOST_REQUIRE(wordcount < targetBuffer.size());
        position = targetWidth;
        bufferWord = targetBuffer[wordcount];
      }
      auto comparing = length;
      if (comparing > position) {
        comparing = position;
      }
      position -= comparing;
      target_type mask = ((target_type)1 << comparing) - 1;
      if (((bufferWord >> position) & mask) != ((c >> (length - comparing)) & mask)) {
        std::cout << "Decoding error at wordcount: " << wordcount << std::endl
                  << " length: " << length << std::endl
                  << " comparing: " << comparing << std::endl
                  << " position: " << position << std::endl
                  << " mask: " << std::hex << mask << std::endl
                  << std::endl
                  << " bufferWord: " << std::hex << bufferWord << std::endl
                  << " c: " << std::hex << c << std::endl;
      }

      BOOST_REQUIRE(((bufferWord >> position) & mask) == ((c >> (length - comparing)) & mask));

      length -= comparing;
    }
  }

  return true;
}

BOOST_AUTO_TEST_CASE(test_DataDeflaterRaw)
{
  using TestDataDeflater = o2dc::DataDeflater<uint8_t>;
  TestDataDeflater deflater;

  using target_type = TestDataDeflater::target_type;
  std::vector<target_type> targetBuffer;
  auto writerfct = [&](const target_type& value) -> bool {
    targetBuffer.emplace_back(value);
    return true;
  };

  std::array<char, 8> data = {'d', 'e', 'a', 'd', 'b', 'e', 'e', 'f'};

  const auto bitwidth = 7;
  for (auto c : data) {
    deflater.writeRaw(c, bitwidth, writerfct);
  }
  deflater.close(writerfct);
  compare(data, bitwidth, targetBuffer);
}

BOOST_AUTO_TEST_CASE(test_DataDeflaterCodec)
{
  constexpr auto bitwidth = 7;
  using Codec = o2dc::CodecIdentity<uint8_t, bitwidth>;
  using TestDataDeflater = o2dc::DataDeflater<uint8_t, Codec>;
  using target_type = TestDataDeflater::target_type;
  TestDataDeflater deflater;

  std::vector<target_type> targetBuffer;
  auto writerfct = [&](const target_type& value) -> bool {
    targetBuffer.emplace_back(value);
    return true;
  };

  std::array<char, 8> data = {'d', 'e', 'a', 'd', 'b', 'e', 'e', 'f'};

  for (auto c : data) {
    deflater.write(c, writerfct);
  }
  deflater.close(writerfct);
  compare(data, bitwidth, targetBuffer);
}

// define a simple parameter model to mask a data value
template <int NBits>
class ParameterModelBitMask
{
 public:
  ParameterModelBitMask() = default;
  ~ParameterModelBitMask() = default;

  static const int sBitlength = NBits;
  using converted_type = uint64_t;

  template <typename T>
  int convert(T value, converted_type& content, uint8_t& bitlength)
  {
    bitlength = sBitlength; // number of valid bits in the value
    uint32_t mask = 0x1 << bitlength;
    mask -= 1;
    content = value & mask;
    return 0;
  }

  void reset() {}

 private:
};

BOOST_AUTO_TEST_CASE(test_TruncatedPrecisionConverter)
{
  using Codec = o2dc::TruncatedPrecisionConverter<ParameterModelBitMask<7>>;
  using TestDataDeflater = o2dc::DataDeflater<uint8_t, Codec>;
  using target_type = TestDataDeflater::target_type;
  TestDataDeflater deflater;

  std::vector<target_type> targetBuffer;
  auto writerfct = [&](const target_type& value) -> bool {
    targetBuffer.emplace_back(value);
    return true;
  };

  std::array<char, 8> data = {'d', 'e', 'a', 'd', 'b', 'e', 'e', 'f'};

  for (auto c : data) {
    deflater.write(c, writerfct);
  }
  deflater.close(writerfct);
  compare(data, Codec::sMaxLength, targetBuffer);
}
