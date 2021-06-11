// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_HuffmanCodec.cxx
/// @author Matthias Richter
/// @since  2016-08-11
/// @brief  Test program for Huffman codec template class

#define BOOST_TEST_MODULE HuffmanCodec unit test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <bitset>
#include <thread>
#include <stdexcept> // exeptions, runtime_error
#include "../include/DataCompression/dc_primitives.h"
#include "../include/DataCompression/HuffmanCodec.h"
#include "CommonUtils/StringUtils.h"
#include "DataGenerator.h"
#include "Fifo.h"

namespace o2
{
namespace data_compression
{

// a decoder process working on the FIFO of encoded data and comparing to
// original data from the corresponding FIFO
template <class RandvalStreamT, class EncodedStreamT, class CodecT>
void decoderProcess(RandvalStreamT& fifoRandvals, EncodedStreamT& fifoEncoded, CodecT& codec)
{
  uint16_t decodedLen = 0;
  typename CodecT::model_type::value_type decodedValue;
  do {
  } while (fifoEncoded.pull([&](typename EncodedStreamT::value_type c) {
    codec.Decode(decodedValue, c, decodedLen);
    return fifoRandvals.pull([&](typename RandvalStreamT::value_type v) {
      if (decodedValue != v) {
        throw std::runtime_error("decoding mismatch");
        return false;
      } else {
        // std::cout << "decoded: "
        //          << std::setw(4) << decodedValue
        //          << " code: " << c
        //          << std::endl;
      }
      return true;
    });
  }));
}

template <typename CodecT, typename GeneratorT>
void checkRandom(CodecT& codec, GeneratorT& generator, int nRolls = 1000000)
{
  using ValueT = typename CodecT::value_type;
  using CodeT = typename CodecT::code_type;
  auto const& huffmanmodel = codec.getCodingModel();

  ////////////////////////////////////////////////////////////////////////////
  // test loop for random values
  //

  // FIFO for the random numbers
  o2::test::Fifo<ValueT> fifoRandvals;

  // FIFO for encoded values
  using FifoBuffer_t = o2::test::Fifo<uint32_t>;
  FifoBuffer_t fifoEncoded;

  int n = nRolls;
  std::cout << std::endl
            << "Testing encoding-decoding with " << nRolls << " random value(s) ..." << std::endl;

  std::thread decoderThread([&]() { decoderProcess(fifoRandvals, fifoEncoded, codec); });

  while (n-- > 0) {
    uint16_t codeLen = 0;
    CodeT code;
    ValueT value = generator();
    codec.Encode(value, code, codeLen);
    fifoRandvals.push(value);
    if (huffmanmodel.OrderMSB) {
      code <<= (code.size() - codeLen);
    }
    fifoEncoded.push(code.to_ulong(), n == 0);
    // std::cout << "encoded: " << std::setw(4) << value << " code: " << code << std::endl;
  }

  decoderThread.join();

  std::cout << "... done" << std::endl;
}

auto setupCodec(int verbosity = 0)
{
  // defining a contiguous alphabet of integral 16 bit unsigned numbers
  // in the range [-7, 10] including the upper bound
  // the first definition is a data type, then an object of this type is
  // defined
  using TestDistribution_t = o2::test::normal_distribution<double>;
  using DataGenerator_t = o2::test::DataGenerator<int16_t, TestDistribution_t>;
  DataGenerator_t dg(-7, 10, 1, 0., 1.);
  using SimpleRangeAlphabet_t = ContiguousAlphabet<DataGenerator_t::value_type, -7, 10>;
  SimpleRangeAlphabet_t alphabet;

  ////////////////////////////////////////////////////////////////////////////
  // Using the Huffman propability model for the alphabet
  //
  // HuffmanModel_t is a data type, huffmanmodel an object of this type
  // the node type is defined to be HuffmanNode specialized to bitset<16>
  // third template parameter determines whether code has to be decoded
  // MSB to LSB (true) or LSB to MSB (false)
  using HuffmanModel_t =
    HuffmanModel<ProbabilityModel<SimpleRangeAlphabet_t>, std::bitset<32>, true>;
  HuffmanModel_t huffmanmodel;

  huffmanmodel.init(0.);
  if (verbosity > 0) {
    std::cout << std::endl
              << "Huffman probability model after initialization: " << std::endl;
    for (auto s : alphabet) {
      std::cout << "val = " << std::setw(2) << s << " --> weight = " << huffmanmodel[s] << std::endl;
    }
  }

  // add probabilities from data generator as weights for every symbol
  for (auto s : alphabet) {
    huffmanmodel.addWeight(s, dg.getProbability(s));
  }

  // normalizing the weight to the total weight thus having the probability
  // for every symbol
  huffmanmodel.normalize();
  if (verbosity > 0) {
    std::cout << std::endl
              << "Probabilities from DataGenerator:" << std::endl;
    for (auto i : huffmanmodel) {
      std::cout << "val = " << std::setw(2) << i.first << " --> weight = " << i.second << std::endl;
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  // generate the Huffman tree in the Huffman model and create a Huffman
  // codec operating on the probability table
  huffmanmodel.GenerateHuffmanTree();
  if (verbosity > 0) {
    std::cout << std::endl
              << "Generating binary tree and Huffman codes" << std::endl;
    huffmanmodel.print();
  }
  return std::pair<HuffmanCodec<HuffmanModel_t>, DataGenerator_t>(huffmanmodel, dg);
}

BOOST_AUTO_TEST_CASE(test_HuffmanCodec_basic)
{
  auto setup = setupCodec();
  auto& codec = setup.first;
  auto& dg = setup.second;
  auto const& huffmanmodel = codec.getCodingModel();
  using ValueT = decltype(setup.first)::value_type;
  using CodeT = decltype(setup.first)::code_type;

  ////////////////////////////////////////////////////////////////////////////
  // print Huffman code summary and perform an encoding-decoding check for
  // every symbol
  std::cout << std::endl
            << "Huffman code summary: " << (huffmanmodel.OrderMSB ? "MSB to LSB" : "LSB to MSB") << std::endl;
  for (auto const& i : huffmanmodel) {
    uint16_t codeLen = 0;
    CodeT code;
    codec.Encode(i.first, code, codeLen);
    std::cout << "value: " << std::setw(4) << i.first << "   code length: " << std::setw(3) << codeLen << "   code: ";
    if (not huffmanmodel.OrderMSB) {
      std::cout << std::setw(code.size() - codeLen);
    }
    for (int k = 0; k < codeLen; k++) {
      std::cout << code[codeLen - 1 - k];
    }
    std::cout << std::endl;
    if (huffmanmodel.OrderMSB) {
      code <<= (code.size() - codeLen);
    }
    uint16_t decodedLen = 0;
    ValueT value;
    codec.Decode(value, code, decodedLen);
    if (codeLen != decodedLen || value != i.first) {
      std::cout << "mismatch in decoded value: " << value << "(" << decodedLen << ")" << std::endl;
    }
  }

  checkRandom(codec, dg);
}

BOOST_AUTO_TEST_CASE(test_HuffmanCodec_configuration)
{
  auto setup = setupCodec();
  auto& codec = setup.first;
  auto& dg = setup.second;
  auto const& huffmanmodel = codec.getCodingModel();
  using ValueT = decltype(setup.first)::value_type;
  using CodeT = decltype(setup.first)::code_type;

  // check writing and reading of the huffman configuration
  std::stringstream filename;
  filename << o2::utils::Str::create_unique_path(std::filesystem::temp_directory_path().native()) << "_testHuffmanCodec.zlib";

  auto nNodes = codec.writeConfiguration(filename.str().c_str(), "zlib");
  BOOST_CHECK(nNodes > 0);
  auto result = codec.loadConfiguration(filename.str().c_str(), "zlib");
  BOOST_CHECK(result == 0);
  std::filesystem::remove(filename.str());

  checkRandom(codec, dg);
}

} // namespace data_compression
} // namespace o2
