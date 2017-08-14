// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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

//  @file   test_HuffmanCodec.cxx
//  @author Matthias Richter
//  @since  2016-08-11
//  @brief  Test program for Huffman codec template class

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <bitset>
#include <thread>
#include <stdexcept>  // exeptions, runtime_error
#include "../include/DataCompression/dc_primitives.h"
#include "../include/DataCompression/HuffmanCodec.h"
#include "DataGenerator.h"
#include "Fifo.h"

template<class RandvalStreamT
         , class EncodedStreamT
         , class CodecT
         >
void decoderProcess(RandvalStreamT& fifoRandvals, EncodedStreamT& fifoEncoded, CodecT& codec)
{
  uint16_t decodedLen = 0;
  typename CodecT::model_type::value_type decodedValue;
  do {}
  while (fifoEncoded.pull([&](typename EncodedStreamT::value_type c)
                          {
                            codec.Decode(decodedValue, c, decodedLen);
                            return fifoRandvals.pull([&](typename RandvalStreamT::value_type v)
                                                     {
                                                       if (decodedValue != v) {
                                                         throw std::runtime_error("decoding mismatch");
                                                         return false;
                                                       } else {
                                                         //std::cout << "decoded: "
                                                         //          << std::setw(4) << decodedValue
                                                         //          << " code: " << c
                                                         //          << std::endl;
                                                       }
                                                       return true;
                                                     }
                                                     );
                          }
                          )
         );
}

BOOST_AUTO_TEST_CASE(test_HuffmanCodec)
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
  using HuffmanModel_t = o2::HuffmanModel<
    ProbabilityModel<SimpleRangeAlphabet_t>
    , o2::HuffmanNode<std::bitset<32> >
    , true
    >;
  HuffmanModel_t huffmanmodel;

  std::cout << std::endl << "Huffman probability model after initialization: " << std::endl;
  huffmanmodel.init(0.);
  for (auto s : alphabet) {
    std::cout << "val = " << std::setw(2) << s << " --> weight = " << huffmanmodel[s] << std::endl;
  }

  // add probabilities from data generator as weights for every symbol
  for (auto s : alphabet) {
    huffmanmodel.addWeight(s, dg.getProbability(s));
  }

  // normalizing the weight to the total weight thus having the probability
  // for every symbol
  std::cout << std::endl << "Probabilities from DataGenerator:" << std::endl;
  huffmanmodel.normalize();
  for (auto i : huffmanmodel) {
    std::cout << "val = " << std::setw(2) << i.first << " --> weight = " << i.second << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////
  // generate the Huffman tree in the Huffman model and create a Huffman
  // codec operating on the probability table
  std::cout << std::endl << "Generating binary tree and Huffman codes" << std::endl;
  huffmanmodel.GenerateHuffmanTree();
  huffmanmodel.print();
  using Codec_t = o2::HuffmanCodec<HuffmanModel_t >;
  Codec_t codec(huffmanmodel);

  ////////////////////////////////////////////////////////////////////////////
  // print Huffman code summary and perform an encoding-decoding check for
  // every symbol
  std::cout << std::endl << "Huffman code summary: "
            << (HuffmanModel_t::orderMSB?"MSB to LSB":"LSB to MSB")
            << std::endl;
  for ( auto i : huffmanmodel) {
    uint16_t codeLen = 0;
    HuffmanModel_t::code_type code;
    codec.Encode(i.first, code, codeLen);
    std::cout << "value: " << std::setw(4) << i.first 
              << "   code length: " << std::setw(3) << codeLen
              << "   code: ";
    if (not HuffmanModel_t::orderMSB) std::cout << std::setw(code.size() - codeLen);
    for (int k = 0; k<codeLen; k++) {
      std::cout << code[codeLen-1-k];
    }
    std::cout << std::endl;
    if (HuffmanModel_t::orderMSB) code <<= (code.size()-codeLen);
    uint16_t decodedLen = 0;
    HuffmanModel_t::value_type value;
    codec.Decode(value, code, decodedLen);
    if (codeLen != decodedLen || value != i.first) {
      std::cout << "mismatch in decoded value: " << value << "(" << decodedLen << ")" << std::endl;
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  // test loop for random values
  //

  // FIFO for the random numbers
  o2::test::Fifo<DataGenerator_t::value_type> fifoRandvals;

  // FIFO for encoded values
  using FifoBuffer_t = o2::test::Fifo<uint32_t>;
  FifoBuffer_t fifoEncoded;

  const int nRolls = 1000000;
  int n = nRolls;
  std::cout << std::endl << "Testing encoding-decoding with " << nRolls << " random value(s) ..." << std::endl;

  std::thread decoderThread([&](){decoderProcess(fifoRandvals, fifoEncoded, codec);});

  while (n-- > 0) {
    uint16_t codeLen = 0;
    HuffmanModel_t::code_type code;
    DataGenerator_t::value_type value = dg();
    codec.Encode(value, code, codeLen);
    fifoRandvals.push(value);
    if (HuffmanModel_t::orderMSB) code <<= (code.size()-codeLen);
    fifoEncoded.push(code.to_ulong(), n == 0);
    //std::cout << "encoded: " << std::setw(4) << value << " code: " << code << std::endl;
  }

  decoderThread.join();
  std::cout << "... done" << std::endl;
}
