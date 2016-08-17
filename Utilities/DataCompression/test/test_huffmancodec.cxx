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

//  @file   test_huffmancodec.cxx
//  @author Matthias Richter
//  @since  2015-08-11
//  @brief  Test program for Huffman codec template class

// Compilation: make sure variable BOOST_ROOT points to your boost installation
/*
   g++ --std=c++11 -g -ggdb -I$BOOST_ROOT/include -I../include -o test_huffmancodec test_huffmancodec.cxx
*/

#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <bitset>
#include <random> // std::exponential_distribution
#include <cmath>  // std::exp
#include <boost/mpl/string.hpp>
#include "DataCompression/dc_primitives.h"
#include "DataCompression/HuffmanCodec.h"

int main()
{
  // defining a contiguous alphabet of integral 16 bit unsigned numbers
  // in the range [-1, 10] including the upper bound
  // the first definition is a data type, then an object of this type is
  // defined
  typedef ContiguousAlphabet<int16_t, -1, 15> SimpleRangeAlphabet_t;
  SimpleRangeAlphabet_t alphabet;
  std::cout << "alphabet '" << alphabet.getName()
	    << "' has " <<  SimpleRangeAlphabet_t::size::value
	    << " element(s)"
	    << std::endl;
  for (auto a : alphabet) {
    // trying also the postfix ++ operator; a is not the iterator itself
    // so it has no consequence on the iteration
    std::cout << a++ << " ";
    // simple testing of the type defines in the alphabet iterator
    SimpleRangeAlphabet_t::iterator::value_type v = a;
    SimpleRangeAlphabet_t::iterator::reference r = a;
  }
  std::cout << std::endl;

  std::cout << std::endl << "checking some numbers to be valid symbols in the alphabet" << std::endl;
  std::vector<int16_t> values = {0 , 5, 15, -2, -1};
  for (auto v : values) {
    std::cout << v << " is " << (alphabet.isValid(v)?"valid":"not valid") << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////
  // Using the Huffman propability model for the alphabet
  //
  // HuffmanModel_t is a data type, huffmanmodel an object of this type
  // the node type is defined to be HuffmanNode specialized to bitset<16>
  // third template parameter determines whether code has to be decoded
  // MSB to LSB (true) or LSB to MSB (false)
  typedef AliceO2::HuffmanModel<
    ProbabilityModel<SimpleRangeAlphabet_t>
    , AliceO2::HuffmanNode<std::bitset<16> >
    , true
    > HuffmanModel_t;
  HuffmanModel_t huffmanmodel;

  std::cout << std::endl << "Huffman probability model after initialization: " << std::endl;
  huffmanmodel.init(0.);
  for (auto s : alphabet) {
    std::cout << "val = " << std::setw(2) << s << " --> weight = " << huffmanmodel[s] << std::endl;
  }

  // an exponential distribution random generator to be used to create a
  // sequence of random numbers to be encoded
  double lambda=1.;
  std::default_random_engine generator;
  std::exponential_distribution<double> distribution(lambda);

  // add weights directly for every symbol, using the expenential distribution
  // and an epsilon value to make the Huffman table more 'interesting'
  // set epsilon to 0 to get a completely 'linear' tree
  int x = 0;
  for (auto s : alphabet) {
    double epsilon = (SimpleRangeAlphabet_t::size::value - x)*0.1;
    double p = lambda * std::exp(-lambda*x++) + epsilon;
    //p = p - 1.; // compensate for the 1 we have previously initialized
    huffmanmodel.addWeight(s, p);
  }

  // normalizing the weight to the total weight thus having the probability
  // for every symbol
  std::cout << std::endl << "Probabilities:" << std::endl;
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
  typedef AliceO2::HuffmanCodec<HuffmanModel_t > Codec_t;
  Codec_t codec(huffmanmodel);

  ////////////////////////////////////////////////////////////////////////////
  // print Huffman code summary
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
    for (int i = 0; i<codeLen; i++)
      std::cout << code[codeLen-1-i];
    std::cout << std::endl;
    if (HuffmanModel_t::orderMSB) code <<= (code.size()-codeLen);
    uint16_t decodedLen = 0;
    HuffmanModel_t::value_type value;
    codec.Decode(value, code, decodedLen);
    if (codeLen != decodedLen || value != i.first) {
      std::cout << "mismatch in decoded value: " << value << "(" << decodedLen << ")" << std::endl;
    }
  }
}
