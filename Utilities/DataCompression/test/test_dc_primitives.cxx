// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "../include/DataCompression/dc_primitives.h"
#include <boost/mpl/size.hpp>
#include <boost/type.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/vector_c.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

/**
 * TODO: would like to have a general Tester for different kinds
 * of meta programs, but did not succeed so far to define a templated
 * Tester which can take the iterator argument at a placeholder
 * location. Some reading
 * http://stackoverflow.com/questions/24954220/boostmplfor-each-without-instantiating
 * http://stackoverflow.com/questions/2840640/how-to-loop-through-a-boostmpllist
 * http://stackoverflow.com/questions/16087806/boost-mpl-nested-lambdas
 */
struct getmaxTester
{
  template<typename T> void operator()(boost::type<T>) {
    std::cout << "Max number in " << std::setw(2) << T::value << "-bit range: "<< getmax<uint64_t,  T::value>::value << std::endl;
  }
};

struct upperbinaryboundTester
{
  template<typename T> void operator()(boost::type<T>) {
    std::cout << "number of bits required for value " << std::setw(4) << T::value << ": " << std::setw(3) << upperbinarybound<T::value>::value << std::endl;
  }
};

template<typename ValueList>
struct AlphabetTester
{
  ValueList mList;
  AlphabetTester();
  AlphabetTester(const ValueList& list) : mList(list) {}
  template<typename Alphabet> void operator()(Alphabet& alphabet) {
    for (const auto v : mList) {
      std::cout << "Alphabet '" << alphabet.getName() << "': value " << std::setw(2) << v << " is " << (alphabet.isValid(v)?"valid":"not valid") << std::endl;
    }
  }
};

BOOST_AUTO_TEST_CASE(test_dc_primitives)
{
  // test the getmax meta program
  std::cout << std::endl << "Testing getmax meta program ..." << std::endl;
  using bitranges = boost::mpl::vector_c<uint16_t, 0, 1, 2, 3, 4, 31, 32, 64>;
  boost::mpl::for_each<bitranges, boost::type<boost::mpl::_> >(getmaxTester());

  // test the getnofelements meta program
  std::cout << std::endl << "Testing getnofelements meta program ..." << std::endl;
  constexpr uint16_t lowerelement = 0;
  constexpr uint16_t upperelement = 10;
  std::cout << "Number of elements in range [" 
            << lowerelement << "," << upperelement << "]: "
            << getnofelements<uint16_t, lowerelement, upperelement >::value
            << std::endl;

  // test the upperbinarybound compile time evaluation
  std::cout << std::endl << "Testing upperbinarybound meta program ..." << std::endl;
  boost::mpl::for_each<boost::mpl::vector_c<int, 6, 1000, 86, 200>, boost::type<boost::mpl::_> >(upperbinaryboundTester());

  std::cout << std::endl << "Testing alphabet template ..." << std::endl;
  // declare two types of alphabets: a contiguous range alphabet with symbols
  // between -1 and 10 and a bit-range alphabet for a 10-bit word
  using TestAlphabetName = boost::mpl::string<'T','e','s','t'>::type;
  using TenBitAlphabetName = boost::mpl::string<'1','0','-','b','i','t'>::type;
  using TestAlphabet = ContiguousAlphabet<int16_t, -1, 10, TestAlphabetName>;
  using TenBitAlphabet = BitRangeContiguousAlphabet<int16_t, 10, TenBitAlphabetName>;

  // now check a set of values if they are valid in each of the alphabets
  // the check is done at runtime on types of alphabets rather than on
  // actual objects
  std::vector<int16_t> values = {0 , 5, 15, -2, -1};
  using ParameterSet = boost::mpl::vector<TestAlphabet, TenBitAlphabet>;
  boost::mpl::for_each<ParameterSet>( AlphabetTester<std::vector<int16_t>>(values) );
}
