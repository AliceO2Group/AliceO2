// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef RANGE_TOKENIZER_H
#define RANGE_TOKENIZER_H

/// @file   RangeTokenizer.h
/// @author Matthias Richter
/// @since  2018-09-18
/// @brief  Helper function to tokenize sequences and ranges of integral numbers

#include <vector>
#include <string>
#include <sstream>
#include <utility>    // std::move
#include <functional> // std::function

namespace o2
{

/// @class RangeTokenizer
/// @brief Tokenize a string according to delimiter ',' and extract values of type T
///
/// Extract a sequence of elements of specified type T from a string argument. Elements are
/// separated by comma. If T is an integral type, also ranges are supported using '-'.
///
/// The default conversion from token to type is using std stringstream operator>> which
/// supports a variety of built-in conversions.
/// A custom handler function of type std::function<T(std::string const&)> can be provided
/// to convert string tokens to the specified output type.
///
/// @return std::vector of type T
///
/// Usage:
///   // the simple case using integral type
///   std::vector<int> tokens = RangeTokenizer::tokenize<int>("0-5,10,13");
///
///   // simple case using string type
///   std::vector<std::string> tokens = RangeTokenizer::tokenize<std::string>("apple,strawberry,tomato");
///
///   // process a custom type according to a map
///   // use a custom mapper function, this evetually throws an exception if the token is not in the map
///   enum struct Food { Apple,
///                      Strawberry,
///                      Tomato };
///   const std::map<std::string, Food> FoodMap {
///     { "apple", Food::Apple },
///     { "strawberry", Food::Strawberry },
///     { "tomato", Food::Tomato },
///   };
///   std::vector<Food> tokens = RangeTokenizer::tokenize<Food>("apple,tomato",
///                                                             [FoodMap](auto const& token) {
///                                                               return FoodMap.at(token);
///                                                             } );
struct RangeTokenizer {
  template <typename T>
  static std::vector<T> tokenize(
    std::string input, std::function<T(std::string const&)> convert = [](std::string const& token) {T value; std::istringstream(token) >> value; return value; })
  {
    std::istringstream stream(input);
    std::string token;
    std::vector<T> res;
    while (std::getline(stream, token, ',')) {
      if (std::is_integral<T>::value && token.find('-') != token.npos) {
        // extract range
        if constexpr (std::is_integral<T>::value) { // c++17 compile time
          insertRange(res, token, convert);
        }
      } else {
        res.emplace_back(convert(token));
      }
    }
    return std::move(res);
  }

  /// extract a range of an integral type from a token string and add to vector
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value == true, int> = 0>
  static void insertRange(std::vector<T>& res, std::string token, std::function<T(std::string const&)> convert)
  {
    std::istringstream tokenstream(token);
    std::string bound;
    T lowerBound, upperBound;
    if (std::getline(tokenstream, bound, '-')) {
      lowerBound = convert(bound);
      if (std::getline(tokenstream, bound, '-')) {
        upperBound = convert(bound);
        for (T index = lowerBound; index <= upperBound; index++) {
          res.emplace_back(index);
        }
      }
    }
  }
};
}; // namespace o2

#endif
