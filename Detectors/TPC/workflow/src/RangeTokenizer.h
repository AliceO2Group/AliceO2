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
#include <utility> // std::move
#include <functional> // std::function

namespace o2
{

struct RangeTokenizer {
  /// tokenize a string according to delimiter ',' and extract values of type T
  template <typename T>
  static std::vector<T> tokenize(std::string input, std::function<T(std::string const&)> convert = [](std::string const& token) {T value; std::istringstream(token) >> value; return value; })
  {
    std::istringstream stream(input);
    std::string token;
    std::vector<T> res;
    while (std::getline(stream, token, ',')) {
      T value;
      if (std::is_integral<T>::value && token.find('-') != token.npos) {
        // extract range
        insertRange(res, token, convert);
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

  // this is needed to make the compilation work, but never called
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value == false, int> = 0>
  static void insertRange(std::vector<T>&, std::string, std::function<T(std::string const&)>)
  {
  }
};
};

#endif
