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

/// @file   RangeTolenizer.h
/// @author Matthias Richter
/// @since  2018-09-18
/// @brief  Helper function to tokenize sequences and ranges of integral numbers

#include <vector>
#include <string>
#include <sstream>
#include <utility> // std::move

namespace o2
{

struct RangeTokenizer {
  /// tokenize a string according to delimiter ',' and extract values of type T
  template <typename T>
  static std::vector<T> tokenize(std::string input)
  {
    std::istringstream stream(input);
    std::string token;
    std::vector<T> res;
    while (std::getline(stream, token, ',')) {
      T value;
      if (std::is_integral<T>::value && token.find('-') != token.npos) {
        // extract range
        insertRange(res, token);
      } else {
        std::istringstream(token) >> value;
        res.emplace_back(value);
      }
    }
    return std::move(res);
  }

  /// extract a range of an integral type from a token string and add to vector
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value == true, int> = 0>
  static void insertRange(std::vector<T>& res, std::string token)
  {
    std::istringstream tokenstream(token);
    std::string bound;
    T lowerBound, upperBound;
    if (std::getline(tokenstream, bound, '-')) {
      std::istringstream(bound) >> lowerBound;
      if (std::getline(tokenstream, bound, '-')) {
        std::istringstream(bound) >> upperBound;
        for (T index = lowerBound; index <= upperBound; index++) {
          res.emplace_back(index);
        }
      }
    }
  }

  // this is needed to make the compilation work, but never called
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value == false, int> = 0>
  static void insertRange(std::vector<T>& res, std::string token)
  {
  }
};
};

#endif
