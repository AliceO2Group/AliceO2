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

///
/// \file   StringUtils.h
/// \author Barthelemy von Haller
///

#ifndef ALICEO2_STRINGUTILS_H
#define ALICEO2_STRINGUTILS_H

#include <sstream>
#include <vector>
#include <algorithm>
#include <fmt/format.h>
#include <Rtypes.h>

namespace o2
{
namespace utils
{

struct Str {

  // Code for trimming coming from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring

  /**
 * Trim from start (in place)
 * @param s
 */
  static inline void ltrim(std::string& s)
  {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !std::isspace(ch); }));
  }

  static inline void ltrim(std::string& s, const std::string& start)
  {
    if (beginsWith(s, start)) {
      s.erase(0, start.size());
    }
  }

  /** Trim from end (in place)
 *
 * @param s
 */
  static inline void rtrim(std::string& s)
  {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(), s.end());
  }

  static inline void rtrim(std::string& s, const std::string& ending)
  {
    if (endsWith(s, ending)) {
      s.erase(s.size() - ending.size(), ending.size());
    }
  }

  /**
 * Trim from both ends (in place)
 * @param s
 */
  static inline void trim(std::string& s)
  {
    ltrim(s);
    rtrim(s);
  }

  /**
 * Trim from start (copying)
 * @param s
 * @return
 */
  static inline std::string ltrim_copy(const std::string& s)
  {
    std::string ss = s;
    ltrim(ss);
    return ss;
  }

  static inline std::string ltrim_copy(const std::string& s, const std::string& start)
  {
    std::string ss = s;
    ltrim(ss, start);
    return ss;
  }

  /**
 * Trim from end (copying)
 * @param s
 * @return
 */
  static inline std::string rtrim_copy(const std::string& s)
  {
    std::string ss = s;
    rtrim(ss);
    return ss;
  }

  static inline std::string rtrim_copy(const std::string& s, const std::string& ending)
  {
    std::string ss = s;
    rtrim(ss, ending);
    return ss;
  }

  /**
 * Trim from both sides (copying)
 * @param s
 * @return
 */
  static inline std::string trim_copy(const std::string& s)
  {
    std::string ss = s;
    rtrim(ss);
    return ss;
  }

  static inline bool endsWith(const std::string& s, const std::string& ending)
  {
    return (ending.size() > s.size()) ? false : std::equal(ending.rbegin(), ending.rend(), s.rbegin());
  }

  static inline bool beginsWith(const std::string& s, const std::string& start)
  {
    return (start.size() > s.size()) ? false : std::equal(start.begin(), start.end(), s.begin());
  }

  // return vector of tokens from the string with provided delimiter. If requested, trim the spaces from tokens
  static std::vector<std::string> tokenize(const std::string& src, char delim, bool trimToken = true, bool skipEmpty = true);

  // concatenate arbitrary number of strings
  template <typename... Ts>
  static std::string concat_string(Ts const&... ts)
  {
    std::stringstream s;
    (s << ... << ts);
    return s.str();
  }

  // generate random string of given length, suitable for file names
  static std::string getRandomString(int length);

  // Check if the path exists
  static bool pathExists(const std::string_view p);

  // Check if the path is a directory
  static bool pathIsDirectory(const std::string_view p);

  // create full path
  static std::string getFullPath(const std::string_view p);

  // rectify directory, applying convention "none"==""
  static std::string rectifyDirectory(const std::string_view p);

  // create unique non-existing path name starting with prefix. Loose equivalent of boost::filesystem::unique_path()
  // in absence of such a function in std::filesystem
  static std::string create_unique_path(const std::string_view prefix = "", int length = 16);

  ClassDefNV(Str, 1);
};

} // namespace utils
} // namespace o2

#endif //ALICEO2_STRINGUTILS_H
