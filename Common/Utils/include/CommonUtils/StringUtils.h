// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <sys/stat.h>
#include <cstdlib>
#include <fmt/format.h>

namespace o2
{
namespace utils
{

// Code for trimming coming from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring

/**
 * Trim from start (in place)
 * @param s
 */
static inline void ltrim(std::string& s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
          }));
}

/** Trim from end (in place)
 *
 * @param s
 */
static inline void rtrim(std::string& s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
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
static inline std::string ltrim_copy(std::string s)
{
  ltrim(s);
  return s;
}

/**
 * Trim from end (copying)
 * @param s
 * @return
 */
static inline std::string rtrim_copy(std::string s)
{
  rtrim(s);
  return s;
}

// concatenate arbitrary number of strings
template <typename... Ts>
std::string concat_string(Ts const&... ts)
{
  std::stringstream s;
  (s << ... << ts);
  return s.str();
}

// Check if the path exists  
static inline bool pathExists(const std::string_view p)
{
  struct stat buffer;
  return (stat(p.data(), &buffer) == 0);
}
  
// Check if the path is a directory
static inline bool pathIsDirectory(const std::string_view p)
{
  struct stat buffer;
  return (stat(p.data(), &buffer) == 0) && S_ISDIR(buffer.st_mode);
}

static inline std::string getFullPath(const std::string_view p)
{
  std::unique_ptr<char[]> real_path(realpath(p.data(), nullptr));
  return std::string(real_path.get());
}

static inline std::string rectifyDirectory(const std::string& _dir)
{
  std::string dir = _dir;
  if (dir.empty() || dir == "none") {
    dir = "";
  } else {
    dir = getFullPath(dir);
    if (!pathIsDirectory(dir)) {
      throw std::runtime_error(fmt::format("{:s} is not an accessible directory", dir));
    } else {
      dir += '/';
    }
  }
  return dir;
}

  
} // namespace utils
} // namespace o2

#endif //ALICEO2_STRINGUTILS_H
