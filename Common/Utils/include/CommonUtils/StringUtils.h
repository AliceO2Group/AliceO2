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

} // namespace utils
} // namespace o2

#endif //ALICEO2_STRINGUTILS_H
