// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file strtag.h
/// \author David Rohr

#ifndef STRTAG_H
#define STRTAG_H

#include <stdexcept>
#include <string>

template <class T = unsigned long>
#if defined(__cplusplus) && __cplusplus >= 201402L
constexpr
#endif
T qStr2Tag(const char* str)
{
  if (strlen(str) != sizeof(T)) {
    throw std::runtime_error("Invalid tag length");
  }
  T tmp;
  for (unsigned int i = 0; i < sizeof(T); i++) {
    ((char*)&tmp)[i] = str[i];
  }
  return tmp;
}

template <class T>
std::string qTag2Str(const T tag)
{
  T str[2];
  str[0] = tag;
  str[1] = 0;
  return std::string((const char*)str);
}

#endif
