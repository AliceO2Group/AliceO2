// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qconfig_helpers.h
/// \author David Rohr

#ifndef QCONFIG_HELPERS_H
#define QCONFIG_HELPERS_H

#include <string>
#include <sstream>

#define qon_mcat(a, b) a##b
#define qon_mxcat(a, b) qon_mcat(a, b)
#define qon_mcat3(a, b, c) a##b##c
#define qon_mxcat3(a, b, c) qon_mcat3(a, b, c)
#define qon_mstr(a) #a
#define qon_mxstr(a) qon_mstr(a)

namespace qConfig
{
template <class T>
inline std::string print_type(T val)
{
  std::ostringstream s;
  s << val;
  return s.str();
};
template <>
inline std::string print_type<char>(char val)
{
  return std::to_string(val);
};
template <>
inline std::string print_type<unsigned char>(unsigned char val)
{
  return std::to_string(val);
};
template <>
inline std::string print_type<bool>(bool val)
{
  return val ? "true" : "false";
};
} // namespace qConfig

#endif
