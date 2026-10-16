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

/// \file qconfig_rtc.h
/// \author David Rohr

#ifndef QCONFIG_RTC_H
#define QCONFIG_RTC_H

#include "qconfig.h"
#include "qconfig_helpers.h"

template <class T>
static std::string qConfigPrintRtc(const T& tSrc, bool useConstexpr)
{
  std::stringstream out;
  out << std::hexfloat;
#define QCONFIG_PRINT_RTC
#include "qconfig_internal.h"
#undef QCONFIG_PRINT_RTC
  return out.str();
}

#endif
