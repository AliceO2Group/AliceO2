// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qconfigrtc.h
/// \author David Rohr

#ifndef QCONFIG_RTC_H
#define QCONFIG_RTC_H

#include "qconfig.h"
#include "qconfig_helpers.h"

#ifndef qon_mxstr
#define qon_mstr(a) #a
#define qon_mxstr(a) qon_mstr(a)
#endif

template <class T>
static std::string qConfigPrintRtc(const T& tSrc, bool useConstexpr)
{
#if defined(__cplusplus) && __cplusplus >= 201703L
  std::stringstream out;
#define QCONFIG_PRINT_RTC
#include "qconfig.h"
#undef QCONFIG_PRINT_RTC
  return out.str();
#else
  throw std::runtime_error("not supported");
#endif
}

#define QCONFIG_CONVERT_RTC
#include "qconfig.h"
#undef QCONFIG_CONVERT_RTC

#endif
