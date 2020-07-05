// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_TRACING_H_
#define O2_FRAMEWORK_TRACING_H_

#if DPL_ENABLE_TRACING && __has_include(<tracy/Tracy.hpp>)
#define DPL_HAS_TRACING
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped \
  while (false) {  \
  }
#define FrameMark \
  while (false) { \
  }
#define TracyPlot(...) \
  while (false) {      \
  }
#define ZoneScopedN(...) \
  while (false) {        \
  }
#define ZoneScopedNS(...) \
  while (false) {         \
  }
#define TracyAlloc(...) \
  while (false) {       \
  }
#define TracyFree(...) \
  while (false) {      \
  }
#define TracyAppInfo(...) \
  while (false) {         \
  }
#endif

#endif // O2_FRAMEWORK_TRACING_H_
