// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_LOGGER_H_
#define O2_FRAMEWORK_LOGGER_H_

// FIXME: until we actually have fmt widely available we simply print out the
// format string.
// If FairLogger is not available, we use fmt::printf directly, with no level.
#if __has_include(<fairlogger/Logger.h>)
#include <fairlogger/Logger.h>
#if __has_include(<fmt/format.h>)
#include <fmt/format.h>
#define LOGF(level, ...) LOG(level) << fmt::format(__VA_ARGS__)
#else
#define O2_FIRST_ARG(N, ...) N
#define LOGF(level, ...) LOG(level) << O2_FIRST_ARG(__VA_ARGS__)
#endif
#define O2DEBUG(...) LOGF(debug, __VA_ARGS__)
#define O2INFO(...) LOGF(info, __VA_ARGS__)
#define O2ERROR(...) LOGF(error, __VA_ARGS__)
#elif __has_include(<fmt/format.h>)
#include <fmt/format.h>
#define LOGF(level, ...) fmt::printf(__VA_ARGS__)
#define O2DEBUG(...) LOGF("dummy", __VA_ARGS__)
#define O2INFO(...) LOGF("dummy", __VA_ARGS__)
#define O2ERROR(...) LOGF("dummy", __VA_ARGS__)
#endif

#endif // O2_FRAMEWORK_LOGGER_H_
