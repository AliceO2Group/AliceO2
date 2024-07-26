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

#include "Framework/RuntimeError.h"

#include <cstdio>
#include <climits>
#include <atomic>
#include <cstdarg>
#include <cstring>
#include <unistd.h>
#include <cstdlib>
#include <cxxabi.h>
#include <execinfo.h>
#include <stdexcept>

namespace o2::framework
{

namespace
{
static RuntimeError gError[RuntimeError::MAX_RUNTIME_ERRORS];
static std::atomic<bool> gErrorBooking[RuntimeError::MAX_RUNTIME_ERRORS];

bool canDumpBacktrace()
{
#ifdef DPL_ENABLE_BACKTRACE
  return true;
#else
  return false;
#endif
}
} // namespace

void clean_all_runtime_errors()
{
  for (auto& i : gErrorBooking) {
    i.store(false);
  }
}

void clean_runtime_error(int i)
{
  gErrorBooking[i].store(false);
}

RuntimeError& error_from_ref(RuntimeErrorRef ref)
{
  return gError[ref.index];
}

RuntimeErrorRef runtime_error_f(const char* format, ...)
{
  int i = 0;
  bool expected = false;
  while (gErrorBooking[i].compare_exchange_strong(expected, true) == false) {
    ++i;
    if (i >= RuntimeError::MAX_RUNTIME_ERRORS) {
      throw std::runtime_error("Too many o2::framework::runtime_error thrown without proper cleanup.");
    }
  }
  va_list args;
  va_start(args, format);
  vsnprintf(gError[i].what, RuntimeError::MAX_RUNTIME_ERROR_SIZE, format, args);
  va_end(args);
  gError[i].maxBacktrace = canDumpBacktrace() ? backtrace(gError[i].backtrace, BacktraceHelpers::MAX_BACKTRACE_SIZE) : 0;
  return RuntimeErrorRef{i};
}

RuntimeErrorRef runtime_error(const char* s)
{
  int i = 0;
  bool expected = false;
  while (gErrorBooking[i].compare_exchange_strong(expected, true) == false) {
    ++i;
  }
  strncpy(gError[i].what, s, RuntimeError::MAX_RUNTIME_ERROR_SIZE);
  gError[i].maxBacktrace = canDumpBacktrace() ? backtrace(gError[i].backtrace, BacktraceHelpers::MAX_BACKTRACE_SIZE) : 0;
  return RuntimeErrorRef{i};
}

void throw_error(RuntimeErrorRef ref)
{
  throw ref;
}

} // namespace o2::framework
