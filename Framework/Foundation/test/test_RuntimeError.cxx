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

#include <catch_amalgamated.hpp>
#include "Framework/RuntimeError.h"
#include <unistd.h>
#include <execinfo.h>

TEST_CASE("TestRuntimeError")
{
  try {
    throw o2::framework::runtime_error("foo");
  } catch (o2::framework::RuntimeErrorRef ref) {
    auto& err = o2::framework::error_from_ref(ref);
    REQUIRE(strncmp(err.what, "foo", 3) == 0);
#ifdef DPL_ENABLE_BACKTRACE
    backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
#endif
  }

  try {
    throw o2::framework::runtime_error_f("foo %d", 1);
  } catch (o2::framework::RuntimeErrorRef ref) {
    auto& err = o2::framework::error_from_ref(ref);
    REQUIRE(strncmp(err.what, "foo", 3) == 0);
#ifdef DPL_ENABLE_BACKTRACE
    backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
    o2::framework::demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
#endif
  }
}
