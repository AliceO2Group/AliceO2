// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework RuntimeError
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/RuntimeError.h"
#include <execinfo.h>

BOOST_AUTO_TEST_CASE(TestRuntimeError)
{
  try {
    throw o2::framework::runtime_error("foo");
  } catch (o2::framework::RuntimeErrorRef ref) {
    auto& err = o2::framework::error_from_ref(ref);
    BOOST_CHECK_EQUAL(strncmp(err.what, "foo", 3), 0);
#ifdef DPL_ENABLE_BACKTRACE
    backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
#endif
  }

  try {
    throw o2::framework::runtime_error_f("foo %d", 1);
  } catch (o2::framework::RuntimeErrorRef ref) {
    auto& err = o2::framework::error_from_ref(ref);
    BOOST_CHECK_EQUAL(strncmp(err.what, "foo", 3), 0);
#ifdef DPL_ENABLE_BACKTRACE
    backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
#endif
  }
}
