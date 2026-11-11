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

#define BOOST_TEST_MODULE Test SimSetup StaticTLSReserve
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimSetup/StaticTLSReserve.h"

using o2::withStaticTLSReserve;

BOOST_AUTO_TEST_CASE(missing_or_broken_reserve_is_added)
{
  const std::string reserve = "glibc.rtld.optional_static_tls=32768";
  BOOST_CHECK_EQUAL(withStaticTLSReserve(nullptr, 32768), reserve);
  BOOST_CHECK_EQUAL(withStaticTLSReserve("", 32768), reserve);
  // the value as it arrives in GRID jobs
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls", 32768), reserve);
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=", 32768), reserve);
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=512", 32768), reserve);
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=-1", 32768), reserve);
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=32768x", 32768), reserve);
}

BOOST_AUTO_TEST_CASE(sufficient_reserve_is_kept)
{
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=32768", 32768), "");
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=1048576", 32768), "");
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.malloc.arena_max=4:glibc.rtld.optional_static_tls=65536", 32768), "");
  // glibc also takes hex and octal
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=0x100000", 32768), "");
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=0100000", 32768), "");
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=0x200", 32768), "glibc.rtld.optional_static_tls=32768");
}

BOOST_AUTO_TEST_CASE(malformed_entries_are_dropped)
{
  // glibc ignores the whole string if one entry has no value, so a good reserve next to it does not count
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=32768:glibc.malloc.arena_max", 32768),
                    "glibc.rtld.optional_static_tls=32768");
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.malloc.arena_max:glibc.malloc.tcache_count=0", 32768),
                    "glibc.rtld.optional_static_tls=32768:glibc.malloc.tcache_count=0");
}

BOOST_AUTO_TEST_CASE(other_tunables_are_kept)
{
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.malloc.arena_max=4", 32768),
                    "glibc.rtld.optional_static_tls=32768:glibc.malloc.arena_max=4");
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls:glibc.malloc.arena_max=4", 32768),
                    "glibc.rtld.optional_static_tls=32768:glibc.malloc.arena_max=4");
  // a longer name is a different tunable
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tlsx=1", 32768),
                    "glibc.rtld.optional_static_tls=32768:glibc.rtld.optional_static_tlsx=1");
}

BOOST_AUTO_TEST_CASE(last_reserve_counts)
{
  // glibc applies the entries in order, so a later small reserve wins over an earlier large one
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=65536:glibc.rtld.optional_static_tls=512", 32768),
                    "glibc.rtld.optional_static_tls=32768");
  BOOST_CHECK_EQUAL(withStaticTLSReserve("glibc.rtld.optional_static_tls=512:glibc.rtld.optional_static_tls=65536", 32768), "");
}
