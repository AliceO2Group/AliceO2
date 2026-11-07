// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testSimRandom.cxx
/// \author Sandro Wenzel (CERN)
/// \brief Checks that o2::base::simRandom() defaults to gRandom and is per thread

#define BOOST_TEST_MODULE Test SimRandom
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DetectorsBase/SimRandom.h"
#include <TRandom.h>
#include <TRandom3.h>
#include <thread>
#include <vector>

BOOST_AUTO_TEST_CASE(defaults_to_gRandom)
{
  BOOST_CHECK(o2::base::simRandom() == gRandom);
  TRandom3 own(1);
  o2::base::setSimRandom(&own);
  BOOST_CHECK(o2::base::simRandom() == &own);
  o2::base::setSimRandom(nullptr);
  BOOST_CHECK(o2::base::simRandom() == gRandom);
}

BOOST_AUTO_TEST_CASE(per_thread_generators)
{
  // each thread draws from its own generator; equal seeds give equal sequences, whatever the other threads do
  constexpr int nthreads = 8;
  constexpr int ndraws = 1000;
  std::vector<std::vector<double>> draws(nthreads);
  std::vector<std::thread> threads;
  for (int t = 0; t < nthreads; ++t) {
    threads.emplace_back([t, &draws]() {
      TRandom3 own(t % 2 + 1);
      o2::base::setSimRandom(&own);
      for (int i = 0; i < ndraws; ++i) {
        draws[t].push_back(o2::base::simRandom()->Rndm());
      }
      o2::base::setSimRandom(nullptr);
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  for (int t = 2; t < nthreads; ++t) {
    BOOST_CHECK(draws[t] == draws[t % 2]);
  }
  BOOST_CHECK(draws[0] != draws[1]);
  // the main thread was never changed
  BOOST_CHECK(o2::base::simRandom() == gRandom);
}
