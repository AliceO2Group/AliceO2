// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Utils
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <TMath.h>
#include <TProfile.h>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include "MathUtils/Utils.h"

using namespace o2;
using namespace utils;

BOOST_AUTO_TEST_CASE(Utils_test)
{
  // test MathUtils/Utils.h

  { // test FastATan2()
    int n = 1000;
    TProfile* p = new TProfile("pFastATan2", "FastATan2(y,x)", n, -TMath::Pi(), TMath::Pi());
    double maxDiff = 0;
    for (int i = 0; i < n; i++) {
      double phi0 = -TMath::Pi() + i * TMath::TwoPi() / n;
      float x = TMath::Cos(phi0);
      float y = TMath::Sin(phi0);
      float phi = utils::FastATan2(y, x);
      double diff = phi - phi0;
      p->Fill(phi0, diff);
      diff = fabs(diff);
      if (diff > maxDiff)
        maxDiff = diff;
    }
    //p->Draw();

    std::cout << "test FastATan2:" << std::endl;
    std::cout << " Max inaccuracy " << maxDiff << std::endl;

    // test the speed
    std::cout << " Speed: " << std::endl;

    const int M = 1000;
    uint32_t iterations = 10000;
    float sum = 0;
    float vx[M], vy[M];
    double vxd[M], vyd[M];

    // set some arbitrary x, y values
    {
      float x = 1.e-4, y = 1.e-3, d = 1.e-5;
      for (int i = 0; i < M; ++i, x += d, y += d) {
        vx[i] = (i % 2) ? x : -x;
        vy[i] = (i % 3) ? y : -y;
        vxd[i] = vx[i];
        vyd[i] = vy[i];
      }
    }

    double scale = 1. / iterations / M;
    // dry run
    sum = 0;
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      for (int j = 0; j < M; ++j) {
        sum += vx[j] + vy[j];
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time1 = scale * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << "  dry run: time " << time1 << " ns. checksum " << sum << std::endl;

    // double precision
    double dsum = 0;
    begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      for (int j = 0; j < M; ++j) {
        dsum += atan2(vyd[j], vxd[j]);
      }
    }
    end = std::chrono::high_resolution_clock::now();
    auto time2 = scale * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << "  atan2(): time " << time2 << " ns. checksum " << dsum << std::endl;

    // single precision
    sum = 0;
    begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      for (int j = 0; j < M; ++j) {
        sum += atan2f(vy[j], vx[j]);
      }
    }
    end = std::chrono::high_resolution_clock::now();
    auto time3 = scale * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << "  atan2f(): time " << time3 << " ns. checksum " << sum << std::endl;

    // fast method
    sum = 0;
    begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      for (int j = 0; j < M; ++j) {
        sum += utils::FastATan2(vy[j], vx[j]);
      }
    }
    end = std::chrono::high_resolution_clock::now();
    auto time4 = scale * std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << "  FastATan2: time " << time4 << " ns. checksum " << sum << std::endl;
    std::cout << "  speed up to atan2f(): " << (time3 - time1) / (time4 - time1) << " times " << std::endl;

    BOOST_CHECK(maxDiff < 1.e-3);

  } // test FastATan2()
}
