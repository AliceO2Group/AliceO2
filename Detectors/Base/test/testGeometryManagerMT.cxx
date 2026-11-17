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

#define BOOST_TEST_MODULE Test GeometryManager multithreaded TGeo navigation
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <TGeoVolume.h>
#include <TGeoMatrix.h>
#include "DetectorsBase/GeometryManager.h"

using o2::base::GeometryManager;

namespace
{
using Ray = std::array<float, 6>;

void buildGeometry()
{
  delete gGeoManager;
  auto* geo = new TGeoManager("testMT", "test geometry");
  auto* vac = new TGeoMaterial("Vac", 0, 0, 0, 1e9, 1e9);
  auto* matA = new TGeoMaterial("A", 27., 13., 2.7, 8.9, 39.);
  auto* matB = new TGeoMaterial("B", 56., 26., 7.9, 1.76, 16.8);
  auto* medVac = new TGeoMedium("Vac", 1, vac);
  auto* medA = new TGeoMedium("A", 2, matA);
  auto* medB = new TGeoMedium("B", 3, matB);
  auto* top = geo->MakeBox("TOP", medVac, 100., 100., 100.);
  geo->SetTopVolume(top);
  // a few daughters, so that rays cross several boundaries
  for (int i = 0; i < 6; i++) {
    auto* v = geo->MakeBox(Form("BOX%d", i), (i % 2) ? medA : medB, 8. + i, 10. + i, 12.);
    top->AddNode(v, i, new TGeoTranslation(-60. + 24. * i, 5. * (i % 3) - 5., 3. * i - 9.));
  }
  geo->CloseGeometry();
}

std::vector<Ray> makeRays(int n)
{
  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> u(-95.f, 95.f);
  std::vector<Ray> rays(n);
  for (auto& r : rays) {
    for (auto& x : r) {
      x = u(gen);
    }
  }
  return rays;
}

struct Res {
  float rho, x2x0, len;
  double extRho, extX2X0;
  int nCross;
};

Res calc(const Ray& r)
{
  auto b = GeometryManager::meanMaterialBudget(r[0], r[1], r[2], r[3], r[4], r[5]); // nav = nullptr
  auto e = GeometryManager::meanMaterialBudgetExt(r[0], r[1], r[2], r[3], r[4], r[5]);
  return {b.meanRho, b.meanX2X0, b.length, e.meanRho, e.meanX2X0, e.nCross};
}

bool same(const Res& a, const Res& b)
{
  auto eq = [](double x, double y) { return std::abs(x - y) <= 1e-5 * (std::abs(x) + std::abs(y)) + 1e-9; };
  return eq(a.rho, b.rho) && eq(a.x2x0, b.x2x0) && eq(a.len, b.len) && eq(a.extRho, b.extRho) && eq(a.extX2X0, b.extX2X0) && a.nCross == b.nCross;
}
} // namespace

BOOST_AUTO_TEST_CASE(GeometryManagerMT)
{
  buildGeometry();
  const auto rays = makeRays(3000);

  // 1. not enabled: nothing changes, the shared navigator is used under the mutex
  BOOST_CHECK_EQUAL(GeometryManager::getMaxThreads(), 0);
  BOOST_CHECK(GeometryManager::getThreadNavigator() == nullptr);
  std::vector<Res> ref;
  for (const auto& r : rays) {
    ref.push_back(calc(r));
  }

  // 2. enabling: idempotent for the same or smaller capacity, refused (capacity unchanged) for a larger one
  constexpr int NThr = 4;
  BOOST_CHECK_EQUAL(GeometryManager::setMaxThreads(NThr), NThr);
  BOOST_CHECK_EQUAL(GeometryManager::setMaxThreads(NThr), NThr);
  BOOST_CHECK_EQUAL(GeometryManager::setMaxThreads(2), NThr);
  BOOST_CHECK_EQUAL(GeometryManager::setMaxThreads(2 * NThr), NThr);
  BOOST_CHECK_EQUAL(GeometryManager::getMaxThreads(), NThr);
  // the calling thread can still navigate (through its own navigator)
  for (size_t i = 0; i < 100; i++) {
    BOOST_CHECK(same(calc(rays[i]), ref[i]));
  }

  // 3. NThr concurrent threads, all started before any finishes: results must agree with the serial reference
  //    and every thread must get its own navigator. The threads are kept alive after the work (their TGeo slots
  //    are consumed for good) to test the exhaustion below deterministically.
  std::atomic<int> nReady{0}, nBad{0};
  std::atomic<bool> go{false}, release{false};
  std::vector<TGeoNavigator*> navs(NThr, nullptr);
  std::vector<std::thread> workers;
  for (int it = 0; it < NThr; it++) {
    workers.emplace_back([&, it]() {
      navs[it] = GeometryManager::getThreadNavigator();
      nReady++;
      while (!go) {
        std::this_thread::yield();
      }
      {
        // alignment-matrix lookup navigates too: it must use the navigator of this thread
        TGeoHMatrix m;
        if (!GeometryManager::getOriginalMatrix("/TOP_1/BOX3_3", m) || std::abs(m.GetTranslation()[0] - 12.) > 1e-9) {
          nBad++;
        }
      }
      for (size_t i = 0; i < rays.size(); i++) {
        if (!same(calc(rays[(i + it * 97) % rays.size()]), ref[(i + it * 97) % rays.size()])) {
          nBad++;
        }
      }
      while (!release) {
        std::this_thread::yield();
      }
    });
  }
  while (nReady < NThr) {
    std::this_thread::yield();
  }
  go = true;
  // 4. more threads than declared: a clear error, no UB, no silent use of the shared navigator
  //    (only after the workers are done with the work, they are still alive)
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  bool thrown = false;
  std::thread extra([&]() {
    try {
      GeometryManager::getThreadNavigator();
    } catch (const std::runtime_error&) {
      thrown = true;
    }
  });
  extra.join();
  BOOST_CHECK(thrown);
  release = true;
  for (auto& w : workers) {
    w.join();
  }
  BOOST_CHECK_EQUAL(nBad.load(), 0);
  std::set<TGeoNavigator*> uniq(navs.begin(), navs.end());
  BOOST_CHECK(!uniq.count(nullptr));
  BOOST_CHECK_EQUAL(uniq.size(), size_t(NThr));

  // 5. geometry replaced: multithreading is disabled for the new one, which works as before (mutexed)
  buildGeometry();
  BOOST_CHECK_EQUAL(GeometryManager::getMaxThreads(), 0);
  BOOST_CHECK(GeometryManager::getThreadNavigator() == nullptr);
  for (size_t i = 0; i < 100; i++) {
    BOOST_CHECK(same(calc(rays[i]), ref[i]));
  }
  // the threads of the previous geometry are gone: their navigators must not have been removed from the new one
  std::thread late([&]() { BOOST_CHECK(same(calc(rays[0]), ref[0])); });
  late.join();
}
