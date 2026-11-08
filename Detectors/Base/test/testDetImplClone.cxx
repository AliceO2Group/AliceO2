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

/// \file testDetImplClone.cxx
/// \author Sandro Wenzel (CERN)
/// \brief Checks that a worker clone of a detector gets its own hit buffers

#define BOOST_TEST_MODULE Test DetImplClone
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DetectorsBase/Detector.h"
#include "SimulationDataFormat/BaseHits.h"
#include <memory>

namespace o2::test
{
using TestHit = o2::BasicXYZEHit<float>;

// A minimal detector with one hit vector in shared-memory style buffers
class CloneTestDetector : public o2::base::DetImpl<CloneTestDetector>
{
 public:
  CloneTestDetector() : o2::base::DetImpl<CloneTestDetector>("ITS", true) {}
  CloneTestDetector(const CloneTestDetector& rhs) = default;

  Bool_t ProcessHits(FairVolume*) override { return false; }
  void Register() override {}
  void Reset() override {}
  void InitializeO2Detector() override {}

  std::vector<TestHit>* getHits(int i) const { return i == 0 ? mHits : nullptr; }
  bool initialized() const { return mInitialized; }
  void* cachedBuffer(int b) const { return mCachedPtr[b].empty() ? nullptr : mCachedPtr[b][0]; }
  const void* busyFlag(int b) const { return mShmBusy[b]; }

  std::vector<TestHit>* mHits = nullptr;
};
} // namespace o2::test

namespace o2::base
{
template <>
struct UseShm<o2::test::CloneTestDetector> {
  static constexpr bool value = true;
};
} // namespace o2::base

// the test detector has no ROOT dictionary, so the ClassDef members of its base are defined here
template <>
TClass* o2::base::DetImpl<o2::test::CloneTestDetector>::Class()
{
  return nullptr;
}
template <>
void o2::base::DetImpl<o2::test::CloneTestDetector>::Streamer(TBuffer&)
{
}

BOOST_AUTO_TEST_CASE(clone_gets_own_hit_buffers)
{
  o2::test::CloneTestDetector master;
  master.initializeLate();
  master.BeginEvent();
  BOOST_REQUIRE(master.getHits(0) != nullptr);

  std::unique_ptr<o2::test::CloneTestDetector> clone(static_cast<o2::test::CloneTestDetector*>(master.CloneModule()));
  BOOST_CHECK(!clone->initialized());
  for (int b = 0; b < 3; ++b) {
    BOOST_CHECK(clone->cachedBuffer(b) == nullptr);
    BOOST_CHECK(clone->busyFlag(b) == nullptr);
  }

  clone->initializeLate();
  clone->BeginEvent();
  BOOST_REQUIRE(clone->getHits(0) != nullptr);
  BOOST_CHECK(clone->getHits(0) != master.getHits(0));
  for (int b = 0; b < 3; ++b) {
    BOOST_CHECK(clone->cachedBuffer(b) != master.cachedBuffer(b));
    BOOST_CHECK(clone->busyFlag(b) != master.busyFlag(b));
  }

  // hits of the clone do not reach the master, and the master survives the clone
  clone->getHits(0)->emplace_back();
  BOOST_CHECK(master.getHits(0)->empty());
  clone.reset();
  master.getHits(0)->emplace_back();
  BOOST_CHECK_EQUAL(master.getHits(0)->size(), 1);
}
