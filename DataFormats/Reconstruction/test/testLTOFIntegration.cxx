// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TrackLTIntegral class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "ReconstructionDataFormats/PID.h"
#include "CommonConstants/PhysicsConstants.h"

namespace o2
{

// L,ToF container update test
BOOST_AUTO_TEST_CASE(TrackLTIntegral)
{
  o2::track::TrackPar trc;
  trc.setQ2Pt(2);
  trc.setSnp(0.1);
  auto trc1 = trc;
  trc1.setTgl(0.5);
  o2::track::TrackLTIntegral lt, lt1;
  const int nStep = 100;
  const float dx2x0 = 0.01f;
  for (int i = 0; i < nStep; i++) {
    lt.addStep(1., trc);
    lt1.addStep(1., trc1);
    lt1.addX2X0(dx2x0);
  }
  trc.printParam();
  lt.print();
  trc1.printParam();
  lt1.print();
  float tc = lt.getL() * 1000.f / o2::constants::physics::LightSpeedCm2NS; // fastest time
  printf("TOF @ speed of light: %7.1f ps\n", tc);
  for (int i = 1; i < lt.getNTOFs(); i++) {
    BOOST_CHECK(tc < lt.getTOF(i));            // nothing is faster than the light
    BOOST_CHECK(lt1.getTOF(i) < lt.getTOF(i)); // higher P track is faster
  }
  BOOST_CHECK_CLOSE(lt1.getX2X0(), nStep * dx2x0, 1e-4);
}

} // namespace o2
