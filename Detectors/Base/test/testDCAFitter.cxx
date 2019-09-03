// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCTruthContainer class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DetectorsBase/DCAFitter.h"
#include <TRandom.h>

namespace o2
{
namespace base
{

void checkResults(const DCAFitter& fitter, const std::array<float, 3> xyz)
{
  int nCand = fitter.getNCandidates();
  BOOST_CHECK(nCand > 0);
  for (int ic = 0; ic < nCand; ic++) {
    const auto& vtx = fitter.getPCACandidate(ic);
    float dx = vtx.x - xyz[0], dy = vtx.y - xyz[1], dz = vtx.z - xyz[2];
    float dst = TMath::Sqrt(dx * dx + dy * dy + dz * dz);

    const auto &trc0 = fitter.getTrack0(ic), &trc1 = fitter.getTrack1(ic); // track parameters at V0
    printf("Candidate %d: DCA:%+e Vtx: %+e %+e %+e [Diff to true: %+e %+e %+e -> %+e]\n",
           ic, fitter.getChi2AtPCACandidate(ic), vtx.x, vtx.y, vtx.z, dx, dy, dz, dst);
    printf("Track X-parameters at PCA: %+e %+e\n", trc0.getX(), trc1.getX());
    trc0.print();
    trc1.print();
    BOOST_CHECK(dst < 5e-2);
  }
}

BOOST_AUTO_TEST_CASE(PairDCAFitter)
{
  double bz = 5.0;

  // create V0
  std::array<float, 15> cv = {1e-6, 0, 1e-6, 0, 0, 1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 0, 1e-5};
  std::array<float, 5> pr = {0., 0., -0.2, 0.6, 1.};
  DCAFitter::Track t0(0., 0., pr, cv);
  t0.propagateTo(10, bz);

  std::array<float, 3> xyz;
  t0.getXYZGlo(xyz);
  printf("true vertex : %+e %+e %+e\n", xyz[0], xyz[1], xyz[2]);

  DCAFitter::Track tA(t0), tB(t0);
  tB.setParam(-tA.getParam(4), 4);
  tB.setParam(-0.5 * tA.getParam(3), 3);
  tB.setParam(-0.2 * tA.getParam(2), 2);

  tA.rotate(-0.3);
  tB.rotate(0.3);

  printf("True track params at PCA:\n");
  tA.print();
  tB.print();

  tA.propagateTo(tA.getX() + 5., bz);
  tB.propagateTo(tB.getX() + 8., bz);

  DCAFitter df(bz, 10.);

  // check with abs DCA minimization
  {
    df.setUseAbsDCA(true);
    // we may supply track directly to the fitter (they are not modified)
    int nCand = df.process(tA, tB);
    printf("\n\nTesting with abs DCA minimization: %d candidates found\n", nCand);
    // we can have up to 2 candidates
    checkResults(df, xyz);
  }

  // check with weighted DCA minimization
  {
    df.setUseAbsDCA(false);
    // we may supply track directly to the fitter (they are not modified)
    DCAFitter::TrcAuxPar trcAAux(tA, bz), trcBAux(tB, bz);
    int nCand = df.process(tA, trcAAux, tB, trcBAux);
    printf("\n\nTesting with abs DCA minimization: %d candidates found\n", nCand);
    // we can have up to 2 candidates
    checkResults(df, xyz);
  }

  {
    // direct minimization w/o preliminary propagation to XY crossing
    df.setUseAbsDCA(true);
    int nCand = df.processAsIs(tA, tB);
    printf("\n\nTesting with abs DCA minimization w/o taking to XY crossing: %d candidates found\n", nCand);
    checkResults(df, xyz);
  }

  {
    // direct minimization w/o preliminary propagation to XY crossing (errors unreliable)
    df.setUseAbsDCA(false);
    int nCand = df.processAsIs(tA, tB);
    printf("\n\nTesting with weighted DCA minimization w/o taking to XY crossing: %d candidates found\n", nCand);
    checkResults(df, xyz);
  }
}

} // namespace base
} // namespace o2
