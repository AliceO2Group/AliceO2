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

/// \file testTRDIonisationModel.cxx
/// \brief Checks the TRD gas collision model against its own tables

#define BOOST_TEST_MODULE Test TRD IonisationModel
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TRDSimulation/IonisationModel.h"

#include <TRandom3.h>
#include <cmath>

namespace o2
{
namespace trd
{

/// The sampled mean energy per cm reproduces the tabulated restricted dE/dx
BOOST_AUTO_TEST_CASE(TRDIonisationModel_meanEnergy)
{
  TRandom3 rnd(4711);
  for (auto gas : {SimParam::GasMixture::Xenon, SimParam::GasMixture::Argon}) {
    IonisationModel model(gas);
    for (double bg : {0.08, 0.5, 4., 30., 3000.}) {
      const int n = 20000;
      double sum = 0.;
      for (int i = 0; i < n; ++i) {
        sum += model.stepEnergy(1., bg, 1., rnd);
      }
      BOOST_TEST(sum / n == model.restrictedDEdx(bg), boost::test_tools::tolerance(0.02));
    }
  }
}

/// The fraction of steps without a collision is Poisson, and scales with the charge squared
BOOST_AUTO_TEST_CASE(TRDIonisationModel_zeroCollisions)
{
  TRandom3 rnd(815);
  IonisationModel model(SimParam::GasMixture::Xenon);
  const double bg = 4., length = 0.02;
  for (double z2 : {1., 4.}) {
    const int n = 100000;
    int zero = 0;
    for (int i = 0; i < n; ++i) {
      zero += model.stepEnergy(length, bg, z2, rnd) == 0.;
    }
    const double expected = std::exp(-length * z2 * model.collisionsPerCm(bg));
    BOOST_TEST(double(zero) / n == expected, boost::test_tools::tolerance(0.02));
  }
}

/// The collision rate has its minimum near beta*gamma 3-4, at 40-47 collisions per cm in Xe/CO2, and a plateau
BOOST_AUTO_TEST_CASE(TRDIonisationModel_minimum)
{
  IonisationModel model(SimParam::GasMixture::Xenon);
  double bgMin = 0., nMin = 1e9;
  for (double lbg = 0.; lbg <= 1.; lbg += 0.01) {
    const double n = model.collisionsPerCm(std::pow(10., lbg));
    if (n < nMin) {
      nMin = n;
      bgMin = std::pow(10., lbg);
    }
  }
  BOOST_TEST(nMin > 40.);
  BOOST_TEST(nMin < 47.);
  BOOST_TEST(bgMin > 2.5);
  BOOST_TEST(bgMin < 5.);
  BOOST_TEST(model.collisionsPerCm(1e5) == model.collisionsPerCm(1e6));
}

/// A single collision transfers between the first grid energy (the ionisation potential) and the delta-ray cut
BOOST_AUTO_TEST_CASE(TRDIonisationModel_collisionEnergy)
{
  TRandom3 rnd(42);
  IonisationModel model(SimParam::GasMixture::Xenon);
  // steps short enough that almost every step with energy has exactly one collision
  const double length = 1e-4, bg = 4.;
  int n = 0;
  double sum = 0.;
  for (int i = 0; i < 2000000; ++i) {
    const double e = model.stepEnergy(length, bg, 1., rnd);
    if (e > 0.) {
      BOOST_TEST(e >= 12.);
      BOOST_TEST(e <= IonisationModel::deltaRayCut());
      sum += e;
      ++n;
    }
  }
  BOOST_TEST(n > 5000);
  BOOST_TEST(sum / n == model.restrictedDEdx(bg) / model.collisionsPerCm(bg), boost::test_tools::tolerance(0.2));
}

} // namespace trd
} // namespace o2
