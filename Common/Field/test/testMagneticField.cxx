// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MagneticField
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "Field/MagneticField.h"
#include "Field/MagFieldFast.h"
#include <memory>
#include "FairLogger.h" // for FairLogger
#include <TStopwatch.h>
#include <TRandom.h>

using namespace o2::field;

BOOST_AUTO_TEST_CASE(MagneticField_test)
{
  // create magnetic field
  std::unique_ptr<MagneticField> fld = std::make_unique<MagneticField>("Maps", "Maps", 1., 1., o2::field::MagFieldParam::k5kG);
  double bz0 = fld->solenoidField();
  LOG(INFO) << "Created default magnetic field for " << bz0 << "kG";
  const double nomBz = 5.00685;
  BOOST_CHECK_CLOSE(bz0, nomBz, 0.1);

  const int ntst = 10000;
  float rnd[3];
  double xyz[ntst][3] = {}, bxyz[ntst][3] = {}, dummyAcc = 0.;
  // fill input
  for (int it = ntst; it--;) {
    gRandom->RndmArray(3, rnd);
    xyz[it][0] = rnd[0] * 400. * TMath::Cos(rnd[1] * TMath::Pi() * 2);
    xyz[it][1] = rnd[1] * 400. * TMath::Sin(rnd[1] * TMath::Pi() * 2);
    xyz[it][2] = (rnd[0] - 0.5) * 250;
  }

  const int repFactor = 50;
  // timing: slow field
  TStopwatch swSlow;
  swSlow.Start();
  for (int ii = repFactor; ii--;) {
    for (int it = ntst; it--;) {
      fld->Field(xyz[it], bxyz[it]);
    }
  }
  swSlow.Stop();

  // init fast field
  fld->AllowFastField(true);

  // timing: fast field
  TStopwatch swFast;
  swFast.Start();
  double bfast[3];
  for (int ii = repFactor; ii--;) {
    for (int it = ntst; it--;) {
      fld->Field(xyz[it], bfast);
    }
  }
  swFast.Stop();
  //
  double sS = swSlow.CpuTime() / (ntst * repFactor);
  double sF = swFast.CpuTime() / (ntst * repFactor);
  double rat = sF > 0. ? sS / sF : -1;
  LOG(INFO) << "Timing: Exact param: " << sS << " Fast param: " << sF
            << "s/call -> factor " << rat;

  // compare slow/fast param precision
  double mean[3] = {0.}, rms[3] = {0.};
  const char comp[] = "XYZ";
  LOG(INFO) << "Relative precision of fast field wrt exact field";
  for (int it = ntst; it--;) {
    fld->Field(xyz[it], bfast);
    for (int i = 0; i < 3; i++) {
      double df = bxyz[it][i] - bfast[i];
      mean[i] += df;
      rms[i] += df * df;
    }
  }
  for (int i = 0; i < 3; i++) {
    mean[i] /= ntst;
    rms[i] /= ntst;
    rms[i] -= mean[i] * mean[i];
    rms[i] = TMath::Sqrt(rms[i]);
    LOG(INFO) << "deltaB" << comp[i] << ": "
              << " mean=" << mean[i] << "(" << mean[i] / nomBz * 100. << "%)"
              << " RMS =" << rms[i] << "(" << rms[i] / nomBz * 100. << "%)";
    BOOST_CHECK(TMath::Abs(mean[i] / nomBz) < 1.e-3);
    BOOST_CHECK(TMath::Abs(rms[i] / nomBz) < 1.e-3);
  }
}
