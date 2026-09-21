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

#define BOOST_TEST_MODULE Test MagneticField
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "Field/MagneticField.h"
#include "Field/MagFieldFast.h"
#include <memory>
#include <fairlogger/Logger.h> // for FairLogger
#include <TFile.h>
#include <TStopwatch.h>
#include <TRandom.h>

using namespace o2::field;

BOOST_AUTO_TEST_CASE(MagneticField_test)
{
  // create magnetic field
  std::unique_ptr<MagneticField> fld = std::make_unique<MagneticField>("Maps", "Maps", 1., 1., o2::field::MagFieldParam::k5kG);
  double bz0 = fld->solenoidField();
  LOG(info) << "Created default magnetic field for " << bz0 << "kG";
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
  LOG(info) << "Timing: Exact param: " << sS << " Fast param: " << sF
            << "s/call -> factor " << rat;

  // compare slow/fast param precision
  double mean[3] = {0.}, rms[3] = {0.};
  const char comp[] = "XYZ";
  LOG(info) << "Relative precision of fast field wrt exact field";
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
    LOG(info) << "deltaB" << comp[i] << ": "
              << " mean=" << mean[i] << "(" << mean[i] / nomBz * 100. << "%)"
              << " RMS =" << rms[i] << "(" << rms[i] / nomBz * 100. << "%)";
    BOOST_CHECK(TMath::Abs(mean[i] / nomBz) < 1.e-3);
    BOOST_CHECK(TMath::Abs(rms[i] / nomBz) < 1.e-3);
  }
}

BOOST_AUTO_TEST_CASE(MagneticField_reinitialization_test)
{
  // The measured map is transient, so a MagneticField read back from a file has to be
  // re-created before it can be used. That must reproduce the field vectors, not merely
  // their magnitude: a sign flip leaves |B| untouched.
  const double points[][3] = {
    {0., 0., 0.},      // solenoid, on axis
    {100., 50., 100.}, // solenoid, off axis
    {10., 10., -900.}, // muon dipole
    {0., 0., 1000.},   // compensator 1A, side A
    {0., 0., -2049.},  // compensator 2C, side C
    {0., 0., 2049.}    // compensator 2A, side A
  };
  const int npoints = sizeof(points) / sizeof(points[0]);
  const double tolerance = 1.e-9; // kGauss

  std::unique_ptr<MagneticField> fld = std::make_unique<MagneticField>("Maps", "Maps", 1., 1., MagFieldParam::k5kG);
  const double facSol = fld->getFactorSolenoid(), facDip = fld->getFactorDipole();
  double bref[npoints][3] = {};
  for (int ip = 0; ip < npoints; ip++) {
    fld->Field(points[ip], bref[ip]);
    // a point where the field vanishes would make the comparisons below vacuous
    BOOST_CHECK(TMath::Abs(bref[ip][0]) + TMath::Abs(bref[ip][1]) + TMath::Abs(bref[ip][2]) > tolerance);
  }

  const char* fname = "testMagneticFieldReinitialization.root";
  {
    TFile fout(fname, "recreate");
    fout.WriteObject(fld.get(), "field");
  }
  TFile fin(fname);
  auto* fldRead = fin.Get<MagneticField>("field");
  BOOST_REQUIRE(fldRead != nullptr);
  fldRead->CreateField();
  BOOST_CHECK_EQUAL(fldRead->getFactorSolenoid(), facSol);
  BOOST_CHECK_EQUAL(fldRead->getFactorDipole(), facDip);
  for (int ip = 0; ip < npoints; ip++) {
    double b[3] = {};
    fldRead->Field(points[ip], b);
    for (int i = 0; i < 3; i++) {
      BOOST_CHECK_SMALL(b[i] - bref[ip][i], tolerance);
    }
  }
}
