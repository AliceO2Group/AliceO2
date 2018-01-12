// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GRPObject.cxx
/// \brief Implementation of General Run Parameters object
/// \author ruben.shahoyan@cern.ch

#include "DataFormatsParameters/GRPObject.h"
#include <cmath>
#include "CommonConstants/PhysicsConstants.h"

using namespace o2::parameters;
using namespace o2::constants::physics;
using namespace o2::constants::lhc;
using o2::Base::DetID;

//_______________________________________________
float GRPObject::getSqrtS() const
{
  // get center of mass energy
  double e0 = getBeamEnergyPerNucleon(BeamClockWise);
  double e1 = getBeamEnergyPerNucleon(BeamAntiClockWise);
  if (e0 <= MassProton || e1 <= MassProton) {
    return 0.f;
  }
  double beta0 = 1. - MassProton * MassProton / (e0 * e0);
  double beta1 = 1. - MassProton * MassProton / (e1 * e1);
  beta0 = beta0 > 0 ? sqrt(beta0) : 0.;
  beta1 = beta1 > 0 ? sqrt(beta1) : 0.;
  double ss = 2. * (MassProton * MassProton + e0 * e1 * (1. + beta0 * beta1 * cos(mCrossingAngle)));
  return ss > 0. ? sqrt(ss) : 0.;
}

//_______________________________________________
void GRPObject::print() const
{
  // print itself
  printf("Run: %8d\nFill: %6d\nPeriod: %s\n", getRun(), getFill(), getDataPeriod().data());
  printf("LHC State: %s\n", getLHCState().data());
  std::time_t t = mTimeStart; //system_clock::to_time_t(mTimeStart);
  printf("Start: %s", std::ctime(&t));
  t = mTimeEnd; //system_clock::to_time_t(mTimeEnd);
  printf("End  : %s", std::ctime(&t));
  printf("Beam0: Z:A = %3d:%3d, Energy = %.3f\n", getBeamZ(BeamClockWise), getBeamA(BeamClockWise),
         getBeamEnergyPerNucleon(BeamClockWise));
  printf("Beam1: Z:A = %3d:%3d, Energy = %.3f\n", getBeamZ(BeamAntiClockWise), getBeamA(BeamAntiClockWise),
         getBeamEnergyPerNucleon(BeamAntiClockWise));
  printf("sqrt[s]    = %.3f\n", getSqrtS());
  printf("crossing angle (radian) = %e\n", getCrossingAngle());
  printf("magnet currents (A) L3 = %.3f, Dipole = %.f\n", getL3Current(), getDipoleCurrent());
  printf("Detectors in readout: ");
  for (auto i = DetID::First; i <= DetID::Last; i++) {
    if (isDetReadOut(DetID(i))) {
      printf(" %s", DetID(i).getName());
    }
  }
  printf("\n");
  printf("Detectors in trigger: ");
  for (auto i = DetID::First; i <= DetID::Last; i++) {
    if (isDetTriggers(DetID(i))) {
      printf(" %s", DetID(i).getName());
    }
  }
  printf("\n");
}
