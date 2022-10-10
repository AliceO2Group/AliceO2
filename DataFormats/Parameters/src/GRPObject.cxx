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

/// \file GRPObject.cxx
/// \brief Implementation of General Run Parameters object
/// \author ruben.shahoyan@cern.ch

#include <fairlogger/Logger.h>
#include <TFile.h>
#include "DataFormatsParameters/GRPObject.h"
#include <cmath>
#include "CommonConstants/PhysicsConstants.h"
#include "CommonUtils/NameConf.h"

using namespace o2::parameters;
using namespace o2::constants::physics;
using namespace o2::constants::lhc;
using o2::detectors::DetID;

//_______________________________________________
float GRPObject::getSqrtS() const
{
  // get center of mass energy
  double e0 = getBeamEnergyPerNucleon(BeamC);
  double e1 = getBeamEnergyPerNucleon(BeamA);
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
  printf("Run: %8d\nFill: %6d\nPeriod: %s, isMC:%d\n", getRun(), getFill(), getDataPeriod().data(), isMC());
  printf("LHC State: %s\n", getLHCState().data());
  std::time_t t = mTimeStart; // system_clock::to_time_t(mTimeStart);
  printf("Start: %s", std::ctime(&t));
  t = mTimeEnd; // system_clock::to_time_t(mTimeEnd);
  printf("End  : %s", std::ctime(&t));
  printf("1st orbit: %u, %u orbits per TF\n", mFirstOrbit, mNHBFPerTF);
  printf("Beam0: Z:A = %3d:%3d, Energy = %.3f\n", getBeamZ(BeamC), getBeamA(BeamC),
         getBeamEnergyPerNucleon(BeamC));
  printf("Beam1: Z:A = %3d:%3d, Energy = %.3f\n", getBeamZ(BeamA), getBeamA(BeamA),
         getBeamEnergyPerNucleon(BeamA));
  printf("sqrt[s]    = %.3f\n", getSqrtS());
  printf("crossing angle (radian) = %e\n", getCrossingAngle());
  printf("magnet currents (A) L3 = %.3f, Dipole = %.f\n", getL3Current(), getDipoleCurrent());
  printf("Detectors: Cont.RO Triggers\n");
  for (auto i = DetID::First; i <= DetID::Last; i++) {
    if (!isDetReadOut(DetID(i))) {
      continue;
    }
    printf("%9s: ", DetID(i).getName());
    printf("%7s ", isDetContinuousReadOut(DetID(i)) ? "   +   " : "   -   ");
    printf("%7s ", isDetTriggers(DetID(i)) ? "   +   " : "   -   ");
    printf("\n");
  }
}

//_______________________________________________
void GRPObject::setDetROMode(o2::detectors::DetID id, ROMode status)
{
  /// set detector readout mode status
  if (!(status & PRESENT)) {
    remDetReadOut(id);
    return;
  }
  addDetReadOut(id);
  if ((status & CONTINUOUS) == CONTINUOUS) {
    addDetContinuousReadOut(id);
  } else {
    remDetContinuousReadOut(id);
  }
  if ((status & TRIGGERING) == TRIGGERING) {
    addDetTrigger(id);
  } else {
    remDetTrigger(id);
  }
}

//_______________________________________________
GRPObject::ROMode GRPObject::getDetROMode(o2::detectors::DetID id) const
{
  GRPObject::ROMode status = ABSENT;
  if (isDetReadOut(id)) {
    status = PRESENT;
  } else {
    if (isDetContinuousReadOut(id)) {
      status = GRPObject::ROMode(status | CONTINUOUS);
    }
    if (isDetTriggers(id)) {
      status = GRPObject::ROMode(status | TRIGGERING);
    }
  }
  return status;
}

//_______________________________________________
GRPObject* GRPObject::loadFrom(const std::string& grpFileName)
{
  // load object from file
  auto fname = o2::base::NameConf::getGRPFileName(grpFileName);
  TFile flGRP(fname.c_str());
  if (flGRP.IsZombie()) {
    LOG(error) << "Failed to open " << fname;
    throw std::runtime_error("Failed to open GRP file");
  }
  auto grp = reinterpret_cast<o2::parameters::GRPObject*>(flGRP.GetObjectChecked(o2::base::NameConf::CCDBOBJECT.data(), Class()));
  if (!grp && !(grp = reinterpret_cast<o2::parameters::GRPObject*>(flGRP.GetObjectChecked("GRP", Class())))) { // for BWD compatibility
    throw std::runtime_error(fmt::format("Failed to load GRP object from {}", fname));
  }
  return grp;
}
