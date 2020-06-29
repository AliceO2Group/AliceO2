// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PID/PIDTOF.h"

namespace o2::pid::tof
{

//_________________________________________________________________________
double Param::GetExpectedSigma(float mom, float time, float evtimereso, float mass) const
{
  //
  // Return the expected sigma of the PID signal for the specified
  // particle mass/Z.
  // If the operation is not possible, return a negative value.
  //

  double dpp = mPar[0] + mPar[1] * mom + mPar[2] * mass / mom; //mean relative pt resolution;
  double sigma = dpp * time / (1. + mom * mom / (mass * mass));

  //   Int_t index = GetMomBin(mom);
  //   double t0res = fT0resolution[index];

  return TMath::Sqrt(sigma * sigma + mPar[3] * mPar[3] / mom / mom + mSigma * mSigma + evtimereso * evtimereso);
}

// //_________________________________________________________________________
// double Param::GetNSigma(float mom, float time, float exptime, float evtime, float evtimereso, float mass) const
// {
//   return (time - evtime - exptime) / GetExpectedSigma(mom, time, evtimereso, mass);
// }

//_________________________________________________________________________
// float Response::ComputeExpectedMomentum(float exptime, float length, float mass)
// {
//   float beta_exp = GetBeta(length, exptime, 0);
//   return (mass * beta_exp / sqrt(1. - (beta_exp * beta_exp)));
// }

//_________________________________________________________________________
float Response::ComputeExpectedTime(float tofexpmom, float length, float massZ) const
{
  const float energy = sqrt((mass * mass) + (expp * expp));
  return length * energy / (kCSPEED * expp);
}

//_________________________________________________________________________
float Response::GetBeta(float length, float time, float evtime) const
{
  if (time <= 0)
    return -999f;
  return length / (time - evtime) / kCSPEED;
}

//_________________________________________________________________________
float Response::GetBetaExpectedSigma(float length, float time, float evtime, float sigmat) const
{
  if (time <= 0)
    return -999f;
  return GetBeta(length, time, evtime) / (time - evtime) * sigmat;
}

//_________________________________________________________________________
float GetExpectedBeta(float mom, float mass) const
{
  if (mom > 0)
    return mom / TMath::Sqrt(mom * mom + mass * mass);
  return 0;
}

} // namespace o2::pid::tof
