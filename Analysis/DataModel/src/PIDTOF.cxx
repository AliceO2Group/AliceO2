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
uint EventTime::GetMomBin(float mom) const
{
  for (int i = 0; i < mNmomBins; i++)
    if (abs(mom) < mMomBins[i + 1])
      return i;
  return mNmomBins - 1;
}

//_________________________________________________________________________
double Param::GetExpectedSigma(float mom, float time, float evtimereso, float mass) const
{
  mom = abs(mom);
  if (mom <= 0)
    return -999;
  double dpp = mPar[0] + mPar[1] * mom + mPar[2] * mass / mom; //mean relative pt resolution;
  double sigma = dpp * time / (1. + mom * mom / (mass * mass));
  return TMath::Sqrt(sigma * sigma + mPar[3] * mPar[3] / mom / mom + mSigma * mSigma + evtimereso * evtimereso);
}

//_________________________________________________________________________
float Response::ComputeExpectedTime(float tofexpmom, float length, float massZ)
{
  const float energy = sqrt((massZ * massZ) + (tofexpmom * tofexpmom));
  return length * energy / (kCSPEED * tofexpmom);
}

//_________________________________________________________________________
float Response::GetBeta(float length, float time, float evtime)
{
  if (time <= 0)
    return -999.f;
  return length / (time - evtime) / kCSPEED;
}

//_________________________________________________________________________
float Response::GetBetaExpectedSigma(float length, float time, float evtime, float sigmat)
{
  if (time <= 0)
    return -999.f;
  return GetBeta(length, time, evtime) / (time - evtime) * sigmat;
}

//_________________________________________________________________________
float Response::GetExpectedBeta(float mom, float mass)
{
  if (mom > 0)
    return mom / TMath::Sqrt(mom * mom + mass * mass);
  return 0;
}

} // namespace o2::pid::tof
