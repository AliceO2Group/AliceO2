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
#include "TMath.h"

namespace o2::pid::tof
{

//_________________________________________________________________________
Double_t Param::GetExpectedSigma(Float_t mom, Float_t time, Float_t evtimereso, Float_t mass) const
{
  //
  // Return the expected sigma of the PID signal for the specified
  // particle mass/Z.
  // If the operation is not possible, return a negative value.
  //

  Double_t dpp = mPar[0] + mPar[1] * mom + mPar[2] * mass / mom; //mean relative pt resolution;
  Double_t sigma = dpp * time / (1. + mom * mom / (mass * mass));

  //   Int_t index = GetMomBin(mom);
  //   Double_t t0res = fT0resolution[index];

  return TMath::Sqrt(sigma * sigma + mPar[3] * mPar[3] / mom / mom + mSigma * mSigma + evtimereso * evtimereso);
}

//_________________________________________________________________________
Double_t Param::GetNSigma(Float_t mom, Float_t time, Float_t exptime, Float_t evtime, Float_t evtimereso, Float_t mass) const
{
  return (time - evtime - exptime) / GetExpectedSigma(mom, time, evtimereso, mass);
}

//_________________________________________________________________________
float beta(float l, float t, float t0)
{
  if (t <= 0)
    return -999;
  return l / (t - t0) / 0.029979246;
}

//_________________________________________________________________________
float betaerror(float l, float t, float t0, float sigmat)
{
  if (t <= 0)
    return -999;
  return beta(l, t, t0) / (t - t0) * sigmat;
}

//_________________________________________________________________________
float expbeta(float p, float m)
{
  if (p > 0)
    return p / TMath::Sqrt(p * p + m * m);
  return 0;
}

//_________________________________________________________________________
float p(float eta, float signed1Pt)
{
  return cosh(eta) / fabs(signed1Pt);
}

} // namespace o2::pid::tof