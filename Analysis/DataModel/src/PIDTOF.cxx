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

//_________________________________________________________________________
float Response::GetExpectedSigma(DetectorResponse<Response>& response, o2::track::PID::ID id) const
{
  const float x[4] = {mMomentum, mTOFSignal, mEventTime.GetEvTimeReso(mMomentum), o2::track::PID::getMass2Z(id)};
  return response(DetectorResponse<Response>::kSigma, x);
}

} // namespace o2::pid::tof
