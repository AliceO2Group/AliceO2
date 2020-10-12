// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PID/PIDTPC.h"

namespace o2::pid::tpc
{

void Response::UpdateTrack(float mom, float tpcsignal, float tpcpoints)
{
  mMomentum = mom;
  mTPCSignal = tpcsignal;
  mTPCPoints = tpcpoints;
};

float Response::GetExpectedSignal(DetectorResponse<Response>& response, o2::track::PID::ID id) const
{
  const float x[2] = {mMomentum / o2::track::PID::getMass(id), (float)o2::track::PID::getCharge(id)};
  return response(DetectorResponse<Response>::kSignal, x);
}

float Response::GetExpectedSigma(DetectorResponse<Response>& response, o2::track::PID::ID id) const
{
  const float x[2] = {mTPCSignal, mTPCPoints};
  return response(DetectorResponse<Response>::kSigma, x);
}

} // namespace o2::pid::tpc
