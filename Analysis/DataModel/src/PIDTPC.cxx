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
#include "TPCSimulation/Detector.h"

namespace o2::pid::tpc
{

float BetheBlochF(float x, const float p[5])
{
  // float bb = 0;
  float bb = o2::tpc::Detector::BetheBlochAleph(x, p[0], p[1], p[2], p[3], p[4]);
  // AliExternalTrackParam::BetheBlochAleph(betaGamma, fKp1, fKp2, fKp3, fKp4, fKp5);
  // return bb * fMIP;
  return bb;
}

float Param::GetExpectedSignal(float mom, float tpc, float mass, float charge) const
{
  return BetheBloch(mom / mass) * GetChargeFactor(charge);
}

} // namespace o2::pid::tpc
