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

float BetheBlochF(float betagamma, const std::array<float, 5> p)
{
  // Parameters of the ALEPH Bethe-Bloch formula
  return o2::tpc::Detector::BetheBlochAleph(betagamma, p[0], p[1], p[2], p[3], p[4]);
}

float RelResolutionF(float npoints, const std::array<float, 2> p)
{
  // relative dEdx resolution rel sigma = fRes0*sqrt(1+fResN2/npoint)
  return p[0] * (npoints > 0 ? sqrt(1. + p[1] / npoints) : 1.f);
}

float Param::GetExpectedSignal(float mom, float mass, float charge) const
{
  return mBetheBloch.GetValue(mom / mass) * mMIP * GetChargeFactor(charge);
}

float Param::GetExpectedSigma(float npoints, float tpcsignal) const
{
  return tpcsignal * mRelResolution.GetValue(npoints);
}

} // namespace o2::pid::tpc
