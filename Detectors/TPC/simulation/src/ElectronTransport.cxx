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

/// \file ElectronTransport.cxx
/// \brief Implementation of the electron transport
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/ElectronTransport.h"
#include "TPCBase/CDBInterface.h"

#include <cmath>

using namespace o2::tpc;
using namespace o2::math_utils;

ElectronTransport::ElectronTransport() : mRandomGaus(), mRandomFlat(RandomRing<>::RandomType::Flat)
{
  updateParameters();
}

void ElectronTransport::updateParameters(float vdrift)
{
  mGasParam = &(ParameterGas::Instance());
  mDetParam = &(ParameterDetector::Instance());
  mVDrift = vdrift > 0 ? vdrift : mGasParam->DriftV;
}

GlobalPosition3D ElectronTransport::getElectronDrift(GlobalPosition3D posEle, float& driftTime)
{
  /// For drift lengths shorter than 1 mm, the drift length is set to that value
  float driftl = mDetParam->TPClength - std::abs(posEle.Z());
  if (driftl < 0.01) {
    driftl = 0.01;
  }
  driftl = std::sqrt(driftl);
  const float sigT = driftl * mGasParam->DiffT;
  const float sigL = driftl * mGasParam->DiffL;

  /// The position is smeared by a Gaussian with mean around the actual position and a width according to the diffusion
  /// coefficient times sqrt(drift length)
  GlobalPosition3D posEleDiffusion((mRandomGaus.getNextValue() * sigT) + posEle.X(),
                                   (mRandomGaus.getNextValue() * sigT) + posEle.Y(),
                                   (mRandomGaus.getNextValue() * sigL) + posEle.Z());

  /// If there is a sign change in the z position, the hit has changed sides
  /// This is not possible, but rather just an elongation of the drift time.
  /// In such cases, the old z position of the hit is returned, and the drift time is computed accordingly

  if (posEle.Z() / posEleDiffusion.Z() < 0.f) {
    driftTime = getDriftTime(posEleDiffusion.Z(), -1.f);
    posEleDiffusion.SetZ(posEle.Z());

  } else {
    driftTime = getDriftTime(posEleDiffusion.Z());
  }

  return posEleDiffusion;
}

bool ElectronTransport::isCompletelyOutOfSectorCoarseElectronDrift(GlobalPosition3D posEle, const Sector& sector) const
{
  /// For drift lengths shorter than 1 mm, the drift length is set to that value
  float driftl = mDetParam->TPClength - std::abs(posEle.Z());
  if (driftl < 0.01) {
    driftl = 0.01;
  }
  driftl = std::sqrt(driftl);

  /// Three sigma of the expected average transverse diffusion
  const float threeSigmaT = 3.f * driftl * mGasParam->DiffT;

  auto& mapper = Mapper::instance();
  return mapper.isOutOfSector(posEle, sector, threeSigmaT);
}
