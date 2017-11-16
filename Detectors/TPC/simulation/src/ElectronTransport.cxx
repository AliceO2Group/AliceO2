// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ElectronTransport.cxx
/// \brief Implementation of the electron transport
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCBase/ParameterDetector.h"
#include "TPCSimulation/ElectronTransport.h"

#include <cmath>

using namespace o2::TPC;

ElectronTransport::ElectronTransport()
  : mRandomGaus(),
    mRandomFlat()
{
  mRandomGaus.initialize(RandomRing::RandomType::Gaus);
  mRandomFlat.initialize(RandomRing::RandomType::Flat);
}

ElectronTransport::~ElectronTransport()
= default;

GlobalPosition3D ElectronTransport::getElectronDrift(GlobalPosition3D posEle)
{
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
  /// For drift lengths shorter than 1 mm, the drift length is set to that value
  float driftl = detParam.getTPClength()-std::abs(posEle.Z());
  if(driftl<0.01) {
    driftl=0.01;
  }
  driftl = std::sqrt(driftl);
  const float sigT = driftl*gasParam.getDiffT();
  const float sigL = driftl*gasParam.getDiffL();
  
  /// The position is smeared by a Gaussian with mean around the actual position and a width according to the diffusion coefficient times sqrt(drift length)
  GlobalPosition3D posEleDiffusion((mRandomGaus.getNextValue() * sigT) + posEle.X(),
                                   (mRandomGaus.getNextValue() * sigT) + posEle.Y(),
                                   (mRandomGaus.getNextValue() * sigL) + posEle.Z());
  return posEleDiffusion;
}
