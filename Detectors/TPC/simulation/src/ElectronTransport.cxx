/// \file ElectronTransport.cxx
/// \brief Implementation of the electron transport
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/Constants.h"

#include <cmath>

using namespace o2::TPC;

ElectronTransport::ElectronTransport()
  : mRandomGaus()
  , mRandomFlat()
{
  mRandomGaus.initialize(RandomRing::RandomType::Gaus);
  mRandomFlat.initialize(RandomRing::RandomType::Flat);
}

ElectronTransport::~ElectronTransport()
= default;

GlobalPosition3D ElectronTransport::getElectronDrift(GlobalPosition3D posEle)
{
  /// For drift lengths shorter than 1 mm, the drift length is set to that value
  float driftl = posEle.getZ();
  if(driftl<0.01) {
    driftl=0.01;
  }
  driftl = std::sqrt(driftl);
  const float sigT = driftl*DIFFT;
  const float sigL = driftl*DIFFL;
  
  /// The position is smeared by a Gaussian with mean around the actual position and a width according to the diffusion coefficient times sqrt(drift length)
  GlobalPosition3D posEleDiffusion((mRandomGaus.getNextValue() * sigT) + posEle.getX(),
                                   (mRandomGaus.getNextValue() * sigT) + posEle.getY(),
                                   (mRandomGaus.getNextValue() * sigL) + posEle.getZ());
  return posEleDiffusion;
}
