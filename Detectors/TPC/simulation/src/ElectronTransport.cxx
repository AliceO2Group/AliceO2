/// \file ElectronTransport.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/Constants.h"

#include <cmath>

using namespace AliceO2::TPC;

ElectronTransport::ElectronTransport()
  : mRandomGaus()
  , mRandomFlat()
{
  mRandomGaus.initialize(RandomRing::RandomType::Gaus);
  mRandomFlat.initialize(RandomRing::RandomType::Flat);
}

ElectronTransport::~ElectronTransport()
{}

GlobalPosition3D ElectronTransport::getElectronDrift(GlobalPosition3D posEle)
{
  float driftl = posEle.getZ();
  if(driftl<0.01) {
    driftl=0.01;
  }
  driftl = std::sqrt(driftl);
  const float sigT = driftl*DIFFT;
  const float sigL = driftl*DIFFL;
  
  GlobalPosition3D posEleDiffusion((mRandomGaus.getNextValue() * sigT) + posEle.getX(),
                                   (mRandomGaus.getNextValue() * sigT) + posEle.getY(),
                                   (mRandomGaus.getNextValue() * sigL) + posEle.getZ());
  return posEleDiffusion;
}