/// \file ElectronTransport.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/Constants.h"

#include <Vc/Vc>
#include <cmath>

using namespace AliceO2::TPC;

ElectronTransport::ElectronTransport()
  : mRandomGaus()
  , mVc_size(Vc::float_v::Size)
{
  mRandomGaus.initialize(RandomRing::RandomType::Gaus);
}

ElectronTransport::~ElectronTransport()
{}

void ElectronTransport::getElectronDrift(float *posEle)
{
  float driftl=posEle[2];
  if(driftl<0.01) driftl=0.01;
  driftl=std::sqrt(driftl);
  const float sigT = driftl*DIFFT;
  const float sigL = driftl*DIFFL;
  posEle[0]=(mRandomGaus.getNextValue() * sigT) + posEle[0];
  posEle[1]=(mRandomGaus.getNextValue() * sigT) + posEle[1];
  posEle[2]=(mRandomGaus.getNextValue() * sigL) + posEle[2];
}

void ElectronTransport::getElectronDriftVc(float *posEle)
{
  float driftl=posEle[2];
  if(driftl<0.01) driftl=0.01;
  driftl=std::sqrt(driftl);
  const float sigT = driftl*DIFFT;
  const float sigL = driftl*DIFFL;
  const float sig[3] = {sigT, sigT, sigL};
  for(int i = 0; i < 3; i += mVc_size) {
    diffusion(mRandomGaus.getNextValueVc(), Vc::float_v(&sig[i]), Vc::float_v(&posEle[i])).store(&posEle[i]);
  }
}
