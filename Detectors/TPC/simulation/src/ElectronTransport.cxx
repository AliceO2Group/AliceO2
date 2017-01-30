/// \file ElectronTransport.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/Constants.h"

#include <Vc/Vc>

#include "TMath.h"

using namespace AliceO2::TPC;

using float_v = Vc::Vector<float>;
using sfloat_v = Vc::Scalar::Vector<float>;

ElectronTransport::ElectronTransport()
  : mRandomGaus(),
    mVc_size(float_v::Size)
{
  mRandomGaus.initialize(RandomRing::RandomType::Gaus);
}

ElectronTransport::~ElectronTransport()
{}

void ElectronTransport::getElectronDrift(Float_t *posEle)
{
  Float_t driftl=posEle[2];
  if(driftl<0.01) driftl=0.01;
  driftl=TMath::Sqrt(driftl);
  const Float_t sigT = driftl*DIFFT;
  const Float_t sigL = driftl*DIFFL;
  posEle[0]=(mRandomGaus.getNextValue() * sigT) + posEle[0];
  posEle[1]=(mRandomGaus.getNextValue() * sigT) + posEle[1];
  posEle[2]=(mRandomGaus.getNextValue() * sigL) + posEle[2];
}

void ElectronTransport::getElectronDriftVc(Float_t *posEle)
{
  Float_t driftl=posEle[2];
  if(driftl<0.01) driftl=0.01;
  driftl=TMath::Sqrt(driftl);
  const Float_t sigT = driftl*DIFFT;
  const Float_t sigL = driftl*DIFFL;
  const Float_t sig[3] = {sigT, sigT, sigL};
  for(int i = 0; i < 3; i += mVc_size) {
    diffusion(mRandomGaus.getNextValueVc(), float_v(&sig[i]), float_v(&posEle[i])).store(&posEle[i]);
  }
}
