/// \file ElectronTransport.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/Constants.h"

#include "TMath.h"

using namespace AliceO2::TPC;

ElectronTransport::ElectronTransport()
  : mRandomGaus()
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
  Float_t sigT = driftl*DIFFT;
  Float_t sigL = driftl*DIFFL;
  posEle[0]=(mRandomGaus.getNextValue() * sigT) + posEle[0];
  posEle[1]=(mRandomGaus.getNextValue() * sigT) + posEle[1];
  posEle[2]=(mRandomGaus.getNextValue() * sigL) + posEle[2];
}