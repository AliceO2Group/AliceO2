/// \file GEMAmplification.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/Constants.h"

using namespace AliceO2::TPC;
using boost::format;

GEMAmplification::GEMAmplification()
  : mEffGainGEM1(0)
  , mEffGainGEM2(0)
  , mEffGainGEM3(0)
  , mEffGainGEM4(0)
  , mRandomPolya()
{}

GEMAmplification::GEMAmplification(Float_t effGainGEM1, Float_t effGainGEM2, Float_t effGainGEM3, Float_t effGainGEM4)
  : mEffGainGEM1(effGainGEM1)
  , mEffGainGEM2(effGainGEM2)
  , mEffGainGEM3(effGainGEM3)
  , mEffGainGEM4(effGainGEM4)
  , mRandomPolya()
{
  Float_t kappa = 1/(SIGMAOVERMU*SIGMAOVERMU);
  Float_t s = 1/kappa;
  
  boost::format polya("1/(TMath::Gamma(%1%)*%2%) *std::pow(x/%3%, %4%) *std::exp(-x/%5%)");
  polya % kappa % s % s % (kappa-1) % s;
  
  TF1 polyaDistribution("polya", (polya.str()).data(), 0, 10);
  // this dramatically alters the speed with which the filling is executed... without this, the distribution makes discrete steps at every int
  polyaDistribution.SetNpx(100000);
  mRandomPolya.initialize(polyaDistribution);  
}

GEMAmplification::~GEMAmplification()
{}


Int_t GEMAmplification::getStackAmplification()
{
  const Int_t nElectronsGEM1 = getSingleGEMAmplification(1, mEffGainGEM1);
  const Int_t nElectronsGEM2 = getSingleGEMAmplification(nElectronsGEM1, mEffGainGEM2);
  const Int_t nElectronsGEM3 = getSingleGEMAmplification(nElectronsGEM2, mEffGainGEM3);
  const Int_t nElectronsGEM4 = getSingleGEMAmplification(nElectronsGEM3, mEffGainGEM4);
  return nElectronsGEM4;
}

Int_t GEMAmplification::getStackAmplification(Int_t nElectrons)
{
  Int_t nElectronsAmplified = 0;
  for(int i=0; i< nElectrons; ++i) {
    nElectronsAmplified += getStackAmplification();
  }
  return nElectronsAmplified;
}