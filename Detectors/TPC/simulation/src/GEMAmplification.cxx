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

GEMAmplification::GEMAmplification(float effGainGEM1, float effGainGEM2, float effGainGEM3, float effGainGEM4)
  : mEffGainGEM1(effGainGEM1)
  , mEffGainGEM2(effGainGEM2)
  , mEffGainGEM3(effGainGEM3)
  , mEffGainGEM4(effGainGEM4)
  , mRandomPolya()
{
  float kappa = 1/(SIGMAOVERMU*SIGMAOVERMU);
  float s = 1/kappa;
  
  boost::format polya("1/(TMath::Gamma(%1%)*%2%) *std::pow(x/%3%, %4%) *std::exp(-x/%5%)");
  polya % kappa % s % s % (kappa-1) % s;
  // TODO Get from root file or write own random generator
  TF1 polyaDistribution("polya", (polya.str()).data(), 0, 10);
  // this dramatically alters the speed with which the filling is executed... without this, the distribution makes discrete steps at every int
  polyaDistribution.SetNpx(100000);
  mRandomPolya.initialize(polyaDistribution);  
}

GEMAmplification::~GEMAmplification()
{}


int GEMAmplification::getStackAmplification()
{
  const int nElectronsGEM1 = getSingleGEMAmplification(1, mEffGainGEM1);
  const int nElectronsGEM2 = getSingleGEMAmplification(nElectronsGEM1, mEffGainGEM2);
  const int nElectronsGEM3 = getSingleGEMAmplification(nElectronsGEM2, mEffGainGEM3);
  const int nElectronsGEM4 = getSingleGEMAmplification(nElectronsGEM3, mEffGainGEM4);
  return nElectronsGEM4;
}

int GEMAmplification::getStackAmplification(int nElectrons)
{
  int nElectronsAmplified = 0;
  for(int i=0; i< nElectrons; ++i) {
    nElectronsAmplified += getStackAmplification();
  }
  return nElectronsAmplified;
}
