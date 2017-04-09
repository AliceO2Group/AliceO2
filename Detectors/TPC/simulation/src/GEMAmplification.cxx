/// \file GEMAmplification.cxx
/// \brief Implementation of the GEM amplification
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/Constants.h"

using namespace o2::TPC;
using boost::format;

GEMAmplification::GEMAmplification()
  : mRandomGaus()
  , mRandomFlat()
  , mGain()
{
  float kappa = 1/(SIGMAOVERMU*SIGMAOVERMU);
  boost::format polya("1/(TMath::Gamma(%1%)*%2%) *std::pow(x/%3%, %4%) *std::exp(-x/%5%)");

  for(int i=0; i<4; ++i) {
    float s = MULTIPLICATION[i]/kappa;
    polya % kappa % s % s % (kappa-1) % s;
    /// \todo Get from root file or write own random generator
    TF1 polyaDistribution("polya", (polya.str()).data(), 0, 10.f*MULTIPLICATION[i]);
    /// this dramatically alters the speed with which the filling is executed... without this, the distribution makes discrete steps at every int
    polyaDistribution.SetNpx(100000);
    mGain[i].initialize(polyaDistribution);
  }
  mRandomGaus.initialize(RandomRing::RandomType::Gaus);
  mRandomFlat.initialize(RandomRing::RandomType::Flat);
}

GEMAmplification::~GEMAmplification()
= default;

int GEMAmplification::getStackAmplification(int nElectrons)
{
  /// We start with an arbitrary number of electrons given to the first amplification stage
  /// The amplification in the GEM stack is handled for each electron individually and the resulting amplified electrons are passed to the next amplification stage.
  const int nElectronsGEM1 = getSingleGEMAmplification(nElectrons, 1);
  const int nElectronsGEM2 = getSingleGEMAmplification(nElectronsGEM1, 2);
  const int nElectronsGEM3 = getSingleGEMAmplification(nElectronsGEM2, 3);
  const int nElectronsGEM4 = getSingleGEMAmplification(nElectronsGEM3, 4);
  return nElectronsGEM4;
}

int GEMAmplification::getSingleGEMAmplification(int nElectrons, int GEM)
{
  /// The effective gain of the GEM foil is given by three components
  /// -# Collection of the electrons, which is related to the collection efficiency ε_coll
  /// -# Amplification of the charges in the GEM holes, which is related to the GEM absolute gain G_abs
  /// -# Extraction of the electrons, which is related to the extraction efficiency ε_extr
  /// The effective gain, and thus the overall amplification of the GEM is then given by
  /// G_eff  = ε_coll * G_abs * ε_extr
  /// Each of the three processes is handled by a sub-routine
  int collectionGEM    = getElectronLosses(nElectrons, COLLECTION[GEM-1]);
  int amplificationGEM = getGEMMultiplication(collectionGEM, GEM);
  int extractionGEM    = getElectronLosses(amplificationGEM, EXTRACTION[GEM-1]);
  return extractionGEM;
}

int GEMAmplification::getGEMMultiplication(int nElectrons, int GEM)
{
  /// Total charge multiplication in the GEM
  /// We take into account fluctuations of the avalanche process
  if(nElectrons < 1) {
    /// All electrons are lost in case none are given to the GEM
    return 0;
  }
  else if(nElectrons > 500) {
   /// For this condition the central limit theorem holds and we can approximate the amplification fluctuations by a Gaussian for all electrons
   /// The mean is given by nElectrons * G_abs and the width by sqrt(nElectrons) * Sigma/Mu (Polya) * G_abs
    return ((mRandomGaus.getNextValue() * std::sqrt(static_cast<float>(nElectrons)) * SIGMAOVERMU) + nElectrons) * MULTIPLICATION[GEM-1];
  }
  else {
    /// Otherwise we compute the gain fluctuations as the convolution of many single electron amplification fluctuations
    int electronsOut = 0;
    for(int i=0; i<nElectrons; ++i) {
      electronsOut+=mGain[GEM-1].getNextValue();
    }
    return electronsOut;
  }
}

int GEMAmplification::getElectronLosses(int nElectrons, float probability)
{
  /// Electrons losses due to collection or extraction processes
  float electronsFloat = static_cast<float>(nElectrons);
  if(nElectrons < 1 || probability < 0.00001) {
    /// All electrons are lost in case none are given to the GEM, or the probability is negligible
    return 0;
  }
  else if(probability > 0.99999) {
    /// For sufficiently large probabilities all electrons are passed further on
    return nElectrons;
  }
  else if(electronsFloat * probability >= 5.f && electronsFloat * (1.f-probability) >= 5.f) {
    /// Condition whether the binomial distribution can be approximated by a Gaussian with mean n*p+0.5 and width sqrt(n*p*(1-p))
    return (mRandomGaus.getNextValue() * std::sqrt(electronsFloat*probability*(1-probability))) + electronsFloat*probability +0.5;
  }
  else {
    /// Explicit handling of the probability for each individual electron
    /// \todo For further amplification of the process one could also just draw a random number from a binomial distribution, but it should be checked whether this is faster
    int nElectronsOut = 0;
    for(int i=0; i<nElectrons; ++i) {
      if(mRandomFlat.getNextValue() < probability) {
        ++ nElectronsOut;
      }
    }
  return nElectronsOut;
  }
}
