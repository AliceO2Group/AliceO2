// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GEMAmplification.cxx
/// \brief Implementation of the GEM amplification
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/GEMAmplification.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterGEM.h"
#include <TStopwatch.h>
#include <iostream>
#include "MathUtils/CachingTF1.h"
#include <TFile.h>

using namespace o2::TPC;
using boost::format;

GEMAmplification::GEMAmplification()
  : mRandomGaus(),
    mRandomFlat(),
    mGain()
{
  TStopwatch watch;
  watch.Start();
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterGEM &gemParam = ParameterGEM::defaultInstance();
  const float sigmaOverMu = gasParam.getSigmaOverMu();
  const float kappa = 1/(sigmaOverMu*sigmaOverMu);
  boost::format polya("1/(TMath::Gamma(%1%)*%2%) *std::pow(x/%3%, %4%) *std::exp(-x/%5%)");

  bool cacheexists = false;
  // FIXME: this should come from the CCDB (just as the other parameters)
  auto outfile = TFile::Open(TString::Format("tpc_polyadist.root").Data());
  if(outfile) {
    cacheexists=true;
  }
  else {
    outfile = TFile::Open(TString::Format("tpc_polyadist.root").Data(), "create");
  }
    
  for(int i=0; i<4; ++i) {
    float s = gemParam.getAbsoluteGain(i+1)/kappa;
    polya % kappa % s % s % (kappa-1) % s;
      std::string name = polya.str();
      o2::Base::CachingTF1* polyaDistribution = nullptr;
      if (!cacheexists) {
        polyaDistribution = new o2::Base::CachingTF1("polya", name.c_str(), 0, 10.f*gemParam.getAbsoluteGain(i+1));
        /// this dramatically alters the speed with which the filling is executed... without this, the distribution makes discrete steps at every int
        polyaDistribution->SetNpx(100000);
      }
      else {
        polyaDistribution = (o2::Base::CachingTF1*)outfile->Get(TString::Format("func%d",i).Data());
        // FIXME: verify that distribution corresponds to the parameters used here
      }
    mGain[i].initialize(*polyaDistribution);
  
    if(!cacheexists) {
        outfile->WriteTObject(polyaDistribution, TString::Format("func%d",i).Data());
    }
    delete polyaDistribution;
  }

  if(outfile) outfile->Close();
  mRandomGaus.initialize(RandomRing::RandomType::Gaus);
  mRandomFlat.initialize(RandomRing::RandomType::Flat);
  watch.Stop();
  std::cerr << "GEM SETUP TOOK " << watch.CpuTime() << "\n";
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
  const static ParameterGEM &gemParam = ParameterGEM::defaultInstance();
  int collectionGEM    = getElectronLosses(nElectrons, gemParam.getCollectionEfficiency(GEM));
  int amplificationGEM = getGEMMultiplication(collectionGEM, GEM);
  int extractionGEM    = getElectronLosses(amplificationGEM, gemParam.getExtractionEfficiency(GEM));
  return extractionGEM;
}

int GEMAmplification::getGEMMultiplication(int nElectrons, int GEM)
{
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterGEM &gemParam = ParameterGEM::defaultInstance();

  /// Total charge multiplication in the GEM
  /// We take into account fluctuations of the avalanche process
  if(nElectrons < 1) {
    /// All electrons are lost in case none are given to the GEM
    return 0;
  }
  else if(nElectrons > 500) {
   /// For this condition the central limit theorem holds and we can approximate the amplification fluctuations by a Gaussian for all electrons
   /// The mean is given by nElectrons * G_abs and the width by sqrt(nElectrons) * Sigma/Mu (Polya) * G_abs
    return ((mRandomGaus.getNextValue() * std::sqrt(static_cast<float>(nElectrons)) * gasParam.getSigmaOverMu()) + nElectrons) * gemParam.getAbsoluteGain(GEM);
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
