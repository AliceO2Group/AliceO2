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
#include <TStopwatch.h>
#include "MathUtils/CachingTF1.h"
#include <TFile.h>
#include "TPCBase/CDBInterface.h"
#include <fstream>
#include "FairLogger.h"

using namespace o2::tpc;
using namespace o2::math_utils;
using boost::format;

GEMAmplification::GEMAmplification()
  : mRandomGaus(),
    mRandomFlat(RandomRing<>::RandomType::Flat),
    mGain()
{
  updateParameters();

  TStopwatch watch;
  watch.Start();
  const float sigmaOverMu = mGasParam->SigmaOverMu;
  const float kappa = 1 / (sigmaOverMu * sigmaOverMu);
  boost::format polya("1/(TMath::Gamma(%1%)*%2%) * TMath::Power(x/%3%, %4%) * TMath::Exp(-x/%5%)");

  // to be replaced by std::filesystem::exists once we have C++17
  auto fileexists = [](const char* fileName) -> bool {
    std::ifstream infile(fileName);
    return infile.good();
  };

  const char* polyaFileName = "tpc_polya.root";
  TFile* outfile;
  auto cacheexists = fileexists(polyaFileName);
  if (cacheexists) {
    LOG(INFO) << "TPC: GEM setup from existing cache";
    outfile = TFile::Open(polyaFileName);
  } else {
    LOG(INFO) << "TPC: GEM setup - initialization from scratch";
    outfile = TFile::Open(polyaFileName, "CREATE");
  }

  /// Set the polya distribution for the individual GEMs
  for (int i = 0; i < 4; ++i) {
    float s = mGEMParam->AbsoluteGain[i] / kappa;
    polya % kappa % s % s % (kappa - 1) % s;
    std::string name = polya.str();
    o2::base::CachingTF1* polyaDistribution = nullptr;
    if (!cacheexists) {
      polyaDistribution = new o2::base::CachingTF1("polya", name.c_str(), 0, 10.f * mGEMParam->AbsoluteGain[i]);
      /// this dramatically alters the speed with which the filling is executed...
      /// without this, the distribution makes discrete steps at every int
      polyaDistribution->SetNpx(100000);
    } else {
      polyaDistribution = (o2::base::CachingTF1*)outfile->Get(TString::Format("func%d", i).Data());
      // FIXME: verify that distribution corresponds to the parameters used here
    }
    mGain[i].initialize(*polyaDistribution);

    if (!cacheexists) {
      outfile->WriteTObject(polyaDistribution, TString::Format("func%d", i).Data());
    }
    delete polyaDistribution;
  }

  /// Set the polya distribution for the full stack
  const float gainStack = mGEMParam->TotalGainStack;
  const float kappaStack = mGEMParam->KappaStack;
  const float effStack = mGEMParam->EfficiencyStack;
  const float sStack = gainStack / (kappaStack * (1.f - effStack));
  polya % kappaStack % sStack % sStack % (kappaStack - 1) % sStack;
  std::string name = polya.str();
  o2::base::CachingTF1* polyaDistribution = nullptr;
  if (!cacheexists) {
    polyaDistribution = new o2::base::CachingTF1("polya", name.c_str(), 0, 50000);
    polyaDistribution->SetNpx(50000);
  } else {
    polyaDistribution = (o2::base::CachingTF1*)outfile->Get("polyaStack");
  }
  mGainFullStack.initialize(*polyaDistribution);

  if (!cacheexists) {
    outfile->WriteTObject(polyaDistribution, "polyaStack");
  }
  delete polyaDistribution;

  if (outfile)
    outfile->Close();
  watch.Stop();
  LOG(INFO) << "TPC: GEM setup (polya) took " << watch.CpuTime();
}

GEMAmplification::~GEMAmplification() = default;

void GEMAmplification::updateParameters()
{
  auto& cdb = CDBInterface::instance();
  mGEMParam = &(ParameterGEM::Instance());
  mGasParam = &(ParameterGas::Instance());
  mGainMap = &(cdb.getGainMap());
}

int GEMAmplification::getStackAmplification(int nElectrons)
{
  /// We start with an arbitrary number of electrons given to the first amplification stage
  /// The amplification in the GEM stack is handled for each electron individually and the resulting amplified
  /// electrons are passed to the next amplification stage.
  const int nElectronsGEM1 = getSingleGEMAmplification(nElectrons, 0);
  const int nElectronsGEM2 = getSingleGEMAmplification(nElectronsGEM1, 1);
  const int nElectronsGEM3 = getSingleGEMAmplification(nElectronsGEM2, 2);
  const int nElectronsGEM4 = getSingleGEMAmplification(nElectronsGEM3, 3);
  return nElectronsGEM4;
}

int GEMAmplification::getEffectiveStackAmplification(int nElectrons)
{
  /// We start with an arbitrary number of electrons given to the first amplification stage
  /// The amplification in the GEM stack is handled for each electron individually and the amplification
  /// in the stack is handled in an effective manner
  int nElectronsGEM = 0;
  for (int i = 0; i < nElectrons; ++i) {
    if (mRandomFlat.getNextValue() < mGEMParam->EfficiencyStack) {
      continue;
    }
    nElectronsGEM += mGainFullStack.getNextValue();
  }
  return nElectronsGEM;
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
  int collectionGEM = getElectronLosses(nElectrons, mGEMParam->CollectionEfficiency[GEM]);
  int amplificationGEM = getGEMMultiplication(collectionGEM, GEM);
  int extractionGEM = getElectronLosses(amplificationGEM, mGEMParam->ExtractionEfficiency[GEM]);
  return extractionGEM;
}

int GEMAmplification::getGEMMultiplication(int nElectrons, int GEM)
{
  /// Total charge multiplication in the GEM
  /// We take into account fluctuations of the avalanche process
  if (nElectrons < 1) {
    /// All electrons are lost in case none are given to the GEM
    return 0;
  } else if (nElectrons > 500) {
    /// For this condition the central limit theorem holds and we can approximate the amplification fluctuations by
    ///a Gaussian for all electrons
    /// The mean is given by nElectrons * G_abs and the width by sqrt(nElectrons) * Sigma/Mu (Polya) * G_abs
    return ((mRandomGaus.getNextValue() * std::sqrt(static_cast<float>(nElectrons)) *
             mGasParam->SigmaOverMu) +
            nElectrons) *
           mGEMParam->AbsoluteGain[GEM];
  } else {
    /// Otherwise we compute the gain fluctuations as the convolution of many single electron amplification
    /// fluctuations
    int electronsOut = 0;
    for (int i = 0; i < nElectrons; ++i) {
      electronsOut += mGain[GEM].getNextValue();
    }
    return electronsOut;
  }
}

int GEMAmplification::getElectronLosses(int nElectrons, float probability)
{
  /// Electrons losses due to collection or extraction processes
  float electronsFloat = static_cast<float>(nElectrons);
  if (nElectrons < 1 || probability < 0.00001) {
    /// All electrons are lost in case none are given to the GEM, or the probability is negligible
    return 0;
  } else if (probability > 0.99999) {
    /// For sufficiently large probabilities all electrons are passed further on
    return nElectrons;
  } else if (electronsFloat * probability >= 5.f && electronsFloat * (1.f - probability) >= 5.f) {
    /// Condition whether the binomial distribution can be approximated by a Gaussian with mean n*p+0.5 and
    /// width sqrt(n*p*(1-p))
    return (mRandomGaus.getNextValue() * std::sqrt(electronsFloat * probability * (1 - probability))) +
           electronsFloat * probability + 0.5;
  } else {
    /// Explicit handling of the probability for each individual electron
    int nElectronsOut = 0;
    for (int i = 0; i < nElectrons; ++i) {
      if (mRandomFlat.getNextValue() < probability) {
        ++nElectronsOut;
      }
    }
    return nElectronsOut;
  }
}
