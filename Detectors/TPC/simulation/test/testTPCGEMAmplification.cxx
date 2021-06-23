// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testGEMAmplification.cxx
/// \brief This task tests the GEMAmplification module of the TPC digitization
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC GEMAmplification
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/GEMAmplification.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/CDBInterface.h"

#include "TH1D.h"
#include "TF1.h"

namespace o2
{
namespace tpc
{

/// \brief Test of the full GEM amplification
/// The full GEM amplification process is simulated and
/// the correct gain and energy resolution is tested
BOOST_AUTO_TEST_CASE(GEMamplification_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  auto& gemParam = ParameterGEM::Instance();
  static GEMAmplification& gemStack = GEMAmplification::instance();
  TH1D hTest("hTest", "", 10000, 0, 1000000);
  TF1 gaus("gaus", "gaus");

  const int nEleIn = 158; /// Number of electrons liberated in Ne-CO2-N2 by an incident Fe-55 photon

  for (int i = 0; i < 100000; ++i) {
    hTest.Fill(gemStack.getStackAmplification(nEleIn));
  }

  hTest.Fit("gaus", "Q0");
  float energyResolution = gaus.GetParameter(2) / gaus.GetParameter(1) * 100.f;

  /// Check the resulting gain
  /// \todo should be more restrictive
  BOOST_CHECK_CLOSE(gaus.GetParameter(1) / static_cast<float>(nEleIn), (gemParam.getEffectiveGain(0) * gemParam.getEffectiveGain(1) * gemParam.getEffectiveGain(2) * gemParam.getEffectiveGain(3)), 20.f);
  /// Check the resulting energy resolution
  /// we allow for 5% variation which is given by the uncertainty of the experimental determination of the energy resolution (12.1 +/- 0.5) %
  BOOST_CHECK_CLOSE(energyResolution, 12.1, 5);
}
/// \brief Test of the full GEM amplification
/// The full GEM amplification process is simulated and
/// the correct gain and energy resolution is tested
BOOST_AUTO_TEST_CASE(GEMamplification_effective_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  auto& gemParam = ParameterGEM::Instance();
  static GEMAmplification& gemStack = GEMAmplification::instance();
  TH1D hTest("hTest", "", 100000, 0, 1000000);
  TF1 gaus("gaus", "gaus");

  const int nEleIn = 158; /// Number of electrons liberated in Ne-CO2-N2 by an incident Fe-55 photon

  for (int i = 0; i < 100000; ++i) {
    hTest.Fill(gemStack.getEffectiveStackAmplification(nEleIn));
  }

  hTest.Fit("gaus", "Q0");
  float energyResolution = gaus.GetParameter(2) / gaus.GetParameter(1) * 100.f;

  /// Check the resulting gain
  /// \todo should be more restrictive
  BOOST_CHECK_CLOSE(gaus.GetParameter(1) / static_cast<float>(nEleIn), (gemParam.TotalGainStack), 5.f);
  /// Check the resulting energy resolution
  BOOST_CHECK_CLOSE(energyResolution, 12.4f, 0.5f);
}

/// \brief Test of the getSingleGEMAmplification function
/// We filter 1000 electrons through a single GEM and compare to the outcome
BOOST_AUTO_TEST_CASE(GEMamplification_singleGEM_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  auto& gemParam = ParameterGEM::Instance();
  static GEMAmplification& gemStack = GEMAmplification::instance();
  TH1D hTest("hTest", "", 10000, 0, 10000);
  TF1 gaus("gaus", "gaus");

  for (int i = 0; i < 100000; ++i) {
    hTest.Fill(gemStack.getSingleGEMAmplification(1000, 0));
  }

  hTest.Fit("gaus", "Q0");

  /// check the resulting gain
  const float multiplication = gemParam.getEffectiveGain(0);
  BOOST_CHECK_CLOSE(gaus.GetParameter(1), multiplication * 1000.f, 0.1);
}

/// \brief Test of the getGEMMultiplication function
/// Different numbers of electrons are filtered through the loss function
/// which follows a binomial distribution
/// The outcome is compared to the expected value
BOOST_AUTO_TEST_CASE(GEMamplification_singleGEMmultiplication_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  auto& gemParam = ParameterGEM::Instance();
  auto& gasParam = ParameterGas::Instance();
  static GEMAmplification& gemStack = GEMAmplification::instance();
  TH1D hTest("hTest", "", 10000, 0, 10000);
  TH1D hTest2("hTest2", "", 10000, 0, 10000);
  TF1 gaus("gaus", "gaus");

  for (int i = 0; i < 100000; ++i) {
    hTest.Fill(gemStack.getGEMMultiplication(1000, 1));
    hTest2.Fill(gemStack.getGEMMultiplication(100, 2));
  }

  hTest.Fit("gaus", "Q0");

  /// All different cases are tested
  /// -# case nElectrons < 1
  BOOST_CHECK(gemStack.getGEMMultiplication(0, 1) == 0);
  /// -# case nElectrons > 500 - Gaussian
  BOOST_CHECK_CLOSE(gaus.GetParameter(1), gemParam.AbsoluteGain[1] * 1000.f, 0.1);
  /// As a gaussian is used the mean is tested as well, but with reduced precision
  BOOST_CHECK_CLOSE(gaus.GetParameter(2), std::sqrt(1000.f) * gasParam.SigmaOverMu * gemParam.AbsoluteGain[1], 2.5);
  /// -# case the probability is explicitly handled for each electron - the mean is a bad estimator, therefore larger tolerance
  BOOST_CHECK_CLOSE(hTest2.GetMean(), gemParam.AbsoluteGain[2] * 100.f, 5);
}

/// \brief Test of the getElectronLosses function
/// Different numbers of electrons are filtered through the loss function
/// which follows a binomial distribution
/// The outcome is compared to the expected value
BOOST_AUTO_TEST_CASE(GEMamplification_losses_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  static GEMAmplification& gemStack = GEMAmplification::instance();
  TH1D hTest("hTest", "", 100, 0, 100);
  TH1D hTest2("hTest2", "", 10, 0, 10);
  TF1 gaus("gaus", "gaus");

  for (int i = 0; i < 100000; ++i) {
    hTest.Fill(gemStack.getElectronLosses(100, 0.6));
    hTest2.Fill(gemStack.getElectronLosses(10, 0.2));
  }

  hTest.Fit("gaus", "Q0");

  /// All different cases are tested
  /// -# case nElectrons < 1 || probability < 0.00001
  BOOST_CHECK(gemStack.getElectronLosses(1, 0.000001) == 0);
  /// -# case probability > 0.99999
  BOOST_CHECK(gemStack.getElectronLosses(100, 1) == 100);
  /// -# case binomial distribution can be approximated by gaussian
  BOOST_CHECK_CLOSE(gaus.GetParameter(1), 60, 1.5);
  /// As a gaussian is used the mean is tested as well
  BOOST_CHECK_CLOSE(gaus.GetParameter(2), std::sqrt(100.f * 0.6 * (1 - 0.6)), 1.5);
  /// -# case the probability is explicitly handled for each electron
  BOOST_CHECK_CLOSE(hTest2.GetMean(), 2, 0.5);
}
} // namespace tpc
} // namespace o2
