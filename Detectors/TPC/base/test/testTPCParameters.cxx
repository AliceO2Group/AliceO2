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

/// \file testTPCParameters.cxx
/// \brief This task tests the Parameter handling
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC Parameters
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterGas.h"
#include <CommonUtils/ConfigurableParam.h>
#include <CommonUtils/ConfigurableParamHelper.h>
#include <boost/property_tree/ptree.hpp>

namespace o2::tpc
{

constexpr float NominalTimeBin = 8 * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;

/// \brief Trivial test of the default initialization of Parameter Detector
/// Precision: 1E-3 %
BOOST_AUTO_TEST_CASE(ParameterDetector_test1)
{
  BOOST_CHECK_CLOSE(ParameterDetector::Instance().PadCapacitance, 0.1f, 1E-3);
  BOOST_CHECK_CLOSE(ParameterDetector::Instance().TPClength, 250.f, 1E-3);
  BOOST_CHECK_CLOSE(ParameterDetector::Instance().TmaxTriggered, 550.f, 1E-12);

  BOOST_CHECK_CLOSE(ParameterDetector::Instance().PadCapacitance,
                    o2::conf::ConfigurableParam::getValueAs<float>("TPCDetParam.PadCapacitance"), 1E-3);
  BOOST_CHECK_CLOSE(ParameterDetector::Instance().TPClength,
                    o2::conf::ConfigurableParam::getValueAs<float>("TPCDetParam.TPClength"), 1E-3);
  BOOST_CHECK(
    ParameterDetector::Instance().TmaxTriggered == o2::conf::ConfigurableParam::getValueAs<TimeBin>("TPCDetParam.TmaxTriggered"));
}

/// \brief Trivial test of the initialization of Parameter Detector
/// Precision: 1E-12 %
BOOST_AUTO_TEST_CASE(ParameterDetector_test2)
{
  o2::conf::ConfigurableParam::updateFromString(
    "TPCDetParam.PadCapacitance=2;TPCDetParam.TPClength=3;TPCDetParam.TmaxTriggered=4");
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCDetParam.PadCapacitance"), 2.f, 1E-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCDetParam.TPClength"), 3.f, 1E-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCDetParam.TmaxTriggered"), 4.f, 1E-12);
  BOOST_CHECK_CLOSE(ParameterDetector::Instance().PadCapacitance, 2.f, 1E-12);
  BOOST_CHECK_CLOSE(ParameterDetector::Instance().TPClength, 3.f, 1E-12);
  BOOST_CHECK_CLOSE(ParameterDetector::Instance().TmaxTriggered, 4.f, 1E-12);
}

/// \brief Trivial test of the default initialization of Parameter Electronics
/// Precision: 1E-3 %
BOOST_AUTO_TEST_CASE(ParameterElectronics_test1)
{
  BOOST_CHECK(ParameterElectronics::Instance().NShapedPoints == 8);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().PeakingTime, 160e-3, 1e-3);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ChipGain, 20, 1e-3);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ADCdynamicRange, 2200, 1e-3);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ADCsaturation, 1024, 1e-3);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ZbinWidth, NominalTimeBin, 1e-3);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ElectronCharge, 1.602e-19, 1e-3);
  BOOST_CHECK(ParameterElectronics::Instance().DigiMode == DigitzationMode::Auto);

  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<int>("TPCEleParam.NShapedPoints") == 8);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.PeakingTime"), 160e-3, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ChipGain"), 20, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ADCdynamicRange"), 2200, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ADCsaturation"), 1024, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ZbinWidth"), NominalTimeBin, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ElectronCharge"), 1.602e-19, 1e-3);
}

/// \brief Trivial test of the initialization of Parameter Detector
/// Precision: 1E-12 %
BOOST_AUTO_TEST_CASE(ParameterElectronics_test2)
{
  o2::conf::ConfigurableParam::updateFromString(
    "TPCEleParam.NShapedPoints=1;TPCEleParam.PeakingTime=2;TPCEleParam.ChipGain=3;TPCEleParam.ADCdynamicRange=4;TPCEleParam.ADCsaturation=5;TPCEleParam.ZbinWidth=6;TPCEleParam.ElectronCharge=7;TPCEleParam.DigiMode=0");
  BOOST_CHECK(ParameterElectronics::Instance().NShapedPoints == 1);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().PeakingTime, 2.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ChipGain, 3.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ADCdynamicRange, 4.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ADCsaturation, 5.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ZbinWidth, 6.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterElectronics::Instance().ElectronCharge, 7.f, 1e-12);
  BOOST_CHECK(ParameterElectronics::Instance().DigiMode == DigitzationMode::FullMode);
  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<int>("TPCEleParam.NShapedPoints") == 1);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.PeakingTime"), 2.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ChipGain"), 3.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ADCdynamicRange"), 4.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ADCsaturation"), 5.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ZbinWidth"), 6.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCEleParam.ElectronCharge"), 7.f, 1e-12);
}

/// \brief Trivial test of the default initialization of Parameter Gas
/// Precision: 1E-3 %
BOOST_AUTO_TEST_CASE(ParameterGas_test1)
{
  BOOST_CHECK_CLOSE(ParameterGas::Instance().Wion, 37.3e-9, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().Ipot, 20.77e-9, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().AttCoeff, 250.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().OxygenCont, 5e-6, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().DriftV, 2.58f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().SigmaOverMu, 0.78f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().DiffT, 0.0209f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().DiffL, 0.0221f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().Nprim, 14.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().ScaleFactorG4, 0.85f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().FanoFactorG4, 0.7f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[0], 0.820172e-1, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[1], 9.94795f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[2], 8.97292e-05f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[3], 2.05873f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[4], 1.65272f, 1e-3);

  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Wion"), 37.3e-9, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Ipot"), 20.77e-9, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.AttCoeff"), 250.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.OxygenCont"), 5e-6, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.DriftV"), 2.58f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.SigmaOverMu"), 0.78f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.DiffT"), 0.0209f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.DiffL"), 0.0221f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Nprim"), 14.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.ScaleFactorG4"), 0.85f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.FanoFactorG4"), 0.7f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[0]"), 0.820172e-1, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[1]"), 9.94795f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[2]"), 8.97292e-05f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[3]"), 2.05873f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[4]"), 1.65272f, 1e-3);
}

/// \brief Trivial test of the initialization of Parameter Gas
/// Precision: 1E-12 %
BOOST_AUTO_TEST_CASE(ParameterGas_test2)
{
  o2::conf::ConfigurableParam::updateFromString(
    "TPCGasParam.Wion=1;TPCGasParam.Ipot=2;TPCGasParam.AttCoeff=3;TPCGasParam.OxygenCont=4;TPCGasParam.DriftV=5;TPCGasParam.SigmaOverMu=6;"
    "TPCGasParam.DiffT=7;TPCGasParam.DiffL=8;"
    "TPCGasParam.Nprim=9;TPCGasParam.ScaleFactorG4=10;TPCGasParam.FanoFactorG4=11;TPCGasParam.BetheBlochParam[0]=12;TPCGasParam.BetheBlochParam[1]=13;TPCGasParam.BetheBlochParam[2]=14;"
    "TPCGasParam.BetheBlochParam[3]=15;TPCGasParam.BetheBlochParam[4]=16");

  BOOST_CHECK_CLOSE(ParameterGas::Instance().Wion, 1.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().Ipot, 2.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().AttCoeff, 3.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().OxygenCont, 4.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().DriftV, 5.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().SigmaOverMu, 6.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().DiffT, 7.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().DiffL, 8.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().Nprim, 9.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().ScaleFactorG4, 10.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().FanoFactorG4, 11.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[0], 12.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[1], 13.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[2], 14.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[3], 15.f, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().BetheBlochParam[4], 16.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Wion"), 1.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Ipot"), 2.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.AttCoeff"), 3.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.OxygenCont"), 4.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.DriftV"), 5.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.SigmaOverMu"), 6.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.DiffT"), 7.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.DiffL"), 8.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Nprim"), 9.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.ScaleFactorG4"), 10.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.FanoFactorG4"), 11.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[0]"), 12.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[1]"), 13.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[2]"), 14.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[3]"), 15.f, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.BetheBlochParam[4]"), 16.f, 1e-12);
}

/// \brief Trivial test of the two ways to setValue for a ConfigurableParameter
/// Precision: 1E-3 %
BOOST_AUTO_TEST_CASE(setValues_test1)
{
  BOOST_CHECK_CLOSE(ParameterGas::Instance().Wion, o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Wion"), 1e-3);
  o2::conf::ConfigurableParam::setValue<float>("TPCGasParam", "Wion", 3.0);
  BOOST_CHECK_CLOSE(ParameterGas::Instance().Wion, 3.0, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Wion"), 3.0, 1e-3);
  o2::conf::ConfigurableParam::setValue("TPCGasParam.Wion", "5.0");
  BOOST_CHECK_CLOSE(ParameterGas::Instance().Wion, 5.0, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.Wion"), 5.0, 1e-3);
}

/// \brief Trivial test of the default initialization of Parameter GEM
/// Precision: 1E-3 %
///
BOOST_AUTO_TEST_CASE(ParameterGEM_test1)
{
  BOOST_CHECK(ParameterGEM::Instance().Geometry[0] == 0);
  BOOST_CHECK(ParameterGEM::Instance().Geometry[1] == 2);
  BOOST_CHECK(ParameterGEM::Instance().Geometry[2] == 2);
  BOOST_CHECK(ParameterGEM::Instance().Geometry[3] == 0);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[0], 4.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[1], 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[2], 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[3], 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[4], 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Potential[0], 270.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Potential[1], 250.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Potential[2], 270.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Potential[3], 340.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[0], 0.4f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[1], 4.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[2], 2.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[3], 0.1f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[4], 4.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().AbsoluteGain[0], 14.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().AbsoluteGain[1], 8.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().AbsoluteGain[2], 53.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().AbsoluteGain[3], 240.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().TotalGainStack, 2000.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().KappaStack, 1.205f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().EfficiencyStack, 0.528f, 1e-3);
  BOOST_CHECK(ParameterGEM::Instance().AmplMode == AmplificationMode::EffectiveMode);

  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<int>("TPCGEMParam.Geometry[0]") == 0);
  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<int>("TPCGEMParam.Geometry[1]") == 2);
  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<int>("TPCGEMParam.Geometry[2]") == 2);
  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<int>("TPCGEMParam.Geometry[3]") == 0);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[0]"), 4.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[1]"), 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[2]"), 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[3]"), 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[4]"), 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Potential[0]"), 270.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Potential[1]"), 250.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Potential[2]"), 270.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Potential[3]"), 340.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[0]"), 0.4f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[1]"), 4.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[2]"), 2.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[3]"), 0.1f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[4]"), 4.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.AbsoluteGain[0]"), 14.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.AbsoluteGain[1]"), 8.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.AbsoluteGain[2]"), 53.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.AbsoluteGain[3]"), 240.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.TotalGainStack"), 2000.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.KappaStack"), 1.205f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.EfficiencyStack"), 0.528f, 1e-3);

  // For fixed values
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().CollectionEfficiency[0], 1.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().CollectionEfficiency[1], 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().CollectionEfficiency[2], 0.25f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().CollectionEfficiency[3], 1.f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ExtractionEfficiency[0], 0.65f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ExtractionEfficiency[1], 0.55f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ExtractionEfficiency[2], 0.12f, 1e-3);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ExtractionEfficiency[3], 0.6f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.CollectionEfficiency[0]"), 1.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.CollectionEfficiency[1]"), 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.CollectionEfficiency[2]"), 0.25f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.CollectionEfficiency[3]"), 1.f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ExtractionEfficiency[0]"), 0.65f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ExtractionEfficiency[1]"), 0.55f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ExtractionEfficiency[2]"), 0.12f, 1e-3);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ExtractionEfficiency[3]"), 0.6f, 1e-3);

  for (int i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(ParameterGEM::Instance().getEffectiveGain(i),
                      ParameterGEM::Instance().AbsoluteGain[i] * ParameterGEM::Instance().CollectionEfficiency[i] * ParameterGEM::Instance().ExtractionEfficiency[i], 1e-3);
  }
}

/// \brief Trivial test of the initialization of Parameter GEM
/// Precision: 1E-12 %
BOOST_AUTO_TEST_CASE(ParameterGEM_test2)
{

  o2::conf::ConfigurableParam::updateFromString(
    "TPCGEMParam.Geometry[0]=1;TPCGEMParam.Geometry[1]=2;TPCGEMParam.Geometry[2]=3;TPCGEMParam.Geometry[3]=4;"
    "TPCGEMParam.Distance[0]=5;TPCGEMParam.Distance[1]=6;TPCGEMParam.Distance[2]=7;TPCGEMParam.Distance[3]=8;TPCGEMParam.Distance[4]=9;"
    "TPCGEMParam.Potential[0]=10;TPCGEMParam.Potential[1]=11;TPCGEMParam.Potential[2]=12;TPCGEMParam.Potential[3]=13;"
    "TPCGEMParam.ElectricField[0]=14;TPCGEMParam.ElectricField[1]=15;TPCGEMParam.ElectricField[2]=16;TPCGEMParam.ElectricField[3]=17;TPCGEMParam.ElectricField[4]=18;"
    "TPCGEMParam.AbsoluteGain[0]=19;TPCGEMParam.AbsoluteGain[1]=20;TPCGEMParam.AbsoluteGain[2]=21;TPCGEMParam.AbsoluteGain[3]=22;"
    "TPCGEMParam.CollectionEfficiency[0]=23;TPCGEMParam.CollectionEfficiency[1]=24;TPCGEMParam.CollectionEfficiency[2]=25;TPCGEMParam.CollectionEfficiency[3]=26;"
    "TPCGEMParam.ExtractionEfficiency[0]=27;TPCGEMParam.ExtractionEfficiency[1]=28;TPCGEMParam.ExtractionEfficiency[2]=29;TPCGEMParam.ExtractionEfficiency[3]=30;"
    "TPCGEMParam.TotalGainStack=31;TPCGEMParam.KappaStack=32;TPCGEMParam.EfficiencyStack=33;TPCGEMParam.AmplMode=0;"
    "");

  BOOST_CHECK(ParameterGEM::Instance().Geometry[0] == 1);
  BOOST_CHECK(ParameterGEM::Instance().Geometry[1] == 2);
  BOOST_CHECK(ParameterGEM::Instance().Geometry[2] == 3);
  BOOST_CHECK(ParameterGEM::Instance().Geometry[3] == 4);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[0], 5, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[1], 6, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[2], 7, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[3], 8, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Distance[4], 9, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Potential[0], 10, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Potential[1], 11, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Potential[2], 12, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().Potential[3], 13, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[0], 14, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[1], 15, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[2], 16, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[3], 17, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ElectricField[4], 18, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().AbsoluteGain[0], 19, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().AbsoluteGain[1], 20, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().AbsoluteGain[2], 21, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().AbsoluteGain[3], 22, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().CollectionEfficiency[0], 23, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().CollectionEfficiency[1], 24, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().CollectionEfficiency[2], 25, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().CollectionEfficiency[3], 26, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ExtractionEfficiency[0], 27, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ExtractionEfficiency[1], 28, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ExtractionEfficiency[2], 29, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().ExtractionEfficiency[3], 30, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().TotalGainStack, 31, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().KappaStack, 32, 1e-12);
  BOOST_CHECK_CLOSE(ParameterGEM::Instance().EfficiencyStack, 33, 1e-3);
  BOOST_CHECK(ParameterGEM::Instance().AmplMode == AmplificationMode::FullMode);
  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Geometry[0]") == 1);
  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Geometry[1]") == 2);
  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Geometry[2]") == 3);
  BOOST_CHECK(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Geometry[3]") == 4);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[0]"), 5, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[1]"), 6, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[2]"), 7, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[3]"), 8, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Distance[4]"), 9, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Potential[0]"), 10, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Potential[1]"), 11, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Potential[2]"), 12, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.Potential[3]"), 13, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[0]"), 14, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[1]"), 15, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[2]"), 16, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[3]"), 17, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[4]"), 18, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.AbsoluteGain[0]"), 19, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.AbsoluteGain[1]"), 20, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.AbsoluteGain[2]"), 21, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.AbsoluteGain[3]"), 22, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.CollectionEfficiency[0]"), 23, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.CollectionEfficiency[1]"), 24, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.CollectionEfficiency[2]"), 25, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.CollectionEfficiency[3]"), 26, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ExtractionEfficiency[0]"), 27, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ExtractionEfficiency[1]"), 28, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ExtractionEfficiency[2]"), 29, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ExtractionEfficiency[3]"), 30, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.TotalGainStack"), 31, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.KappaStack"), 32, 1e-12);
  BOOST_CHECK_CLOSE(o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.EfficiencyStack"), 33, 1e-3);
}
} // namespace o2
