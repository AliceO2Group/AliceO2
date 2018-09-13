// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
namespace o2
{
namespace TPC
{

/// \brief Trivial test of the default initialization of Parameter Detector
/// Precision: 1E-3 %
BOOST_AUTO_TEST_CASE(ParameterDetector_test1)
{
  const static ParameterDetector& detParam = ParameterDetector::defaultInstance();
  BOOST_CHECK_CLOSE(detParam.getPadCapacitance(), 0.1f, 1E-3);
  BOOST_CHECK_CLOSE(detParam.getTPClength(), 250.f, 1E-3);
}

/// \brief Trivial test of the initialization of Parameter Detector
/// Precision: 1E-12 %
BOOST_AUTO_TEST_CASE(ParameterDetector_test2)
{
  ParameterDetector detParam;
  detParam.setPadCapacitance(1.f);
  detParam.setTPClength(2.f);
  BOOST_CHECK_CLOSE(detParam.getPadCapacitance(), 1.f, 1E-12);
  BOOST_CHECK_CLOSE(detParam.getTPClength(), 2.f, 1E-12);
}

/// \brief Trivial test of the default initialization of Parameter Electronics
/// Precision: 1E-3 %
BOOST_AUTO_TEST_CASE(ParameterElectronics_test1)
{
  const static ParameterElectronics& eleParam = ParameterElectronics::defaultInstance();
  BOOST_CHECK(eleParam.getNShapedPoints() == 8);
  BOOST_CHECK_CLOSE(eleParam.getPeakingTime(), 160e-3, 1e-3);
  BOOST_CHECK_CLOSE(eleParam.getChipGain(), 20, 1e-3);
  BOOST_CHECK_CLOSE(eleParam.getADCDynamicRange(), 2200, 1e-3);
  BOOST_CHECK_CLOSE(eleParam.getADCSaturation(), 1024, 1e-3);
  BOOST_CHECK_CLOSE(eleParam.getZBinWidth(), 0.2f, 1e-3);
  BOOST_CHECK_CLOSE(eleParam.getElectronCharge(), 1.602e-19, 1e-3);
}

/// \brief Trivial test of the initialization of Parameter Detector
/// Precision: 1E-12 %
BOOST_AUTO_TEST_CASE(ParameterElectronics_test2)
{
  ParameterElectronics eleParam;
  eleParam.setNShapedPoints(1);
  eleParam.setPeakingTime(2.f);
  eleParam.setChipGain(3.f);
  eleParam.setADCDynamicRange(4.f);
  eleParam.setADCSaturation(5.f);
  eleParam.setZBinWidth(6.f);
  eleParam.setElectronCharge(7.f);
  BOOST_CHECK(eleParam.getNShapedPoints() == 1);
  BOOST_CHECK_CLOSE(eleParam.getPeakingTime(), 2.f, 1e-12);
  BOOST_CHECK_CLOSE(eleParam.getChipGain(), 3.f, 1e-12);
  BOOST_CHECK_CLOSE(eleParam.getADCDynamicRange(), 4.f, 1e-12);
  BOOST_CHECK_CLOSE(eleParam.getADCSaturation(), 5.f, 1e-12);
  BOOST_CHECK_CLOSE(eleParam.getZBinWidth(), 6.f, 1e-12);
  BOOST_CHECK_CLOSE(eleParam.getElectronCharge(), 7.f, 1e-12);
}

/// \brief Trivial test of the default initialization of Parameter Gas
/// Precision: 1E-3 %
BOOST_AUTO_TEST_CASE(ParameterGas_test1)
{
  const static ParameterGas& gasParam = ParameterGas::defaultInstance();
  BOOST_CHECK_CLOSE(gasParam.getWion(), 37.3e-9, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getIpot(), 20.77e-9, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getAttachmentCoefficient(), 250.f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getOxygenContent(), 5e-6, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getVdrift(), 2.58f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getSigmaOverMu(), 0.78f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getDiffT(), 0.0209f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getDiffL(), 0.0221f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getNprim(), 14.f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getScaleG4(), 0.85f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getFanoFactorG4(), 0.7f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(0), 0.76176e-1, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(1), 10.632, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(2), 0.13279e-4, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(3), 1.8631, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(4), 1.9479, 1e-3);
}

/// \brief Trivial test of the initialization of Parameter Gas
/// Precision: 1E-12 %
BOOST_AUTO_TEST_CASE(ParameterGas_test2)
{
  ParameterGas gasParam;
  gasParam.setWion(1.f);
  gasParam.setIpot(2.f);
  gasParam.setAttachmentCoefficient(3.f);
  gasParam.setOxygenContent(4.f);
  gasParam.setVdrift(5.f);
  gasParam.setSigmaOverMu(6.f);
  gasParam.setDiffT(7.f);
  gasParam.setDiffL(8.f);
  gasParam.setNprim(9.f);
  gasParam.setScaleG4(10.f);
  gasParam.setFanoFactorG4(11.f);
  gasParam.setBetheBlochParam(12.f, 13.f, 14, 15.f, 16.f);
  BOOST_CHECK_CLOSE(gasParam.getWion(), 1.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getIpot(), 2.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getAttachmentCoefficient(), 3.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getOxygenContent(), 4.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getVdrift(), 5.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getSigmaOverMu(), 6.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getDiffT(), 7.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getDiffL(), 8.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getNprim(), 9.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getScaleG4(), 10.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getFanoFactorG4(), 11.f, 1e-3);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(0), 12.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(1), 13.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(2), 14.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(3), 15.f, 1e-12);
  BOOST_CHECK_CLOSE(gasParam.getBetheBlochParam(4), 16.f, 1e-12);
}

/// \brief Trivial test of the default initialization of Parameter GEM
/// Precision: 1E-3 %
BOOST_AUTO_TEST_CASE(ParameterGEM_test1)
{
  const static ParameterGEM& gemParam = ParameterGEM::defaultInstance();
  BOOST_CHECK(gemParam.getGeometry(1) == 1);
  BOOST_CHECK(gemParam.getGeometry(2) == 3);
  BOOST_CHECK(gemParam.getGeometry(3) == 3);
  BOOST_CHECK(gemParam.getGeometry(4) == 1);
  BOOST_CHECK_CLOSE(gemParam.getPotential(1), 270.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getPotential(2), 250.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getPotential(3), 270.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getPotential(4), 340.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(1), 0.4f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(2), 4.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(3), 2.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(4), 0.1f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(5), 4.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(1), 14.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(2), 8.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(3), 53.f, 1e-3);
  BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(4), 240.f, 1e-3);
  // For fixed values
  // BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(1), 1.f, 1e-3);
  // BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(2), 0.2f, 1e-3);
  // BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(3), 0.25f, 1e-3);
  // BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(4), 1.f, 1e-3);
  // BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(1), 0.65f, 1e-3);
  // BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(2), 0.55f, 1e-3);
  // BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(3), 0.12f, 1e-3);
  // BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(4), 0.6f, 1e-3);

  for (int i = 1; i < 5; ++i) {
    BOOST_CHECK_CLOSE(
      gemParam.getEffectiveGain(i),
      gemParam.getAbsoluteGain(i) * gemParam.getCollectionEfficiency(i) * gemParam.getExtractionEfficiency(i), 1e-3);
  }
}

/// \brief Trivial test of the initialization of Parameter GEM
/// Precision: 1E-12 %
BOOST_AUTO_TEST_CASE(ParameterGEM_test2)
{
  ParameterGEM gemParam;
  gemParam.setGeometry(1, 1, 2, 3);
  gemParam.setPotential(4.f, 5.f, 6.f, 7.f);
  gemParam.setElectricField(8.f, 9.f, 10.f, 11.f, 12.f);
  gemParam.setAbsoluteGain(13.f, 14.f, 15.f, 16.f);
  gemParam.setCollectionEfficiency(17.f, 18.f, 19.f, 20.f);
  gemParam.setExtractionEfficiency(21.f, 22.f, 23.f, 24.f);
  BOOST_CHECK(gemParam.getGeometry(1) == 1);
  BOOST_CHECK(gemParam.getGeometry(2) == 1);
  BOOST_CHECK(gemParam.getGeometry(3) == 2);
  BOOST_CHECK(gemParam.getGeometry(4) == 3);
  BOOST_CHECK_CLOSE(gemParam.getPotential(1), 4.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getPotential(2), 5.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getPotential(3), 6.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getPotential(4), 7.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(1), 8.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(2), 9.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(3), 10.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(4), 11.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getElectricField(5), 12.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(1), 13.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(2), 14.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(3), 15.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(4), 16.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(1), 17.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(2), 18.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(3), 19.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(4), 20.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(1), 21.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(2), 22.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(3), 23.f, 1e-12);
  BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(4), 24.f, 1e-12);

  for (int i = 1; i < 5; ++i) {
    gemParam.setAbsoluteGain(i, i);
    BOOST_CHECK_CLOSE(gemParam.getAbsoluteGain(i), i, 1e-12);
    gemParam.setCollectionEfficiency(i, i);
    BOOST_CHECK_CLOSE(gemParam.getCollectionEfficiency(i), i, 1e-12);
    gemParam.setExtractionEfficiency(i, i);
    BOOST_CHECK_CLOSE(gemParam.getExtractionEfficiency(i), i, 1e-12);
  }
}
}
}
