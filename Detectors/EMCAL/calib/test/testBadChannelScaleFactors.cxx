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

#define BOOST_TEST_MODULE Test_EMCAL_Calib
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/EMCALChannelScaleFactors.h"
#include "EMCALBase/Geometry.h"

#include <algorithm>

namespace o2
{

namespace emcal
{

BOOST_AUTO_TEST_CASE(BadChannelScaleFactor_test)
{

  auto geo = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);

  struct BadChannelScaleFactorTest {
    BadChannelScaleFactorTest(int id, float e, float scale) : cellID(id), energy(e), scaleFactor(scale) {}
    int cellID;
    float energy;
    float scaleFactor;
  };

  std::vector<float> vecEnergyIntervals = {0., 1., 5., 10., 10000};
  std::vector<BadChannelScaleFactorTest> vecScaleFactors;
  o2::emcal::EMCALChannelScaleFactors scaleFactors;

  for (int iCell = 0; iCell < geo->GetNCells(); iCell++) {
    float energy = 100 * ((double)rand() / (RAND_MAX));
    float scaleFactor = 1.0 + 0.1 * ((double)rand() / (RAND_MAX));
    vecScaleFactors.push_back(BadChannelScaleFactorTest(iCell, energy, scaleFactor));

    // find index of energy interval
    auto it = std::find_if(vecEnergyIntervals.begin(), vecEnergyIntervals.end(), [&](float energyInterval) {
      return energy < energyInterval;
    });
    // cout << "iCell: " << iCell << " energy: " << energy << " scaleFactor: " << scaleFactor << " energyInterval: " << *it << endl;

    scaleFactors.insertVal(iCell, vecEnergyIntervals[*it], vecEnergyIntervals[*it + 1], scaleFactor);
  }

  for (auto& scaleFactor : vecScaleFactors) {
    float scaleFactorFromMap = scaleFactors.getScaleVal(scaleFactor.cellID, scaleFactor.energy);
    BOOST_CHECK_CLOSE(scaleFactor.scaleFactor, scaleFactorFromMap, 0.0001);
  }
}
} // namespace emcal

} // namespace o2