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

/// \file testSignalInduction.cxx
/// \brief This task tests the SignalInduction module of the TPC digitization
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC SignalInduction
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/SignalInduction.h"
#include "TPCBase/Mapper.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "FairLogger.h"

namespace o2
{
namespace tpc
{

/// \brief Low level test of struct
BOOST_AUTO_TEST_CASE(SignalInduction_structPRF)
{
  PadResponse prf(DigitPos(1, {2, 3}), 4);
  BOOST_CHECK(int(prf.digiPos.getCRU()) == 1);
  BOOST_CHECK(int(prf.digiPos.getPadPos().getRow()) == 2);
  BOOST_CHECK(int(prf.digiPos.getPadPos().getPad()) == 3);
  BOOST_CHECK_CLOSE(prf.padResp, 4, 1E-5);
}

bool importPRF(std::string file, const GEMstack gemstack)
{
  SignalInduction& signalInduction = SignalInduction::instance();

  std::string inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/TPC/files/";

  float x, y, normalizedPadResponse;
  int i = 0;
  std::ifstream prfFile(inputDir + file, std::ifstream::in);
  if (!prfFile) {
    BOOST_CHECK(0);
    return false;
  }

  for (std::string line; std::getline(prfFile, line);) {
    std::istringstream is(line);
    while (is >> y >> x >> normalizedPadResponse) {
      BOOST_CHECK_CLOSE(normalizedPadResponse, signalInduction.computePadResponse(x, y, gemstack), 1E-5);
    }
  }
  return true;
}

/// \brief Test of the proper import and interpolation of the PRF
BOOST_AUTO_TEST_CASE(SignalInduction_importPRF)
{
  importPRF("PRF_sampa_IROC_all.dat", GEMstack::IROCgem);
  importPRF("PRF_sampa_OROC1-2_all.dat", GEMstack::OROC1gem);
  importPRF("PRF_sampa_OROC1-2_all.dat", GEMstack::OROC2gem);
  importPRF("PRF_sampa_OROC3_all.dat", GEMstack::OROC3gem);
}

} // namespace tpc
} // namespace o2
