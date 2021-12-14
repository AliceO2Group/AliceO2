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

/// \file TestTrapSim.C
/// \brief Macro to check the TRAP simulation with individual ADC data files

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"

#include "TRDSimulation/TrapSimulator.h"
#include "TRDSimulation/TrapConfig.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"

#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/TriggerRecord.h"

#include <string>
#include <fstream>
#include <iostream>

#endif

void TestTrapSim(std::string fName = "mcm.dat", int mcmPos = 0, int robPos = 4)
{
  std::array<o2::trd::ArrayADC, o2::trd::constants::NADCMCM> adcArrays{};
  std::array<o2::trd::Digit, o2::trd::constants::NADCMCM> digits{};
  std::ifstream inpFile;
  inpFile.open(fName.c_str());
  if (inpFile.is_open()) {
    int counter = 0;
    int adc;
    while (inpFile >> adc) {
      if (counter % (o2::trd::constants::NADCMCM + 1) != 0) {
        // +1 for row index in input file which we skip
        int iTb = counter / (o2::trd::constants::NADCMCM + 1);
        int iChannel = (counter % (o2::trd::constants::NADCMCM + 1)) - 1;
        adcArrays[iChannel][iTb] = (adc < 0) ? 10 : adc;
      }
      counter++;
      // printf("Got %i\n", adc);
    }
  } else {
    printf("ERROR: could not open input file %s\n", fName.c_str());
    return;
  }
  printf("Input digits --------->\n");
  for (int iChannel = 0; iChannel < o2::trd::constants::NADCMCM; ++iChannel) {
    digits[iChannel].setADC(adcArrays[iChannel]);
    digits[iChannel].setChannel(iChannel);
    std::cout << digits[iChannel] << std::endl;
  }
  printf("<---------\n");

  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  std::string trapConfigName = "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549";
  auto trapConfig = ccdbmgr.get<o2::trd::TrapConfig>("TRD/TrapConfig/" + trapConfigName);
  o2::trd::TrapSimulator trapSim;
  int detNumber = 0;
  trapSim.init(trapConfig, detNumber, robPos, mcmPos);

  for (int iChannel = 0; iChannel < o2::trd::constants::NADCMCM; ++iChannel) {
    trapSim.setData(iChannel, digits[iChannel].getADC(), iChannel);
  }

  trapSim.filter();
  trapSim.tracklet();
  auto trackletsOut = trapSim.getTrackletArray64();

  printf("Found %lu tracklets\n", trackletsOut.size());
}
