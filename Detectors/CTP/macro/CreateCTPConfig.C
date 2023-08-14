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

/// \file CreateCTPConfig.C
/// \brief create CTP config, test it and add to database
/// \author Roman Lietava

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <fairlogger/Logger.h>
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Configuration.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
CTPConfiguration CreateCTPConfig(std::string cfgRun3str = "", int writeToFile = 0)
{
  /// Demo configuration
  CTPConfiguration ctpcfg;
  //
  // run3 config
  //
  if (cfgRun3str.find(".rcfg") == std::string::npos) {
    cfgRun3str =
      "run 123 \n \
bcm TOF 100 1288 2476 \n \
bcm bcmPHYS 1226 \n\
bcd10 1khz \n\
bcd20 0 \n\
bcd2m 45khz \n\
#  \n\
LTG tof  \n\
trig  \n\
bcm TOF e \n\
#   \n\
LTG mft \n\
ferst 1 \n\
# \n\
LTG mch \n\
ferst 1 \n\
# 3 clusters for CRU, TRD and oldTTC detectors: \n\
0 cluster clu1 fv0 ft0 fdd its mft mid mch tpc zdc tst tof \n\
0 cl_ph bcmPHYS 3\n\
# \n\
1 cluster clu2 trd \n\
1 cl_45khz bcd2m \n\
2 cluster clu3 hmp phs \n\
2 cl_1khz bcd10 \n \
3 cluster clu4 emc cpv \n \
4 cl_5khz bcd20 \n \
5 cl3 3 5\n \
6 clMTVX MTVX\n";
  } else {
    std::string filename = cfgRun3str;
    std::ifstream in;
    in.open(filename);
    if (!in) {
      std::cout << "Can not open file:" << filename << std::endl;
      exit(1);
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    cfgRun3str = buffer.str();
  }
  //
  ctpcfg.loadConfigurationRun3(cfgRun3str);
  ctpcfg.printStream(std::cout);
  std::cout << "CTP config done" << std::endl;
  ctpcfg.checkConfigConsistency();
  if (0) {
    CTPRunManager* man = new CTPRunManager;
    man->setCCDBHost("http://ccdb-test.cern.ch:8080");
    man->saveRunConfigToCCDB(&ctpcfg, 1665784953);
    // uint64_t classmask = ctpcfg.getClassMaskForInputMask(0x4);
    // std::cout << "classmask:" << std::hex << classmask << std::dec << std::endl;
  }
  if (writeToFile == 1) {
    std::unique_ptr<TFile> myFile(TFile::Open("CTPConfig.root", "RECREATE"));
    myFile->WriteObject(&ctpcfg, "CTPConfig");
    std::cout << "File CTPConfig.root written." << std::endl;
  }
  return ctpcfg;
}
