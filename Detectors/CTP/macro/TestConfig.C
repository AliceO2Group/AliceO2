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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <CCDB/BasicCCDBManager.h>
#include <DataFormatsCTP/Configuration.h>
#endif
using namespace o2::ctp;

void TestConfig(bool test = 0)
{
  if (test == 0) {
    return;
  }
  uint64_t timestamp = 1660196771632;
  std::string run = "523148";
  o2::ctp::CTPRunManager::setCCDBHost("https://alice-ccdb.cern.ch");
  auto ctpcfg = o2::ctp::CTPRunManager::getConfigFromCCDB(timestamp, run);
  CTPConfiguration ctpconfig;
  ctpconfig.loadConfigurationRun3(ctpcfg.getConfigString());
  // ctpconfig.printStream(std::cout);
  // return;
  // ctpconfig.assignDescriptors();
  // ctpconfig.createInputsInDecriptorsFromNames();
  ctpconfig.printStream(std::cout);
  auto& triggerclasses = ctpconfig.getCTPClasses();
  std::cout << "Found " << triggerclasses.size() << " trigger classes" << std::endl;
  int indexInList = 0;
  for (const auto& trgclass : triggerclasses) {
    uint64_t inputmask = 0;
    // auto desc = ctpconfig.getDescriptor(trgclass.descriptorIndex);
    // std::cout << "desc index:" << trgclass.descriptorIndex << " " << desc << std::endl;
    if (trgclass.descriptor != nullptr) {
      inputmask = trgclass.descriptor->getInputsMask();
      std::cout << "inputmask:" << std::hex << inputmask << std::dec << std::endl;
    }
    trgclass.printStream(std::cout);
    std::cout << indexInList << ": " << trgclass.name << ", input mask 0x" << std::hex << inputmask << ", class mask 0x" << trgclass.classMask << std::dec << std::endl;
    indexInList++;
    if (trgclass.cluster->getClusterDetNames().find("EMC") != std::string::npos) {
      std::cout << "Found EMCAL trigger cluster, class mask: 0x" << std::hex << trgclass.classMask << std::dec << std::endl;
    }
  }
  auto classmask = ctpconfig.getClassMaskForInputMask(0x4);
  std::cout << "Found class mask " << classmask << std::endl;
}
