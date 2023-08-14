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

#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCBase/Constants.h"
#include <string>
#include <TFile.h>
#include <map>

#endif

#include "ZDCBase/Helpers.h"
using namespace o2::zdc;
using namespace std;

void CreateModuleConfig(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{
  // Shortcuts: internal, external, test, local, root

  ModuleConfig conf;

  // Conversion factor for baseline
  conf.nBunchAverage = 2; // Number of bunch crossings in average
  int bshift = std::ceil(std::log2(double(o2::zdc::NTimeBinsPerBC) * double(conf.nBunchAverage) * double(ADCRange))) - 16;
  int divisor = 0x1 << bshift;
  conf.baselineFactor = float(divisor) / float(conf.nBunchAverage) / float(o2::zdc::NTimeBinsPerBC);

  // Bunch list for baseline calculation e.g.:
  // conf.resetMap();
  // conf.addBunch(3563);

  int modID;

  //-------------------------------------------
  // Up to 8 modules with four channels
  // setChannel(int slot, int8_t chID, int16_t fID, bool read, bool trig = false, int tF = 0, int tL = 0, int tS = 0, int tT = 0)
  // module id must be in the range 0-7
  // channel id must be in range 0-3
  // frontend id must be in range 0-15 and identify the pair of channels connected to
  // each fibre
  {
    modID = 0;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZNAC, 2 * modID, true, true, -5, 6, 4, 12);
    module.setChannel(1, IdZNASum, 2 * modID, false, false, -5, 6, 4, 12);
    module.setChannel(2, IdZNA1, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZNA2, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  //-------------------------------------------
  {
    modID = 1;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZNAC, 2 * modID, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZNASum, 2 * modID, true, false, -5, 6, 4, 12);
    module.setChannel(2, IdZNA3, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZNA4, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  //-------------------------------------------
  {
    modID = 2;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZNCC, 2 * modID, true, true, -5, 6, 4, 12);
    module.setChannel(1, IdZNCSum, 2 * modID, false, false, -5, 6, 4, 12);
    module.setChannel(2, IdZNC1, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZNC2, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  //-------------------------------------------
  {
    modID = 3;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZNCC, 2 * modID, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZNCSum, 2 * modID, true, false, -5, 6, 4, 12);
    module.setChannel(2, IdZNC3, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZNC4, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  //-------------------------------------------
  {
    modID = 4;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPAC, 2 * modID, true, true, -5, 6, 4, 12);
    module.setChannel(1, IdZEM1, 2 * modID, true, true, -5, 6, 4, 12);
    module.setChannel(2, IdZPA1, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZPA2, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  //-------------------------------------------
  {
    modID = 5;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPAC, 2 * modID, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZPASum, 2 * modID, true, false, -5, 6, 4, 12);
    module.setChannel(2, IdZPA3, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZPA4, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  //-------------------------------------------
  {
    modID = 6;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPCC, 2 * modID, true, true, -5, 6, 4, 12);
    module.setChannel(1, IdZEM2, 2 * modID, true, true, -5, 6, 4, 12);
    module.setChannel(2, IdZPC3, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZPC4, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  //-------------------------------------------
  {
    modID = 7;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPCC, 2 * modID, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZPCSum, 2 * modID, true, false, -5, 6, 4, 12);
    module.setChannel(2, IdZPC1, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZPC2, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  conf.check();
  conf.print();

  std::string ccdb_host = ccdbShortcuts(ccdbHost, conf.Class_Name(), CCDBPathConfigModule);

  if (endsWith(ccdb_host, ".root")) {
    TFile f(TString::Format(ccdb_host.data(), tmin, tmax), "recreate");
    f.WriteObjectAny(&conf, conf.Class_Name(), "ccdb_object");
    f.Close();
    return;
  }

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdb_host.c_str());
  LOG(info) << "CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathConfigModule, metadata, tmin, tmax);
}
