// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "FairLogger.h"
#include "CCDB/CcdbApi.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCBase/Constants.h"
#include <string>
#include <TFile.h>
#include <map>

#endif

using namespace o2::zdc;
using namespace std;

void CreateModuleConfig(long tmin = 0, long tmax = -1,
                        std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{

  ModuleConfig conf;

  int modID;

  //-------------------------------------------
  // Up to 8 modules with four channels
  // setChannel(int slot, int8_t chID, int16_t lID, bool read, bool trig = false, int tF = 0, int tL = 0, int tS = 0, int tT = 0)
  // module id must be in the range 0-7
  // channel id must be in range 0-3
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
    module.setChannel(2, IdZPC1, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZPC2, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  //-------------------------------------------
  {
    modID = 7;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPCC, 2 * modID, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZPCSum, 2 * modID, true, false, -5, 6, 4, 12);
    module.setChannel(2, IdZPC3, 2 * modID + 1, true, false, -5, 6, 4, 12);
    module.setChannel(3, IdZPC4, 2 * modID + 1, true, false, -5, 6, 4, 12);
    //
  }
  conf.check();
  conf.print();

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathConfigModule, metadata, tmin, tmax);

  // return conf;
}
