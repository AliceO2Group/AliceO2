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

#endif

using namespace o2::zdc;

void CreateModuleConfig(long tmin = 0, long tmax = -1,
                        std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{

  ModuleConfig conf;

  int modID;

  //-------------------------------------------
  {
    modID = 0;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZNAC, 0, true, true, -5, 6, 4, 12);
    module.setChannel(1, IdZNASum, 1, false, false);
    module.setChannel(2, IdZNA1, 2, true, false);
    module.setChannel(3, IdZNA2, 3, true, false);
    //
  }
  //-------------------------------------------
  {
    modID = 1;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZNAC, 4, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZNASum, 5, true, false);
    module.setChannel(2, IdZNA1, 6, true, false);
    module.setChannel(3, IdZNA2, 7, true, false);
    //
  }
  //-------------------------------------------
  {
    modID = 2;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZNCC, 8, true, true, -5, 6, 4, 12);
    module.setChannel(1, IdZNCSum, 9, false, false);
    module.setChannel(2, IdZNC1, 10, true, false);
    module.setChannel(3, IdZNC2, 11, true, false);
    //
  }
  //-------------------------------------------
  {
    modID = 3;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZNCC, 12, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZNCSum, 13, true, false);
    module.setChannel(2, IdZNC3, 14, true, false);
    module.setChannel(3, IdZNC4, 15, true, false);
    //
  }
  //-------------------------------------------
  {
    modID = 4;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPAC, 16, true, true, -5, 6, 4, 12);
    module.setChannel(1, IdZEM1, 16, true, true, -5, 6, 4, 12);
    module.setChannel(2, IdZPA1, 17, true, false);
    module.setChannel(3, IdZPA2, 17, true, false);
    //
  }
  //-------------------------------------------
  {
    modID = 5;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPAC, 18, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZPASum, 18, true, false);
    module.setChannel(2, IdZPA1, 19, true, false);
    module.setChannel(3, IdZPA2, 19, true, false);
    //
  }
  //-------------------------------------------
  {
    modID = 6;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPCC, 16, true, true, -5, 6, 4, 12);
    module.setChannel(1, IdZEM2, 16, true, true, -5, 6, 4, 12);
    module.setChannel(2, IdZPC1, 17, true, false);
    module.setChannel(3, IdZPC2, 17, true, false);
    //
  }
  //-------------------------------------------
  {
    modID = 7;
    auto& module = conf.modules[modID];
    module.id = modID;
    module.setChannel(0, IdZPCC, 18, false, true, -5, 6, 4, 12);
    module.setChannel(1, IdZPCSum, 18, true, false);
    module.setChannel(2, IdZPC3, 19, true, false);
    module.setChannel(3, IdZPC4, 19, true, false);
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
