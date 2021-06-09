// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CreateCTPConfig.C
/// \brief create CTP config, test it and add to database
/// \author Roman Lietava

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "FairLogger.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsCTP/Configuration.h"
#include <string>
#include <map>
#include <iostream>
#endif
using namespace o2::ctp;
void CreateCTPConfig(long tmin = 0, long tmax = -1, std::string ccdbHost = "http://ccdb-test.cern.ch:8080")
{
  /// Demo configuration
  CTPConfiguration ctpcfg;
  std::string cfgstr = "PARTITION: TEST \n";
  cfgstr += "VERSION:0 \n";
  cfgstr += "INPUTS: \n";
  cfgstr += "V0A FV0 M 0x1 \n";
  cfgstr += "V0B FV0 M 0x10 \n";
  cfgstr += "T0A FT0 M 0x100 \n";
  cfgstr += "T0B FT0 M 0x1000 \n";
  cfgstr += "DESCRIPTORS: \n";
  cfgstr += "DV0A V0A \n";
  cfgstr += "DV0B V0B \n";
  cfgstr += "DV0AND V0A V0B \n";
  cfgstr += "DT0AND T0A T0B \n";
  cfgstr += "DT0A T0A \n";
  cfgstr += "DT0B T0B \n";
  cfgstr += "DINT4 V0A V0B T0A T0B \n";
  cfgstr += "CLUSTERS: ALL\n";
  cfgstr += "ALL FV0 FT0 TPC \n";
  cfgstr += "CLASSES:\n";
  cfgstr += "CMB1 0 DV0AND ALL \n";
  cfgstr += "CV0A 1 DV0A ALL \n";
  cfgstr += "CV0B 2 DV0B ALL \n";
  cfgstr += "CMB2 3 DT0AND ALL \n";
  cfgstr += "CT0A 4 DT0A ALL \n";
  cfgstr += "CT0B 62 DT0B ALL \n";
  cfgstr += "CINT4 63 DINT4 ALL \n";

  ctpcfg.loadConfiguration(cfgstr);
  ctpcfg.printStream(std::cout);
  ///
  /// add to database
  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&ctpcfg, o2::ctp::CCDBPathCTPConfig, metadata, tmin, tmax);
  std::cout << "CTP config in database" << std::endl;
  /// get frp, database
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(ccdbHost);
  auto ctpconfigdb = mgr.get<CTPConfiguration>(CCDBPathCTPConfig);
  ctpconfigdb->printStream(std::cout);
}
