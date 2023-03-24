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

/// \file checkTrapConfig.C
/// \brief Extract information from a TrapConfig or series of them and then either plot or dump text
/// \author Sean Murray

#if !defined(__CLING__) || defined(__ROOTCLING__)
// ROOT header
#include <TFile.h>
#include <TAxis.h>
#include <TGraph.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TMultiGraph.h>

// O2 header
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsTRD/TrapConfig.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "DataFormatsTRD/Constants.h"

#include <map>
#include <string>
#include <vector>
#endif

using namespace std;
using namespace o2::trd;
using namespace o2::trd::constants;
using timePoint = o2::parameters::GRPECSObject::timePoint;

std::map<timePoint, std::unique_ptr<TrapConfig>> trapConfigMap;
// Download the values and populate the map.
void ccdbDownload(unsigned int runNumber, std::string ccdb, timePoint queryInterval)
{
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL("http://alice-ccdb.cern.ch/");
  auto runDuration = ccdbMgr.getRunDuration(runNumber);
  std::map<std::string, std::string> md;
  md["runNumber"] = std::to_string(runNumber);
  const auto* grp = ccdbMgr.getSpecific<o2::parameters::GRPECSObject>("GLO/Config/GRPECS", (runDuration.first + runDuration.second) / 2, md);
  grp->print();
  const auto startTime = grp->getTimeStart();
  const auto endTime = grp->getTimeEnd();
  ccdbMgr.setURL(ccdb);

  for (timePoint time = startTime; time < endTime; time += queryInterval) {
    ccdbMgr.setTimestamp(time);
    std::cout << "Downloading TrapConfig at time " << time << std::endl;
    // std::unique_ptr<TrapConfig> trapconfig3 = std::make_unique<TrapConfig>(ccdbMgr.get<o2::trd::TrapConfig>("TRD/TrapConfig/TrapConfig"));
    // trapConfigMap[time].push_back(trapconfig3);
    trapConfigMap[time].push_back(std::make_unique<o2::trd::TrapConfig>(ccdbMgr.get<o2::trd::TrapConfig>("TRD/TrapConfig/TrapConfig")));
    /*    for (int iDet = 0; iDet < 540; ++iDet) {
      vmap[iDet].push_back(std::make_tuple(
        calVdriftExB->getVdrift(iDet), calVdriftExB->getExB(iDet), static_cast<int>(time - startTime)));
    }*/
  }
}

void CheckTrapConfig(unsigned int runNumber = 523677, std::string ccdb = "http://ccdb-test.cern.ch:8080", timePoint queryInterval = 900000)
{
  //  ccdbDownload(runNumber, ccdb, queryInterval);
}
