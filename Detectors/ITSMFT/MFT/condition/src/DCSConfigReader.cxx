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

#include <MFTCondition/DCSConfigReader.h>
#include "Framework/Logger.h"
#include "TSystem.h"
#include <fstream>

using namespace o2::mft;

//_______________________________________________________________
void DCSConfigReader::init(bool ver)
{
  mVerbose = ver;
  clear();
}

//_______________________________________________________________
void DCSConfigReader::loadConfig(gsl::span<const char> configBuf)
{

  const char* params = configBuf.data();
  mParams = std::string(params);

  parseConfig();
}

void DCSConfigReader::clear()
{
  mDCSConfig.clear();
}

//_______________________________________________________________

void DCSConfigReader::parseConfig()
{

  char delimiter_newline = '\n';
  char delimiter = ',';

  std::stringstream stream(mParams);
  std::string row;

  mNumRow = 0;
  mNumRU = 0;
  mNumALPIDE = 0;

  int arrAddRUConf[128];
  int arrAddALPIDEConf[128];

  int arrValRUConf[128];
  int arrValALPIDEConf[128];

  while (std::getline(stream, row, delimiter_newline)) {

    std::stringstream srow(row);
    std::string col;

    int ncol = 0;

    while (std::getline(srow, col, delimiter)) {
      int val = atoi(col.c_str());
      if (mNumRow == 0) {
        arrAddRUConf[ncol] = val;
        ++mNumRU;
      } else if (mNumRow == 1) {
        arrValRUConf[ncol] = val;
      } else if (mNumRow == 2) {
        arrAddALPIDEConf[ncol] = val;
        ++mNumALPIDE;
      } else if (mNumRow == 3) {
        arrValALPIDEConf[ncol] = val;
      }
      ++ncol;
    }
    ++mNumRow;
  }

  for (int iRUconf = 0; iRUconf < mNumRU; ++iRUconf) {
    if (mVerbose) {
      LOG(info) << "(" << arrAddRUConf[iRUconf] << ")   "
                << " : " << arrValRUConf[iRUconf];
    }
    o2::mft::DCSConfigInfo conf;
    conf.clear();
    conf.setData(arrValRUConf[iRUconf]);
    conf.setAdd(arrAddRUConf[iRUconf]);
    conf.setType(0);
    conf.setVersion("test");
    mDCSConfig.emplace_back(conf);
  }

  for (int iALPIDEconf = 0; iALPIDEconf < mNumALPIDE; ++iALPIDEconf) {
    if (mVerbose) {
      LOG(info) << "(" << arrAddALPIDEConf[iALPIDEconf] << ")   "
                << " : " << arrValALPIDEConf[iALPIDEconf];
    }
    o2::mft::DCSConfigInfo conf;
    conf.clear();
    conf.setData(arrValALPIDEConf[iALPIDEconf]);
    conf.setAdd(arrAddALPIDEConf[iALPIDEconf]);
    conf.setType(1);
    conf.setVersion("test");
    mDCSConfig.emplace_back(conf);
  }
}
