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
#include "MFTCondition/DCSConfigUtils.h"
#include "Framework/Logger.h"
#include "TSystem.h"
#include <fstream>

using namespace o2::mft;

//_______________________________________________________________
void DCSConfigReader::init(bool ver)
{
  mVerbose = ver;
  clear();
  mDCSUtils = new DCSConfigUtils();
  mDCSUtils->init();
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

void DCSConfigReader::clearDeadmap()
{
  for (int iChip = 0; iChip < 936; ++iChip) {
    mNoiseMap.resetChip(iChip);
  }
}

//_______________________________________________________________

void DCSConfigReader::parseConfig()
{
  clearDeadmap();

  char delimiter_newline = '\n';
  char delimiter = ',';

  std::stringstream stream(mParams);
  std::string row;

  int nrow = 0;
  int nru = 0;
  int nalpide = 0;
  int nubb = 0;
  int ndeadmap = 0;

  std::string verName = "";
  std::vector<int> arrRUAdd;
  std::vector<int> arrRUVal;
  std::vector<int> arrALPIDEAdd;
  std::vector<int> arrALPIDEVal;
  std::vector<std::string> arrUBBName;
  std::vector<double> arrUBBVal;
  std::vector<int> arrDeadChip;

  while (std::getline(stream, row, delimiter_newline)) {

    std::stringstream srow(row);
    std::string col;

    while (std::getline(srow, col, delimiter)) {

      col.erase(std::remove(col.begin(), col.end(), ' '), col.end());
      int val = atoi(col.c_str());

      if (nrow == mDCSUtils->getVersionNameLineInCsv()) {
        verName = col.c_str();
      } else if (nrow == mDCSUtils->getRUConfigAddressLineInCsv()) {
        arrRUAdd.emplace_back(atoi(col.c_str()));
        ++nru;
      } else if (nrow == mDCSUtils->getRUConfigValueLineInCsv()) {
        arrRUVal.emplace_back(atoi(col.c_str()));
      } else if (nrow == mDCSUtils->getALPIDEConfigAddressLineInCsv()) {
        arrALPIDEAdd.emplace_back(atoi(col.c_str()));
        ++nalpide;
      } else if (nrow == mDCSUtils->getALPIDEConfigValueLineInCsv()) {
        arrALPIDEVal.emplace_back(atoi(col.c_str()));
      } else if (nrow == mDCSUtils->getUBBConfigNameLineInCsv()) {
        col = col.substr(0, 13);
        arrUBBName.emplace_back(col.c_str());
        ++nubb;
      } else if (nrow == mDCSUtils->getUBBConfigValueLineInCsv()) {
        arrUBBVal.emplace_back(atof(col.c_str()));
      } else if (nrow == mDCSUtils->getDeadMapLineInCsv()) {
        arrDeadChip.emplace_back(atoi(col.c_str()));
        ++ndeadmap;
      } else {
        LOG(warning) << "Not expected parameters are sent from DCS!!!!";
      }
    }
    ++nrow;
  }
  if (mVerbose) {
    LOG(info) << "Configuration version / type: " << verName;
    LOG(info) << "Found " << nru << " RU parameters";
  }
  for (int iRUconf = 0; iRUconf < nru; ++iRUconf) {
    if (mVerbose) {
      LOG(info) << "(" << arrRUAdd[iRUconf] << ")   "
                << mDCSUtils->getName(arrRUAdd[iRUconf], "RU")
                << " : " << arrRUVal[iRUconf];
    }
    o2::mft::DCSConfigInfo conf;
    conf.clear();
    conf.setData(arrRUVal[iRUconf]);
    conf.setAdd(arrRUAdd[iRUconf]);
    conf.setType(0); // RU = 0
    conf.setVersion(verName);
    mDCSConfig.emplace_back(conf);
  }

  if (mVerbose) {
    LOG(info) << "Found " << nalpide << " ALPIDE parameters";
  }
  for (int iALPIDEconf = 0; iALPIDEconf < nalpide; ++iALPIDEconf) {
    if (mVerbose) {
      LOG(info) << "(" << arrALPIDEAdd[iALPIDEconf] << ")   "
                << mDCSUtils->getName(arrALPIDEAdd[iALPIDEconf], "ALPIDE")
                << " : " << arrALPIDEVal[iALPIDEconf];
    }
    o2::mft::DCSConfigInfo conf;
    conf.clear();
    conf.setData(arrALPIDEVal[iALPIDEconf]);
    conf.setAdd(arrALPIDEAdd[iALPIDEconf]);
    conf.setType(1); // ALPIDE = 1
    conf.setVersion(verName);
    mDCSConfig.emplace_back(conf);
  }

  if (mVerbose) {
    LOG(info) << "Found " << nubb << " UBB parameters";
  }
  for (int iUBBconf = 0; iUBBconf < nubb; ++iUBBconf) {
    if (mVerbose) {
      LOG(info) << "(" << mDCSUtils->getAddress(arrUBBName[iUBBconf], "UBB") << ")   "
                << arrUBBName[iUBBconf]
                << " : " << arrUBBVal[iUBBconf];
    }
    o2::mft::DCSConfigInfo conf;
    conf.clear();
    conf.setData(arrUBBVal[iUBBconf]);
    conf.setAdd(mDCSUtils->getAddress(arrUBBName[iUBBconf], "UBB"));
    conf.setType(2); // UBB = 2
    conf.setVersion(verName);
    mDCSConfig.emplace_back(conf);
  }

  if (mVerbose) {
    LOG(info) << "Found " << ndeadmap << " Dead chips";
  }
  for (int iDeadMap = 0; iDeadMap < ndeadmap; ++iDeadMap) {
    if (mVerbose) {
      LOG(info) << "Chip" << arrDeadChip[iDeadMap] << " is dead.";
    }
    o2::mft::DCSConfigInfo conf;
    conf.clear();
    conf.setData(1);
    conf.setAdd(arrDeadChip[iDeadMap]);
    conf.setType(3); // DeadMap = 3
    conf.setVersion(verName);
    mDCSConfig.emplace_back(conf);

    mNoiseMap.maskFullChip(arrDeadChip[iDeadMap], true);
  }
}
