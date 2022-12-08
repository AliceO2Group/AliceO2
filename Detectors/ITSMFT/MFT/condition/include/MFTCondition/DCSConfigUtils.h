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

#ifndef O2_MFT_DNSUTILS_H
#define O2_MFT_DNSUTILS_H

/// @file   DCSConfigUtils.h
/// @brief  MFT Processor for DCS Config

#include <TString.h>
#include "Framework/Logger.h"
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

namespace o2
{
namespace mft
{

class DCSConfigUtils
{

 public:
  DCSConfigUtils()
  {
    init();
  }
  void init()
  {
    initDictionary();
  }

  void clear()
  {
  }

  int getAddress(std::string name, std::string type)
  {
    return (mNameDict[type])[name];
  }

  std::string getName(int add, std::string type)
  {
    return (mAddressDict[type])[add];
  }

  std::map<int, std::string> getAddressMap(std::string type)
  {
    return mAddressDict[type];
  }

  std::map<int, std::string> getAddressMap(int type)
  {
    return mAddressDict[getTypeName(type)];
  }

  std::map<std::string, int> getNameMap(std::string type)
  {
    return mNameDict[type];
  }

  std::map<std::string, int> getNameMap(int type)
  {
    return mNameDict[getTypeName(type)];
  }

  const std::string getTypeName(int type)
  {
    if (type == 0) {
      return "RU";
    } else if (type == 1) {
      return "ALPIDE";
    } else if (type == 2) {
      return "UBB";
    } else if (type == 3) {
      return "DeadMap";
    } else {
      LOG(error) << "You can select 0 (RU), 1 (ALPIDE), 2 (UBB), 3 (DeadMap)";
      return "Unknown";
    }
  }

  const int getType(std::string type)
  {
    if (type == "RU") {
      return 0;
    } else if (type == "ALPIDE") {
      return 1;
    } else if (type == "UBB") {
      return 2;
    } else if (type == "DeadMap") {
      return 3;
    } else {
      LOG(error) << "You can select RU (0), ALPIDE (1), UBB (2), DeadMap (3)";
      return -999;
    }
  }

  std::string& getVersion()
  {
    return mVersion;
  }

  int getVersionNameLineInCsv()
  {
    return mVerNameLine;
  }
  int getRUConfigAddressLineInCsv()
  {
    return mRUConfAddLine;
  }
  int getRUConfigValueLineInCsv()
  {
    return mRUConfValLine;
  }
  int getALPIDEConfigAddressLineInCsv()
  {
    return mALPIDEConfAddLine;
  }
  int getALPIDEConfigValueLineInCsv()
  {
    return mALPIDEConfValLine;
  }
  int getUBBConfigNameLineInCsv()
  {
    return mUBBNameLine;
  }
  int getUBBConfigValueLineInCsv()
  {
    return mUBBValLine;
  }
  int getDeadMapLineInCsv()
  {
    return mDeadMapLine;
  }

 private:
  const int mVerNameLine = 0;
  const int mALPIDEConfAddLine = 1;
  const int mALPIDEConfValLine = 2;
  const int mRUConfAddLine = 3;
  const int mRUConfValLine = 4;
  const int mUBBNameLine = 5;
  const int mUBBValLine = 6;
  const int mDeadMapLine = 7;

  std::string mVersion;

  std::map<std::string, std::map<std::string, int>> mNameDict;
  std::map<std::string, int> mNameDictRU;
  std::map<std::string, int> mNameDictALPIDE;
  std::map<std::string, int> mNameDictUBB;

  std::map<std::string, std::map<int, std::string>> mAddressDict;
  std::map<int, std::string> mAddressDictRU;
  std::map<int, std::string> mAddressDictALPIDE;
  std::map<int, std::string> mAddressDictUBB;

  void initDictionary()
  {
    mNameDict.clear();
    mNameDictRU.clear();
    mNameDictALPIDE.clear();
    mNameDictUBB.clear();

    mAddressDict.clear();
    mAddressDictRU.clear();
    mAddressDictALPIDE.clear();
    mAddressDictUBB.clear();

    std::vector<std::pair<std::string, uint>> pairRU;
    pairRU.push_back(std::make_pair("MANCHESTER", 1046));
    pairRU.push_back(std::make_pair("ENABLE", 4096));
    pairRU.push_back(std::make_pair("TRIGGER_PERIOD", 4097));
    pairRU.push_back(std::make_pair("PULSE_nTRIGGER", 4098));
    pairRU.push_back(std::make_pair("TRIGGER_MIN_DISTANCE", 4099));
    pairRU.push_back(std::make_pair("OPCODE_GATING", 4101));
    pairRU.push_back(std::make_pair("TRIGGER_DELAY", 4102));
    pairRU.push_back(std::make_pair("ENABLE_PACKER_0", 4103));
    pairRU.push_back(std::make_pair("ENABLE_PACKER_1", 4104));
    pairRU.push_back(std::make_pair("ENABLE_PACKER_2", 4105));
    pairRU.push_back(std::make_pair("TRIG_SOURCE", 4106));
    pairRU.push_back(std::make_pair("TIMEOUT_TO_START", 5376));
    pairRU.push_back(std::make_pair("TIMEOUT_TO_STOP", 5377));
    pairRU.push_back(std::make_pair("TIMEOUT_IN_IDLE", 5378));
    pairRU.push_back(std::make_pair("GBT_LOAD_BALANCING", 5631));

    std::vector<std::pair<std::string, uint>> pairALPIDE;
    pairALPIDE.push_back(std::make_pair("Mode_Control_Register", 1));
    pairALPIDE.push_back(std::make_pair("FROMU_Configration_Register_1", 4));
    pairALPIDE.push_back(std::make_pair("FROMU_Configration_Register_2", 5));
    pairALPIDE.push_back(std::make_pair("FROMU_Configration_Register_3", 6));
    pairALPIDE.push_back(std::make_pair("FROMU_Pulsing_Register_1", 7));
    pairALPIDE.push_back(std::make_pair("FROMU_Pulsing_Register_2", 8));
    pairALPIDE.push_back(std::make_pair("CMUandDMU_Configration_Register", 16));
    pairALPIDE.push_back(std::make_pair("DTU_Configration_Register", 20));
    pairALPIDE.push_back(std::make_pair("DTU_DACs_Register", 21));
    pairALPIDE.push_back(std::make_pair("DTU_Test_Register_1", 24));
    pairALPIDE.push_back(std::make_pair("DTU_Test_Register_2", 25));
    pairALPIDE.push_back(std::make_pair("DTU_Test_Register_3", 26));
    pairALPIDE.push_back(std::make_pair("VCASP", 1539));
    pairALPIDE.push_back(std::make_pair("VCLIP", 1544));
    pairALPIDE.push_back(std::make_pair("IBIAS", 1549));
    pairALPIDE.push_back(std::make_pair("VPULSEH", 255));
    pairALPIDE.push_back(std::make_pair("VPULSEL", 0));
    pairALPIDE.push_back(std::make_pair("VRESETD", 1538));
    pairALPIDE.push_back(std::make_pair("IDB", 1548));
    pairALPIDE.push_back(std::make_pair("AUTO_ROF__NOISE_MASK__MASK_LEV__MC_HIT__MC_ID", 65535));

    std::vector<std::pair<std::string, uint>> pairUBB;
    int idUBB = 0;
    for (int iH = 0; iH <= 1; ++iH) {
      for (int iD = 0; iD <= 4; ++iD) {
        for (int iF = 0; iF <= 1; ++iF) {
          for (int iZ = 0; iZ <= 3; ++iZ) {
            pairUBB.push_back(std::make_pair(Form("U_BB_H%dD%dF%dZ%d", iH, iD, iF, iZ), idUBB));
            ++idUBB;
          }
        }
      }
    }

    for (int iRU = 0; iRU < pairRU.size(); ++iRU) {
      std::pair<std::string, int> p = pairRU[iRU];
      mAddressDictRU.emplace(p.second, p.first);
      mNameDictRU.emplace(p.first, p.second);
    }
    for (int iALPIDE = 0; iALPIDE < pairALPIDE.size(); ++iALPIDE) {
      std::pair<std::string, int> p = pairALPIDE[iALPIDE];
      mAddressDictALPIDE.emplace(p.second, p.first);
      mNameDictALPIDE.emplace(p.first, p.second);
    }
    for (int iUBB = 0; iUBB < pairUBB.size(); ++iUBB) {
      std::pair<std::string, int> p = pairUBB[iUBB];
      mAddressDictUBB.emplace(p.second, p.first);
      mNameDictUBB.emplace(p.first, p.second);
    }

    mNameDict.emplace("RU", mNameDictRU);
    mNameDict.emplace("ALPIDE", mNameDictALPIDE);
    mNameDict.emplace("UBB", mNameDictUBB);

    mAddressDict.emplace("RU", mAddressDictRU);
    mAddressDict.emplace("ALPIDE", mAddressDictALPIDE);
    mAddressDict.emplace("UBB", mAddressDictUBB);
  }

  ClassDefNV(DCSConfigUtils, 1);

}; // end class
} // namespace mft
} // namespace o2

#endif
