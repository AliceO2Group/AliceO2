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

  ~DCSConfigUtils() = default;

  void init()
  {
    initDictionary();
  }

  void clear()
  {
  }

  auto getAddress(const std::string name, const std::string type)
  {
    return static_cast<int>((mNameDictType[type])[name]);
  }

  const std::string getName(int add, const std::string type)
  {
    return static_cast<const std::string>((mAddressDictType[type])[add]);
  }

  std::map<int, const std::string>& getAddressMap(const std::string type)
  {
    return mAddressDictType[type];
  }

  std::map<int, const std::string>& getAddressMap(int type)
  {
    return mAddressDictType[getTypeName(type)];
  }

  std::map<const std::string, int>& getNameMap(const std::string type)
  {
    return mNameDictType[type];
  }

  std::map<const std::string, int>& getNameMap(int type)
  {
    return mNameDictType[getTypeName(type)];
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

  const int getType(const std::string type)
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

  const std::string& getVersion()
  {
    return mVersion;
  }

  auto& getVersionNameLineInCsv() const
  {
    return mVersionNameLine;
  }
  auto& getRuConfigAddressLineInCsv() const
  {
    return mRuConfigAddressLine;
  }
  auto& getRuConfigValueLineInCsv() const
  {
    return mRuConfigValueLine;
  }
  auto& getAlpideConfigAddressLineInCsv() const
  {
    return mAlpideConfigAddressLine;
  }
  auto& getAlpideConfigValueLineInCsv() const
  {
    return mAlpideConfigValueLine;
  }
  auto& getUbbConfigNameLineInCsv() const
  {
    return mUbbNameLine;
  }
  auto& getUbbConfigValueLineInCsv() const
  {
    return mUbbValueLine;
  }
  auto& getDeadMapLineInCsv() const
  {
    return mDeadMapLine;
  }

 private:
  const int mVersionNameLine = 0;
  const int mAlpideConfigAddressLine = 1;
  const int mAlpideConfigValueLine = 2;
  const int mRuConfigAddressLine = 3;
  const int mRuConfigValueLine = 4;
  const int mUbbNameLine = 5;
  const int mUbbValueLine = 6;
  const int mDeadMapLine = 7;

  const std::string mVersion;

  std::map<const std::string, std::map<const std::string, int>> mNameDictType;
  std::map<const std::string, int> mNameDictRu;
  std::map<const std::string, int> mNameDictAlpide;
  std::map<const std::string, int> mNameDictUbb;
  std::map<const std::string, int> mNameDictDeadMap;

  std::map<const std::string, std::map<int, const std::string>> mAddressDictType;
  std::map<int, const std::string> mAddressDictRu;
  std::map<int, const std::string> mAddressDictAlpide;
  std::map<int, const std::string> mAddressDictUbb;
  std::map<int, const std::string> mAddressDictDeadMap;

  void initDictionary()
  {
    mNameDictType.clear();
    mNameDictRu.clear();
    mNameDictAlpide.clear();
    mNameDictUbb.clear();
    mNameDictDeadMap.clear();

    mAddressDictType.clear();
    mAddressDictRu.clear();
    mAddressDictAlpide.clear();
    mAddressDictUbb.clear();
    mAddressDictDeadMap.clear();

    std::vector<std::pair<const std::string, uint>> pairRu;
    pairRu.push_back(std::make_pair("MANCHESTER", 1046));
    pairRu.push_back(std::make_pair("ENABLE", 4096));
    pairRu.push_back(std::make_pair("TRIGGER_PERIOD", 4097));
    pairRu.push_back(std::make_pair("PULSE_nTRIGGER", 4098));
    pairRu.push_back(std::make_pair("TRIGGER_MIN_DISTANCE", 4099));
    pairRu.push_back(std::make_pair("OPCODE_GATING", 4101));
    pairRu.push_back(std::make_pair("TRIGGER_DELAY", 4102));
    pairRu.push_back(std::make_pair("ENABLE_PACKER_0", 4103));
    pairRu.push_back(std::make_pair("ENABLE_PACKER_1", 4104));
    pairRu.push_back(std::make_pair("ENABLE_PACKER_2", 4105));
    pairRu.push_back(std::make_pair("TRIG_SOURCE", 4106));
    pairRu.push_back(std::make_pair("TIMEOUT_TO_START", 5376));
    pairRu.push_back(std::make_pair("TIMEOUT_TO_STOP", 5377));
    pairRu.push_back(std::make_pair("TIMEOUT_IN_IDLE", 5378));
    pairRu.push_back(std::make_pair("GBT_LOAD_BALANCING", 5631));

    std::vector<std::pair<const std::string, uint>> pairAlpide;
    pairAlpide.push_back(std::make_pair("Mode_Control_Register", 1));
    pairAlpide.push_back(std::make_pair("FROMU_Configration_Register_1", 4));
    pairAlpide.push_back(std::make_pair("FROMU_Configration_Register_2", 5));
    pairAlpide.push_back(std::make_pair("FROMU_Configration_Register_3", 6));
    pairAlpide.push_back(std::make_pair("FROMU_Pulsing_Register_1", 7));
    pairAlpide.push_back(std::make_pair("FROMU_Pulsing_Register_2", 8));
    pairAlpide.push_back(std::make_pair("CMUandDMU_Configration_Register", 16));
    pairAlpide.push_back(std::make_pair("DTU_Configration_Register", 20));
    pairAlpide.push_back(std::make_pair("DTU_DACs_Register", 21));
    pairAlpide.push_back(std::make_pair("DTU_Test_Register_1", 24));
    pairAlpide.push_back(std::make_pair("DTU_Test_Register_2", 25));
    pairAlpide.push_back(std::make_pair("DTU_Test_Register_3", 26));
    pairAlpide.push_back(std::make_pair("VCASP", 1539));
    pairAlpide.push_back(std::make_pair("VCLIP", 1544));
    pairAlpide.push_back(std::make_pair("IBIAS", 1549));
    pairAlpide.push_back(std::make_pair("VPULSEH", 255));
    pairAlpide.push_back(std::make_pair("VPULSEL", 0));
    pairAlpide.push_back(std::make_pair("VRESETD", 1538));
    pairAlpide.push_back(std::make_pair("IDB", 1548));
    pairAlpide.push_back(std::make_pair("AUTO_ROF__NOISE_MASK__MASK_LEV__MC_HIT__MC_ID", 65535));

    std::vector<std::pair<const std::string, uint>> pairUbb;
    int idUbb = 0;
    for (int iH = 0; iH <= 1; ++iH) {
      for (int iD = 0; iD <= 4; ++iD) {
        for (int iF = 0; iF <= 1; ++iF) {
          for (int iZ = 0; iZ <= 3; ++iZ) {
            pairUbb.push_back(std::make_pair(Form("U_BB_H%dD%dF%dZ%d", iH, iD, iF, iZ), idUbb));
            ++idUbb;
          }
        }
      }
    }

    std::vector<std::pair<const std::string, uint>> pairDeadMap;
    int idDeadMap = 0;
    for (int iC = 0; iC < 936; ++iC) {
      pairDeadMap.push_back(std::make_pair(Form("ChipID%d", iC), idDeadMap));
      ++idDeadMap;
    }

    for (int iRu = 0; iRu < pairRu.size(); ++iRu) {
      std::pair<const std::string, int> p = pairRu[iRu];
      mAddressDictRu.emplace(p.second, p.first);
      mNameDictRu.emplace(p.first, p.second);
    }
    for (int iAlpide = 0; iAlpide < pairAlpide.size(); ++iAlpide) {
      std::pair<const std::string, int> p = pairAlpide[iAlpide];
      mAddressDictAlpide.emplace(p.second, p.first);
      mNameDictAlpide.emplace(p.first, p.second);
    }
    for (int iUbb = 0; iUbb < pairUbb.size(); ++iUbb) {
      std::pair<const std::string, int> p = pairUbb[iUbb];
      mAddressDictUbb.emplace(p.second, p.first);
      mNameDictUbb.emplace(p.first, p.second);
    }
    for (int iDeadMap = 0; iDeadMap < pairDeadMap.size(); ++iDeadMap) {
      std::pair<const std::string, int> p = pairDeadMap[iDeadMap];
      mAddressDictDeadMap.emplace(p.second, p.first);
      mNameDictDeadMap.emplace(p.first, p.second);
    }

    mNameDictType.emplace("RU", mNameDictRu);
    mNameDictType.emplace("ALPIDE", mNameDictAlpide);
    mNameDictType.emplace("UBB", mNameDictUbb);
    mNameDictType.emplace("DeadMap", mNameDictDeadMap);

    mAddressDictType.emplace("RU", mAddressDictRu);
    mAddressDictType.emplace("ALPIDE", mAddressDictAlpide);
    mAddressDictType.emplace("UBB", mAddressDictUbb);
    mAddressDictType.emplace("DeadMap", mAddressDictDeadMap);
  }

  ClassDefNV(DCSConfigUtils, 1);

}; // end class
} // namespace mft
} // namespace o2

#endif
