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

#ifndef O2_MFT_DCSCONFIGINFO_H
#define O2_MFT_DCSCONFIGINFO_H

#include <TString.h>
#include <unordered_map>
#include <iostream>

namespace o2
{
namespace mft
{
class DCSConfigInfo
{

 public:
  void clear()
  {
    mData = -999;
    mAdd = -999;
    mType = -999;
    mVersion = "v0";
  }
  void setData(int val)
  {
    mData = val;
  }
  void setAdd(int val)
  {
    mAdd = val;
  }
  void setType(int val)
  {
    mType = val;
  }
  void setVersion(std::string str)
  {
    mVersion = str;
  }

  const int& getData() const
  {
    return mData;
  }
  const int& getAdd() const
  {
    return mAdd;
  }
  const int& getType() const
  {
    return mType;
  }
  const std::string& getVersion() const
  {
    return mVersion;
  }

 private:
  int mData;
  int mAdd;
  int mType; // RU = 0, ALPIDE = 1, UBB = 2, DeadMap = 3
  std::string mVersion;

  ClassDefNV(DCSConfigInfo, 1);
};
} // namespace mft
} // namespace o2

#endif
