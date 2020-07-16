// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TriggerAliases_H
#define TriggerAliases_H

#include <map>
#include <string>
#include <vector>
#include <Rtypes.h>

enum triggerAliases {
  kINT7 = 0,
  kEMC7,
  kINT7inMUON,
  kMuonSingleLowPt7,
  kMuonUnlikeLowPt7,
  kMuonLikeLowPt7,
  kCUP8,
  kCUP9,
  kMUP10,
  kMUP11,
  kNaliases
};

class TriggerAliases
{
 public:
  TriggerAliases() = default;
  ~TriggerAliases() = default;
  void AddAlias(int aliasId, std::string classNames) { mAliases[aliasId] = classNames; }
  void AddClassIdToAlias(int aliasId, int classId) { mAliasToClassIds[aliasId].push_back(classId); }
  const std::map<int, std::vector<int>>& GetAliasToClassIdsMap() const { return mAliasToClassIds; }
  void Print();

 private:
  std::map<int, std::string> mAliases;
  std::map<int, std::vector<int>> mAliasToClassIds;
  ClassDefNV(TriggerAliases, 1)
};

#endif
