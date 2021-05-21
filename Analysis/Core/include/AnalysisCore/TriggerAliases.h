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

#include <cstdint>
#include <map>
#include <string>
#include <Rtypes.h>

enum triggerAliases {
  kINT7 = 0,
  kEMC7,
  kINT7inMUON,
  kMuonSingleLowPt7,
  kMuonSingleHighPt7,
  kMuonUnlikeLowPt7,
  kMuonLikeLowPt7,
  kCUP8,
  kCUP9,
  kMUP10,
  kMUP11,
  kALL,
  kNaliases
};

static const char* aliasLabels[kNaliases] = {
  "kINT7",
  "kEMC7",
  "kINT7inMUON",
  "kMuonSingleLowPt7",
  "kMuonSingleHighPt7",
  "kMuonUnlikeLowPt7",
  "kMuonLikeLowPt7",
  "kCUP8",
  "kCUP9",
  "kMUP10",
  "kALL",
  "kMUP11"};

class TriggerAliases
{
 public:
  TriggerAliases() = default;
  ~TriggerAliases() = default;

  void AddAlias(uint32_t aliasId, std::string classNames) { mAliasToClassNames[aliasId] = classNames; }
  void AddClassIdToAlias(uint32_t aliasId, int classId);
  const std::map<uint32_t, std::string>& GetAliasToClassNamesMap() const { return mAliasToClassNames; }
  const std::map<uint32_t, uint64_t>& GetAliasToTriggerMaskMap() const { return mAliasToTriggerMask; }
  const std::map<uint32_t, uint64_t>& GetAliasToTriggerMaskNext50Map() const { return mAliasToTriggerMaskNext50; }
  void Print();

 private:
  std::map<uint32_t, std::string> mAliasToClassNames;
  std::map<uint32_t, uint64_t> mAliasToTriggerMask;
  std::map<uint32_t, uint64_t> mAliasToTriggerMaskNext50;
  ClassDefNV(TriggerAliases, 2)
};

#endif
