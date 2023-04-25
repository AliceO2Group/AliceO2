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

#ifndef O2_DATAPROCESSINGSTATEMANAGER_H_
#define O2_DATAPROCESSINGSTATEMANAGER_H_

#include <array>
#include <vector>
#include <string>

struct DataProcessingStateManager {
  struct StateIndex {
    short id = -1;
    short index = -1;
  };
  struct StateInfo {
    std::string name;
    int64_t lastUpdate = 0;
    int index = -1;
  };

  static constexpr int MAX_STATES = 1024;
  std::vector<std::array<char, 1024>> states = {};
  std::vector<StateInfo> infos = {};
};

#endif
