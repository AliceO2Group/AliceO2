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

/// \file SectorEdgeFluctuations.cxx
/// \brief Class to parse and query time-dependent TPC sector edge fluctuation intervals

#include "TPCCalibration/SectorEdgeFluctuations.h"
#include "DataFormatsTPC/Defs.h"
#include "TTree.h"
#include "TFile.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <map>

#include "Framework/Logger.h"

namespace o2::tpc
{

int SectorEdgeFluctuations::parseSectorId(const std::string& sectorStr)
{
  if (sectorStr.size() < 2) {
    return -1;
  }

  const char side = std::toupper(static_cast<unsigned char>(sectorStr[0]));
  if (side != 'A' && side != 'C') {
    return -1;
  }

  int num = -1;
  try {
    num = std::stoi(sectorStr.substr(1));
  } catch (...) {
    return -1;
  }

  if (num < 0 || num > 17) {
    return -1;
  }

  return (side == 'A') ? num : (num + SECTORSPERSIDE);
}

bool SectorEdgeFluctuations::loadFromCSVFile(const std::string& filename)
{
  mIntervals.clear();

  std::ifstream file(filename);
  if (!file.is_open()) {
    LOGP(error, "SectorEdgeFluctuations: cannot open file: {}", filename);
    return false;
  }

  std::string line;
  int lineNum = 0;
  int nSkipped = 0;
  int nLoaded = 0;

  while (std::getline(file, line)) {
    ++lineNum;

    // skip empty lines and comments
    const auto firstNonSpace = line.find_first_not_of(" \t\r\n");
    if (firstNonSpace == std::string::npos || line[firstNonSpace] == '#') {
      continue;
    }

    // tokenise on comma
    std::vector<std::string> tokens;
    {
      std::stringstream ss(line);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        // trim leading/trailing whitespace
        const auto s = tok.find_first_not_of(" \t\r\n");
        const auto e = tok.find_last_not_of(" \t\r\n");
        tokens.push_back((s == std::string::npos) ? "" : tok.substr(s, e - s + 1));
      }
    }

    // minimum: runNum(0), startMS(1), endMS(2), duration(3), label(4); sectors are optional
    if (tokens.size() < 5) {
      LOGP(warning, "SectorEdgeFluctuations: skipping malformed line {}", lineNum);
      ++nSkipped;
      continue;
    }

    int run = -1;
    try {
      run = std::stoi(tokens[0]);
    } catch (...) {
      LOGP(warning, "SectorEdgeFluctuations: cannot parse run number on line {}", lineNum);
      ++nSkipped;
      continue;
    }

    SectorEdgeInterval interval;
    try {
      interval.startTimeMS = std::stoll(tokens[1]);
      interval.endTimeMS = std::stoll(tokens[2]);
    } catch (...) {
      LOGP(warning, "SectorEdgeFluctuations: cannot parse timestamps on line {}", lineNum);
      ++nSkipped;
      continue;
    }

    if (interval.endTimeMS < interval.startTimeMS) {
      LOGP(warning, "SectorEdgeFluctuations: end < start on line {}, skipping", lineNum);
      ++nSkipped;
      continue;
    }

    // tokens[4] is the human-readable label; sectors start at index 5
    for (size_t i = 5; i < tokens.size(); ++i) {
      std::string sectorStr = tokens[i];
      float scale = 1.0f;

      // parse optional "SectorID=scale" suffix
      const auto eqPos = sectorStr.find('=');
      if (eqPos != std::string::npos) {
        try {
          scale = std::stof(sectorStr.substr(eqPos + 1));
        } catch (...) {
          LOGP(warning, "SectorEdgeFluctuations: cannot parse scale in '{}' on line {}, using 1.0", tokens[i], lineNum);
        }
        sectorStr = sectorStr.substr(0, eqPos);
      }

      const int sectorId = parseSectorId(sectorStr);
      if (sectorId < 0) {
        LOGP(warning, "SectorEdgeFluctuations: unknown sector '{}' on line {}, skipping token", sectorStr, lineNum);
        continue;
      }

      // deduplicate: last occurrence in the line wins
      auto dup = std::find_if(interval.sectors.begin(), interval.sectors.end(), [sectorId](const std::pair<int, float>& p) { return p.first == sectorId; });
      if (dup != interval.sectors.end()) {
        dup->second = scale;
      } else {
        interval.sectors.emplace_back(sectorId, scale);
      }
    }

    if (interval.sectors.empty()) {
      // no sector tokens (or all invalid): apply interval to all 36 sectors
      const int nSec = SECTORSPERSIDE * SIDES;
      for (int s = 0; s < nSec; ++s) {
        interval.sectors.emplace_back(s, 1.0f);
      }
    }
    mIntervals[run].push_back(std::move(interval));
    ++nLoaded;
  }

  // sort each run's intervals by start time so getSectorsAtTime can break early
  for (auto& [run, intervals] : mIntervals) {
    std::sort(intervals.begin(), intervals.end(), [](const SectorEdgeInterval& a, const SectorEdgeInterval& b) {
      return a.startTimeMS < b.startTimeMS;
    });
  }

  LOGP(info, "SectorEdgeFluctuations: loaded {} intervals for {} run(s) from '{}' ({} lines skipped)", nLoaded, mIntervals.size(), filename, nSkipped);
  return true;
}

std::vector<std::pair<int, float>> SectorEdgeFluctuations::getSectorsAtTime(int run, Long64_t timestampMS) const
{
  const auto runIt = mIntervals.find(run);
  if (runIt == mIntervals.end()) {
    return {};
  }

  // Collect all sectors whose interval is active at timestampMS.
  // When the same sector appears in multiple overlapping intervals, keep the
  // scale from the interval with the latest endTimeMS (most specific).
  //   sectorBestScale: sectorId -> {scale, endTimeMS}
  std::map<int, std::pair<float, Long64_t>> sectorBestScale;

  const auto& intervals = runIt->second;
  const auto endIt = std::upper_bound(intervals.begin(), intervals.end(), timestampMS,[](Long64_t ts, const SectorEdgeInterval& iv) { return ts < iv.startTimeMS; });

  for (auto it = intervals.begin(); it != endIt; ++it) {
    if (it->endTimeMS < timestampMS) {
      continue;
    }
    for (const auto& [sector, scale] : it->sectors) {
      auto sit = sectorBestScale.find(sector);
      if (sit == sectorBestScale.end() || it->endTimeMS > sit->second.second) {
        sectorBestScale[sector] = {scale, it->endTimeMS};
      }
    }
  }

  std::vector<std::pair<int, float>> result;
  result.reserve(sectorBestScale.size());
  for (const auto& [sector, scaleAndEnd] : sectorBestScale) {
    result.emplace_back(sector, scaleAndEnd.first);
  }
  return result;
}

void SectorEdgeFluctuations::dumpToFile(const char* file, const char* name, const char* brName)
{
  TFile out(file, "RECREATE");
  TTree tree(name, name);
  tree.SetAutoSave(0);
  tree.Branch(brName, this);
  tree.Fill();
  tree.Write();
  out.Close();
}

void SectorEdgeFluctuations::loadFromFile(const char* inpf, const char* name, const int iEntry, const char* brName)
{
  TFile inp(inpf, "READ");
  if (inp.IsZombie() || !inp.IsOpen()) {
    LOGP(error, "SectorEdgeFluctuations: cannot open file '{}'", inpf);
    return;
  }
  TTree* tree = dynamic_cast<TTree*>(inp.Get(name));
  if (!tree) {
    LOGP(error, "SectorEdgeFluctuations: object '{}' not found or not a TTree in '{}'", name, inpf);
    return;
  }
  setFromTree(*tree, iEntry, brName);
}

void SectorEdgeFluctuations::setFromTree(TTree& tree, const int iEntry, const char* brName)
{
  SectorEdgeFluctuations* msecFlucTmp = this;
  tree.SetBranchAddress(brName, &msecFlucTmp);
  const int entries = tree.GetEntries();
  if (entries > iEntry) {
    tree.GetEntry(iEntry);
  } else {
    LOGP(error, "SectorEdgeFluctuation not found in input file");
  }
  tree.SetBranchAddress(brName, nullptr);
}

} // namespace o2::tpc
