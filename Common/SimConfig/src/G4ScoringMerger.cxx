// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimConfig/G4ScoringMerger.h"
#include <fairlogger/Logger.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <sstream>
#include <vector>

namespace o2::conf
{

namespace
{
// One scorer block of a Geant4 mesh dump: its header lines and the summed rows
struct ScorerBlock {
  std::vector<std::string> header;
  std::vector<std::string> keys; // "iZ,iPHI,iR" in file order
  std::vector<double> sum;
  std::vector<double> sum2;
  std::vector<long> entries;
};

// Read one mesh dump into scorer blocks; returns false on a format error
bool readDump(const std::string& fileName, std::vector<std::string>& meshHeader, std::vector<ScorerBlock>& blocks)
{
  std::ifstream in(fileName);
  if (!in) {
    return false;
  }
  std::string line;
  ScorerBlock* current = nullptr;
  while (std::getline(in, line)) {
    if (line.rfind("# mesh name", 0) == 0) {
      meshHeader.push_back(line);
    } else if (line.rfind("# primitive scorer name", 0) == 0) {
      blocks.emplace_back();
      current = &blocks.back();
      current->header.push_back(line);
    } else if (line.rfind("#", 0) == 0) {
      if (!current) {
        return false;
      }
      current->header.push_back(line);
    } else if (!line.empty()) {
      if (!current) {
        return false;
      }
      // iZ, iPHI, iR, total, total^2, entries
      std::vector<std::string> fields;
      std::stringstream ss(line);
      std::string field;
      while (std::getline(ss, field, ',')) {
        fields.push_back(field);
      }
      if (fields.size() != 6) {
        return false;
      }
      current->keys.push_back(fields[0] + "," + fields[1] + "," + fields[2]);
      current->sum.push_back(std::stod(fields[3]));
      current->sum2.push_back(std::stod(fields[4]));
      current->entries.push_back(std::stol(fields[5]));
    }
  }
  return !blocks.empty();
}
} // namespace

std::string g4ScoringWorkerFileName(const std::string& meshName, int pid)
{
  return meshName + ".worker" + std::to_string(pid) + ".txt";
}

int mergeG4ScoringDumps(const std::string& directory, int expectedWorkers)
{
  namespace fs = std::filesystem;
  const std::regex pattern(R"((.+)\.worker([0-9]+)\.txt)");
  std::map<std::string, std::vector<fs::path>> filesPerMesh;
  for (auto& entry : fs::directory_iterator(directory)) {
    std::smatch match;
    const auto name = entry.path().filename().string();
    if (entry.is_regular_file() && std::regex_match(name, match, pattern)) {
      filesPerMesh[match[1]].push_back(entry.path());
    }
  }

  int merged = 0;
  for (auto& [mesh, files] : filesPerMesh) {
    if (expectedWorkers > 0 && static_cast<int>(files.size()) != expectedWorkers) {
      LOG(error) << "Found " << files.size() << " Geant4 scoring dumps for mesh " << mesh << " but expected " << expectedWorkers;
      return -1;
    }
    std::vector<std::string> meshHeader;
    std::vector<ScorerBlock> total;
    for (auto& file : files) {
      std::vector<std::string> header;
      std::vector<ScorerBlock> blocks;
      if (!readDump(file.string(), header, blocks)) {
        LOG(error) << "Cannot read Geant4 scoring dump " << file;
        return -1;
      }
      if (total.empty()) {
        meshHeader = header;
        total = std::move(blocks);
        continue;
      }
      if (blocks.size() != total.size()) {
        LOG(error) << "Geant4 scoring dump " << file << " has a different set of scorers";
        return -1;
      }
      for (size_t b = 0; b < blocks.size(); ++b) {
        if (blocks[b].header != total[b].header || blocks[b].keys != total[b].keys) {
          LOG(error) << "Geant4 scoring dump " << file << " does not match the mesh layout of the other workers";
          return -1;
        }
        for (size_t i = 0; i < blocks[b].keys.size(); ++i) {
          total[b].sum[i] += blocks[b].sum[i];
          total[b].sum2[i] += blocks[b].sum2[i];
          total[b].entries[i] += blocks[b].entries[i];
        }
      }
    }

    const auto outName = (fs::path(directory) / (mesh + ".txt")).string();
    std::ofstream out(outName);
    out << std::setprecision(16);
    for (auto& line : meshHeader) {
      out << line << "\n";
    }
    for (auto& block : total) {
      for (auto& line : block.header) {
        out << line << "\n";
      }
      for (size_t i = 0; i < block.keys.size(); ++i) {
        out << block.keys[i] << "," << block.sum[i] << "," << block.sum2[i] << "," << block.entries[i] << "\n";
      }
    }
    LOG(info) << "Merged " << files.size() << " Geant4 scoring dumps into " << outName;
    ++merged;
  }
  return merged;
}

} // namespace o2::conf
