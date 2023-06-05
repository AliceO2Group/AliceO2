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

/// @brief meta data of the file produced by O2

#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "Framework/DataTakingContext.h"
#include <Framework/Logger.h>
#include <TMD5.h>
#include <filesystem>
#include <chrono>

using namespace o2::dataformats;

bool FileMetaData::fillFileData(const std::string& fname, bool md5)
{
  try {
    lurl = std::filesystem::canonical(fname).string();
    size = std::filesystem::file_size(lurl);
    if (md5) {
      std::unique_ptr<TMD5> md5ptr{TMD5::FileChecksum(fname.c_str())};
      md5 = md5ptr->AsString();
    }
    ctime = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  } catch (std::exception const& e) {
    LOG(error) << "Failed to fill metadata for file " << fname << ", reason: " << e.what();
    return false;
  }
  return true;
}

std::string FileMetaData::asString() const
{
  std::string ms;

  // obligatory part
  ms += fmt::format("LHCPeriod: {}\n", LHCPeriod);
  ms += fmt::format("run: {}\n", run);
  ms += fmt::format("lurl: {}\n", lurl);

  // optional part
  if (!type.empty()) {
    ms += fmt::format("type: {}\n", type);
  }
  if (ctime) {
    ms += fmt::format("ctime: {}\n", ctime);
  }
  if (size) {
    ms += fmt::format("size: {}\n", size);
  }
  if (!guid.empty()) {
    ms += fmt::format("guid: {}\n", guid);
  }
  if (!surl.empty()) {
    ms += fmt::format("surl: {}\n", surl);
  }
  if (!curl.empty()) {
    ms += fmt::format("curl: {}\n", curl);
  }
  if (!md5.empty()) {
    ms += fmt::format("md5: {}\n", md5);
  }
  if (!xxhash.empty()) {
    ms += fmt::format("xxhash: {}\n", xxhash);
  }
  if (!seName.empty()) {
    ms += fmt::format("seName: {}\n", seName);
  }
  if (!seioDaemons.empty()) {
    ms += fmt::format("seioDaemons: {}\n", seioDaemons);
  }
  if (!priority.empty()) {
    ms += fmt::format("priority: {}\n", priority);
  }
  if (persistent) {
    ms += fmt::format("persistent: {}\n", persistent);
  }
  if (!detComposition.empty()) {
    ms += fmt::format("det_composition: {}\n", detComposition);
  }
  if (!tfOrbits.empty()) {
    ms += fmt::format("TFOrbits: {}", tfOrbits[0]);
    for (size_t i = 1; i < tfOrbits.size(); i++) {
      ms += fmt::format(",{}", tfOrbits[i]);
    }
    ms += "\n";
  }

  return ms;
}

void FileMetaData::setDataTakingContext(const o2::framework::DataTakingContext& dtc)
{
  LHCPeriod = dtc.lhcPeriod;
  detComposition = dtc.detectors;
  run = dtc.runNumber;
}

std::ostream& o2::dataformats::operator<<(std::ostream& stream, const FileMetaData& h)
{
  stream << h.asString();
  return stream;
}
