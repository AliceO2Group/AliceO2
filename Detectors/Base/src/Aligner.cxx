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

#include "DetectorsBase/Aligner.h"
#include "CCDB/BasicCCDBManager.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "CommonUtils/StringUtils.h"
#include <chrono>

O2ParamImpl(o2::base::Aligner);

using namespace o2::base;

using DetID = o2::detectors::DetID;

DetID::mask_t Aligner::getDetectorsMask() const
{
  return DetID::getMask(mDetectors) & (~DetID::getMask(DetID::CTP));
}

void Aligner::applyAlignment(long timestamp, DetID::mask_t addMask, bool keepPhysicalNodes) const
{
  DetID::mask_t msk = getDetectorsMask() & addMask;
  if (msk.none()) {
    return;
  }
  if (!gGeoManager) {
    throw std::runtime_error("Geometry is not loaded, cannot apply alignment");
  }
  if (gGeoManager->IsLocked()) {
    throw std::runtime_error("Geometry is locked, cannot apply alignment");
  }

  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  if (timestamp == 0) {
    timestamp = getTimeStamp();
  }
  ccdbmgr.setURL(o2::base::NameConf::getCCDBServer());
  ccdbmgr.setTimestamp(timestamp);
  DetID::mask_t done, skipped;
  DetID::mask_t detGeoMask(gGeoManager->GetUniqueID());
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (!msk[id] || (detGeoMask.any() && !detGeoMask[id])) {
      continue;
    }
    std::string path = o2::base::DetectorNameConf::getAlignmentPath({id});
    std::vector<o2::detectors::AlignParam>* algV = nullptr;
    auto altPath = getAlignmentPath({id});
    if (altPath != path) {
      // an alternative object may not exist for every timestamp: fall back to the standard one
      bool fatalWhenNull = ccdbmgr.getFatalWhenNull();
      ccdbmgr.setFatalWhenNull(false);
      algV = ccdbmgr.get<std::vector<o2::detectors::AlignParam>>(altPath);
      ccdbmgr.setFatalWhenNull(fatalWhenNull);
      if (algV) {
        LOGP(info, "Using alignment {} for {}", altPath, DetID::getName(id));
        path = altPath;
      } else {
        LOGP(warning, "Alignment {} could not be fetched for timestamp {}, using {} for {}", altPath, timestamp, path, DetID::getName(id));
      }
    }
    if (!algV) {
      algV = ccdbmgr.get<std::vector<o2::detectors::AlignParam>>(path);
    }
    if (!algV) {
      throw std::runtime_error(fmt::format("Failed to fetch alignment from {}:{}", o2::base::NameConf::getCCDBServer(), path));
    }
    if (!algV->empty()) {
      done.set(id);
      o2::base::GeometryManager::applyAlignment(*algV, keepPhysicalNodes);
    } else {
      skipped.set(id);
    }
  }
  std::string log = fmt::format("Alignment from {} for timestamp {}: ", o2::base::NameConf::getCCDBServer(), timestamp);
  if (done.any()) {
    log += fmt::format("applied to [{}]", DetID::getNames(done));
  }
  if (skipped.any()) {
    log += fmt::format(", empty object for [{}]", DetID::getNames(skipped));
  }
  LOG(info) << log;
  gGeoManager->RefreshPhysicalNodes(false);
}

std::string Aligner::getAlignmentPath(DetID id) const
{
  return getAlignmentPath(mAlignPaths, id);
}

std::string Aligner::getAlignmentPath(const std::string& alignPaths, DetID id)
{
  std::string path = o2::base::DetectorNameConf::getAlignmentPath(id);
  DetID::mask_t seen;
  for (const auto& entry : o2::utils::Str::tokenize(alignPaths, ',')) {
    auto sep = entry.find(':');
    auto det = entry.substr(0, sep);
    auto altPath = sep == std::string::npos ? std::string{} : entry.substr(sep + 1);
    o2::utils::Str::trim(det);
    o2::utils::Str::trim(altPath);
    if (det.empty() || altPath.empty()) {
      throw std::runtime_error(fmt::format("Malformed align-geom.mAlignPaths entry \"{}\", expected DET:path", entry));
    }
    int detID = DetID::nameToID(det.c_str());
    if (detID < DetID::First) {
      throw std::runtime_error(fmt::format("Unknown detector {} in align-geom.mAlignPaths", det));
    }
    if (seen[detID]) {
      throw std::runtime_error(fmt::format("Detector {} given more than once in align-geom.mAlignPaths", det));
    }
    seen.set(detID);
    if (detID == id) {
      path = altPath;
    }
  }
  return path;
}

long Aligner::getTimeStamp() const
{
  return mTimeStamp > 0 ? mTimeStamp : std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
