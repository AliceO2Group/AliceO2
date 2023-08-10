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
#include <chrono>
#include <TGeoParallelWorld.h>

O2ParamImpl(o2::base::Aligner);

using namespace o2::base;

using DetID = o2::detectors::DetID;

DetID::mask_t Aligner::getDetectorsMask() const
{
  return DetID::getMask(mDetectors) & (~DetID::getMask(DetID::CTP));
}

void Aligner::applyAlignment(long timestamp, DetID::mask_t addMask) const
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
  TGeoParallelWorld* pw = gGeoManager->CreateParallelWorld("priority_its_sensor");
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (!msk[id] || (detGeoMask.any() && !detGeoMask[id])) {
      continue;
    }
    std::string path = o2::base::DetectorNameConf::getAlignmentPath({id});
    auto algV = ccdbmgr.get<std::vector<o2::detectors::AlignParam>>(path);
    if (!algV) {
      throw std::runtime_error(fmt::format("Failed to fetch alignment from {}:{}", o2::base::NameConf::getCCDBServer(), path));
    }
    if (!algV->empty()) {
      done.set(id);
      o2::base::GeometryManager::applyAlignment(*algV);
    } else {
      skipped.set(id);
    }
  }
  // pw->AddOverlap(gGeoManager->GetVolume("ITSUWrapVol0"));
  pw->CloseGeometry();
  gGeoManager->SetUseParallelWorldNav(true);
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

long Aligner::getTimeStamp() const
{
  return mTimeStamp > 0 ? mTimeStamp : std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
