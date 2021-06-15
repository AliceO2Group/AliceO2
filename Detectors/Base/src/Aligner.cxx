// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsBase/Aligner.h"
#include "CCDB/BasicCCDBManager.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include <chrono>

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
  ccdbmgr.setURL(getCCDB());
  ccdbmgr.setTimestamp(timestamp);
  LOGP(INFO, "applying geometry alignment from {} for timestamp {}", getCCDB(), timestamp);
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (!msk[id]) {
      continue;
    }
    std::string path = o2::base::NameConf::getAlignmentPath({id});
    auto algV = ccdbmgr.get<std::vector<o2::detectors::AlignParam>>(path);
    if (!algV) {
      throw std::runtime_error(fmt::format("Failed to fetch alignment from {}:{}", getCCDB(), path));
    }
    if (!algV->empty()) {
      LOGP(INFO, "applying alignment for {}", DetID::getName(id));
      o2::base::GeometryManager::applyAlignment(*algV);
    } else {
      LOGP(INFO, "skipping empty alignment for {}", DetID::getName(id));
    }
  }
  gGeoManager->RefreshPhysicalNodes(false);
}

long Aligner::getTimeStamp() const
{
  return mTimeStamp > 0 ? mTimeStamp : std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
