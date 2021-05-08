// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CollisionTimeRecoTask.h
/// \brief Definition of the FIT collision time reconstruction task
#ifndef ALICEO2_FIT_COLLISIONTIMERECOTASK_H
#define ALICEO2_FIT_COLLISIONTIMERECOTASK_H

#include <vector>
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FITCalibration/FITCalibrationApi.h"
#include "CCDB/BasicCCDBManager.h"
#include <gsl/span>
#include <bitset>

namespace o2
{
namespace ft0
{
class CollisionTimeRecoTask
{
  using offsetCalib = o2::ft0::FT0ChannelTimeCalibrationObject;
  using CalibApi = o2::fit::FITCalibrationApi;

 public:
  enum : int { TimeMean,
               TimeA,
               TimeC,
               Vertex };
  CollisionTimeRecoTask() = default;
  ~CollisionTimeRecoTask() = default;
  o2::ft0::RecPoints process(o2::ft0::Digit const& bcd,
                             gsl::span<const o2::ft0::ChannelData> inChData,
                             gsl::span<o2::ft0::ChannelDataFloat> outChData);
  void FinishTask();
  o2::ft0::FT0ChannelTimeCalibrationObject calibrateTimeOffset(uint32_t orbit, uint16_t bc)
  {
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL("http://ccdb-test.cern.ch:8080");
    //  getProcessingTimestamp()
    //    double calib = mCalibApi->getTimeCalibration(dig->getChannel(), dig->getTOT() * Geo::TOTBIN_NS);
    auto timeStamp = o2::InteractionRecord::bc2ns(bc, orbit);
    //   mgr.setTimestamp(timeStamp);
    auto caliboffsets = mgr.get<o2::ft0::FT0ChannelTimeCalibrationObject>("FT0/Calibration/ChannelTimeOffset");
    return *caliboffsets;
  }
  void setCalibApi(CalibApi* calibApi)
  {
    mCalibApi = calibApi;
  }

 private:
  CalibApi* mCalibApi = nullptr; //! calib api to handle the FIT calibration

  ClassDefNV(CollisionTimeRecoTask, 1);
};
} // namespace ft0
} // namespace o2
#endif
