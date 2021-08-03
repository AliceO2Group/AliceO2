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

/// \class EMCALChannelCalibrator
/// \brief  Perform the EMCAL bad channel calibration
/// \author Hannah Bossi, Yale University
/// \ingroup EMCALCalib
/// \since Feb 11, 2021

#ifndef EMCAL_CHANNEL_CALIBRATOR_H_
#define EMCAL_CHANNEL_CALIBRATOR_H_

#include "EMCALCalibration/EMCALTimeCalibData.h"
#include "EMCALCalibration/EMCALChannelData.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALBase/Geometry.h"
#include "CCDB/CcdbObjectInfo.h"

#include "Framework/Logger.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/format.hpp>

#include <array>
#include <boost/histogram.hpp>

namespace o2
{
namespace emcal
{
/// \brief class used for managment of bad channel and time calibration
/// template DataInput can be ChannelData or TimeData   // o2::emcal::EMCALChannelData, o2::emcal::EMCALTimeCalibData
/// template HistContainer can be ChannelCalibInitParams or TimeCalibInitParams
template <typename DataInput, typename HistContainer>
class EMCALChannelCalibrator : public o2::calibration::TimeSlotCalibration<o2::emcal::Cell, DataInput>
{
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<DataInput>;
  using Cell = o2::emcal::Cell;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;

 public:
  EMCALChannelCalibrator(int nb = 1000, float r = 0.35) : mNBins(nb), mRange(r){};

  ~EMCALChannelCalibrator() final = default;

  /// \brief Checking if all channels have enough data to do calibration.
  bool hasEnoughData(const Slot& slot) const final;
  /// \brief Initialize the vector of our output objects.
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  o2::calibration::TimeSlot<DataInput>& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  void setIsTest(bool isTest) { mTest = isTest; }
  bool isTest() const { return mTest; }

 private:
  int mNBins = 0;     ///< bins of the histogram for passing
  float mRange = 0.;  ///< range of the histogram for passing
  bool mTest = false; ///< flag to be used when running in test mode: it simplify the processing (e.g. does not go through all channels)

  // output
  CcdbObjectInfoVector mInfoVector; // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying TimeSlewing object

  ClassDefOverride(EMCALChannelCalibrator, 1);
};

//_____________________________________________
template <typename DataInput, typename HistContainer>
void EMCALChannelCalibrator<DataInput, HistContainer>::initOutput()
{
  mInfoVector.clear();
  return;
}

//_____________________________________________
template <typename DataInput, typename HistContainer>
bool EMCALChannelCalibrator<DataInput, HistContainer>::hasEnoughData(const o2::calibration::TimeSlot<DataInput>& slot) const
{

  const DataInput* c = slot.getContainer();
  LOG(INFO) << "Checking statistics";
  return (mTest ? true : c->hasEnoughData());
}

//_____________________________________________
template <typename DataInput, typename HistContainer>
void EMCALChannelCalibrator<DataInput, HistContainer>::finalizeSlot(o2::calibration::TimeSlot<DataInput>& slot)
{
  // Extract results for the single slot
  DataInput* c = slot.getContainer();
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // for the CCDB entry
  std::map<std::string, std::string> md;

  //auto clName = o2::utils::MemFileHelper::getClassName(tm);
  //auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfoVector.emplace_back("EMCAL/ChannelCalib", "clname", "flname", md, slot.getTFStart(), 99999999999999);
}

template <typename DataInput, typename HistContainer>
o2::calibration::TimeSlot<DataInput>& EMCALChannelCalibrator<DataInput, HistContainer>::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = o2::calibration::TimeSlotCalibration<o2::emcal::Cell, DataInput>::getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  HistContainer histcont; // initialize struct with (default) ranges for time or channel calibration.
  slot.setContainer(std::make_unique<DataInput>(histcont));
  return slot;
}

} // end namespace emcal
} // end namespace o2

#endif /*EMCAL_CHANNEL_CALIBRATOR_H_ */
