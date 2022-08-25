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
#include "EMCALCalibration/EMCALCalibExtractor.h"
#include "EMCALCalibration/EMCALCalibParams.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALBase/Geometry.h"
#include "CCDB/CcdbObjectInfo.h"
#include "EMCALCalib/CalibDB.h"

#include "Framework/Logger.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/format.hpp>

#include <array>
#include <boost/histogram.hpp>
#include <fstream>

using boostHisto2d = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>, boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>>, boost::histogram::unlimited_storage<std::allocator<char>>>;

namespace o2
{
namespace emcal
{
/// \brief class used for managment of bad channel and time calibration
/// template DataInput can be ChannelData or TimeData   // o2::emcal::EMCALChannelData, o2::emcal::EMCALTimeCalibData
template <typename DataInput, typename DataOutput>
class EMCALChannelCalibrator : public o2::calibration::TimeSlotCalibration<o2::emcal::Cell, DataInput>
{
  using TFType = o2::calibration::TFType;
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

  ///\brief Set the testing status.
  void setIsTest(bool isTest) { mTest = isTest; }
  bool isTest() const { return mTest; }

  const CcdbObjectInfoVector& getInfoVector() const { return mInfoVector; }
  const std::vector<DataOutput>& getOutputVector() const { return mCalibObjectVector; }

  /// \brief Configure the calibrator
  std::shared_ptr<EMCALCalibExtractor> getCalibExtractor() { return mCalibrator; } // return shared pointer!
  /// \brief setter for mCalibrator
  void SetCalibExtractor(std::shared_ptr<EMCALCalibExtractor> extr) { mCalibrator = extr; };

 private:
  int mNBins = 0;     ///< bins of the histogram for passing
  float mRange = 0.;  ///< range of the histogram for passing
  bool mTest = false; ///< flag to be used when running in test mode: it simplify the processing (e.g. does not go through all channels)
  std::shared_ptr<EMCALCalibExtractor> mCalibrator;

  // output
  CcdbObjectInfoVector mInfoVector; // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying TimeSlewing object
  std::vector<DataOutput> mCalibObjectVector;

  ClassDefOverride(EMCALChannelCalibrator, 1);
};

//_____________________________________________
template <typename DataInput, typename DataOutput>
void EMCALChannelCalibrator<DataInput, DataOutput>::initOutput()
{
  mInfoVector.clear();
  mCalibObjectVector.clear();
  // mNEvents = 0;
  return;
}

//_____________________________________________
template <typename DataInput, typename DataOutput>
bool EMCALChannelCalibrator<DataInput, DataOutput>::hasEnoughData(const o2::calibration::TimeSlot<DataInput>& slot) const
{
  const DataInput* c = slot.getContainer();
  return (mTest ? true : c->hasEnoughData());
}

//_____________________________________________
template <typename DataInput, typename DataOutput>
void EMCALChannelCalibrator<DataInput, DataOutput>::finalizeSlot(o2::calibration::TimeSlot<DataInput>& slot)
{

  // Extract results for the single slot
  DataInput* c = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  std::map<std::string, std::string> md;
  if constexpr (std::is_same<DataInput, o2::emcal::EMCALChannelData>::value) {
    LOG(debug) << "Launching the calibration.";
    auto bcm = mCalibrator->calibrateBadChannels(c->getHisto());
    LOG(debug) << "Done with the calibraiton";
    // for the CCDB entry
    auto clName = o2::utils::MemFileHelper::getClassName(bcm);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
    mInfoVector.emplace_back(CalibDB::getCDBPathBadChannelMap(), clName, flName, md, slot.getStartTimeMS(), o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
    mCalibObjectVector.push_back(bcm);

    if ((EMCALCalibParams::Instance().localRootFilePath).find(".root") != std::string::npos) {
      std::ifstream ffile(EMCALCalibParams::Instance().localRootFilePath.c_str());

      TFile fLocalStorage((EMCALCalibParams::Instance().localRootFilePath).c_str(), ffile.good() == true ? "update" : "recreate");
      fLocalStorage.cd();
      TH2F* histBCMap = (TH2F*)bcm.getHistogramRepresentation();
      std::string nameBCHist = "BadChannels_" + std::to_string(slot.getStartTimeMS());
      histBCMap->Write(nameBCHist.c_str(), TObject::kOverwrite);

      TH2F hCalibHist = o2::utils::TH2FFromBoost(c->getHisto());
      std::string nameBCInputHist = "EnergyVsCellID_" + std::to_string(slot.getStartTimeMS());
      hCalibHist.Write(nameBCInputHist.c_str(), TObject::kOverwrite);
      fLocalStorage.Close();
    }
  } else if constexpr (std::is_same<DataInput, o2::emcal::EMCALTimeCalibData>::value) {
    auto tcd = mCalibrator->calibrateTime(c->getHisto(), EMCALCalibParams::Instance().minTimeForFit_tc, EMCALCalibParams::Instance().maxTimeForFit_tc, EMCALCalibParams::Instance().restrictFitRangeToMax_tc);

    // for the CCDB entry
    auto clName = o2::utils::MemFileHelper::getClassName(slot);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);

    //prepareCCDBobjectInfo
    mInfoVector.emplace_back(CalibDB::getCDBPathTimeCalibrationParams(), clName, flName, md, slot.getStartTimeMS(), o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
    mCalibObjectVector.push_back(tcd);

    if ((EMCALCalibParams::Instance().localRootFilePath).find(".root") != std::string::npos) {
      std::ifstream ffile(EMCALCalibParams::Instance().localRootFilePath.c_str());

      TFile fLocalStorage((EMCALCalibParams::Instance().localRootFilePath).c_str(), ffile.good() == true ? "update" : "recreate");
      fLocalStorage.cd();
      TH1F* histTCparams = (TH1F*)tcd.getHistogramRepresentation(false);
      std::string nameTCHist = "TCParams_" + std::to_string(slot.getStartTimeMS());
      histTCparams->Write(nameTCHist.c_str(), TObject::kOverwrite);

      TH2F hCalibHist = o2::utils::TH2FFromBoost(c->getHisto());
      std::string nameTCInputHist = "TimeVsCellID_" + std::to_string(slot.getStartTimeMS());
      hCalibHist.Write(nameTCInputHist.c_str(), TObject::kOverwrite);
      fLocalStorage.Close();
    }
  }
}

template <typename DataInput, typename DataOutput>
o2::calibration::TimeSlot<DataInput>& EMCALChannelCalibrator<DataInput, DataOutput>::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = o2::calibration::TimeSlotCalibration<o2::emcal::Cell, DataInput>::getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<DataInput>());
  return slot;
}

} // end namespace emcal
} // end namespace o2

#endif /*EMCAL_CHANNEL_CALIBRATOR_H_ */
