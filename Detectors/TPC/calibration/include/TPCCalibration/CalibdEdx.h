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

/// \file CalibdEdx.h
/// \brief This file provides the time dependent dE/dx calibrator, based on the MIP position.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBDEDX_H_
#define ALICEO2_TPC_CALIBDEDX_H_

#include <array>
#include <cstdint>
#include <string_view>

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "TPCCalibration/dEdxHistos.h"
#include "TPCCalibration/FastHisto.h"
#include "CommonUtils/TreeStreamRedirector.h"

namespace o2::tpc
{

/// Container used to store the dE/dx calibration output
struct CalibMIP {
  double ASide;
  double CSide;
};

/// dE/dx calibrator class
class CalibdEdx final : public o2::calibration::TimeSlotCalibration<o2::tpc::TrackTPC, o2::tpc::dEdxHistos>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<dEdxHistos>;
  using CcdbObjectInfoVector = std::vector<o2::ccdb::CcdbObjectInfo>;
  using MIPVector = std::vector<CalibMIP>;

 public:
  /// Contructor that enables track cuts
  CalibdEdx(int nBins = 200, int minEntries = 100, float minP = 0.4, float maxP = 0.6, int minClusters = 60)
    : mNBins{nBins}, mMinEntries{minEntries}, mCuts{minP, maxP, static_cast<float>(minClusters)}
  {
  }

  /// Destructor
  ~CalibdEdx() final = default;

  /// \return if there are enough data to compute the calibration
  bool hasEnoughData(const Slot& slot) const final
  {
    return slot.getContainer()->getASideEntries() >= mMinEntries && slot.getContainer()->getCSideEntries() >= mMinEntries;
  }

  /// Empty the output vectors
  void initOutput() final;

  /// Process time slot data and compute its calibration
  void finalizeSlot(Slot& slot) final;

  /// Creates new time slot
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  void setApplyCuts(bool apply) { mApplyCuts = apply; }
  bool getApplyCuts() { return mApplyCuts; }
  void setCuts(const TrackCuts& cuts) { mCuts = cuts; }

  /// \return the computed calibrations
  const MIPVector& getMIPVector() const { return mMIPVector; }

  /// \return CCDB output informations
  const CcdbObjectInfoVector& getInfoVector() const { return mInfoVector; }

  /// Non const version
  /// \return CCDB output informations
  CcdbObjectInfoVector& getInfoVector() { return mInfoVector; }

  /// Enable debug output to file of the time slots calibrations outputs and dE/dx histograms
  void enableDebugOutput(std::string_view fileName);

  /// Disable debug output to file. Also writes and closes stored time slots.
  void disableDebugOutput();

  /// \return if debug output is enabled
  bool hasDebugOutput() const { return static_cast<bool>(mDebugOutputStreamer); }

  /// Write debug output to file
  void finalizeDebugOutput() const;

 private:
  int mNBins{};          ///< Number of bins in each time slot histogram
  int mMinEntries{};     ///< Minimum amount of tracks in each time slot, to get enough statics
  bool mApplyCuts{true}; ///< Flag to enable tracks cuts
  TrackCuts mCuts;       ///< Cut object

  CcdbObjectInfoVector mInfoVector; ///< vector of CCDB Infos, each element is filled with the CCDB description of the accompanying MIP positions
  MIPVector mMIPVector;             ///< vector of MIP positions, each element is filled in "process" when we finalize one slot (multiple can be finalized during the same "process", which is why we have a vector. Each element is to be considered the output of the device, and will go to the CCDB

  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugOutputStreamer; ///< Debug output streamer

  ClassDefOverride(CalibdEdx, 1);
};

} // namespace o2::tpc
#endif
