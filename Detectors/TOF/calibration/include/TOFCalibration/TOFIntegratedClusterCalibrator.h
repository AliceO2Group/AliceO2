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

/// \file TOFIntegratedClusterCalibrator.h
/// \brief calibrator class for accumulating integrated clusters
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 21, 2023

#ifndef TOF_INTEGRATEDCLUSTERCALIBRATOR_H_
#define TOF_INTEGRATEDCLUSTERCALIBRATOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"

namespace o2
{
namespace tof
{

/// struct containing the integrated TOF currents
struct ITOFC {
  std::vector<float> mITOFCNCl; ///< integrated 1D TOF cluster currents
  std::vector<float> mITOFCQ;   ///< integrated 1D TOF qTot currents
  ClassDefNV(ITOFC, 1);
};

/// class for accumulating integrated TOF currents
class TOFIntegratedClusters
{
 public:
  /// \constructor
  /// \param tFirst first TF of the stored currents
  /// \param tLast last TF of the stored currents
  TOFIntegratedClusters(o2::calibration::TFType tFirst, o2::calibration::TFType tLast) : mTFFirst{tFirst}, mTFLast{tLast} {};

  /// \default constructor for ROOT I/O
  TOFIntegratedClusters() = default;

  /// print summary informations
  void print() const { LOGP(info, "TF Range from {} to {} with {} of remaining data", mTFFirst, mTFLast, mRemainingData); }

  /// accumulate currents for given TF
  /// \param tfID TF ID of incoming data
  /// \param iTOFCNcl integrated TOF currents for one TF for number of clusters
  /// \param iTOFCqTot integrated TOF currents for one TF for qTot
  void fill(const o2::calibration::TFType tfID, const std::vector<float>& iTOFCNcl, const std::vector<float>& iTOFCqTot);

  /// merging TOF currents with previous interval
  void merge(const TOFIntegratedClusters* prev);

  /// \return returns if already all expected TFs are received
  bool hasEnoughData() const { return mRemainingData ? false : true; }

  /// \return returns accumulated TOF currents
  const auto& getITOFCCurrents() const { return mCurrents; }

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "TOFIntegratedClusters.root", const char* outName = "ITOFC") const;

  /// dump object to TTree for visualisation
  /// \param outFileName name of the output file
  void dumpToTree(const char* outFileName = "ITOFCTree.root");

 private:
  ITOFC mCurrents;                          ///< buffer for integrated currents
  o2::calibration::TFType mTFFirst{};       ///< first TF of currents
  o2::calibration::TFType mTFLast{};        ///< last TF of currents
  o2::calibration::TFType mRemainingData{}; ///< counter for received data
  unsigned int mNValuesPerTF{};             ///< number of expected currents per TF (estimated from first received data)
  bool mInitialize{true};                   ///< flag if this object will be initialized when fill method is called

  /// init member when first data is received
  /// \param vec received data which is used to estimate the expected data
  void initData(const std::vector<float>& vec);

  ClassDefNV(TOFIntegratedClusters, 1);
};

class TOFIntegratedClusterCalibrator : public o2::calibration::TimeSlotCalibration<o2::tof::TOFIntegratedClusters>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::tof::TOFIntegratedClusters>;
  using CalibVector = std::vector<ITOFC>;
  using TFinterval = std::vector<std::pair<TFType, TFType>>;
  using TimeInterval = std::vector<std::pair<long, long>>;

 public:
  /// default constructor
  TOFIntegratedClusterCalibrator() = default;

  /// default destructor
  ~TOFIntegratedClusterCalibrator() final = default;

  /// check if given slot has already enough data
  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->hasEnoughData(); }

  /// clearing all calibration objects in the output buffer
  void initOutput() final;

  /// storing the integrated currents for given slot
  void finalizeSlot(Slot& slot) final;

  /// Creates new time slot
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  /// \return CCDB output informations
  const TFinterval& getTFinterval() const { return mIntervals; }

  /// \return Time frame time information
  const TimeInterval& getTimeIntervals() const { return mTimeIntervals; }

  /// \return returns calibration objects (pad-by-pad gain maps)
  auto getCalibs() && { return std::move(mCalibs); }

  /// check if calibration data is available
  bool hasCalibrationData() const { return mCalibs.size() > 0; }

  /// set if debug objects will be created
  void setDebug(const bool debug) { mDebug = debug; }

 private:
  TFinterval mIntervals;       ///< start and end time frames of each calibration time slots
  TimeInterval mTimeIntervals; ///< start and end times of each calibration time slots
  CalibVector mCalibs;         ///< Calibration object containing for each pad a histogram with normalized charge
  bool mDebug{false};          ///< write debug output objects

  ClassDefOverride(TOFIntegratedClusterCalibrator, 1);
};

} // end namespace tof
} // end namespace o2

#endif /* TOF_CHANNEL_CALIBRATOR_H_ */
