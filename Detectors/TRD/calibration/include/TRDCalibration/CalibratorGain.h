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

/// \file CalibratorGain.h
/// \brief TimeSlot-based calibration of gain
/// \author Gauthier Legras

#ifndef O2_TRD_CALIBRATORGAIN_H
#define O2_TRD_CALIBRATORGAIN_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/GainCalibHistos.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsTRD/CalGain.h"
#include "TRDCalibration/CalibrationParams.h"

#include "Rtypes.h"
#include "TProfile.h"
#include "Fit/Fitter.h"
#include "TFile.h"
#include "TTree.h"
#include <TF1Convolution.h>

#include <array>
#include <cstdlib>
#include <memory>

namespace o2
{
namespace trd
{

class CalibratorGain final : public o2::calibration::TimeSlotCalibration<o2::trd::GainCalibHistos>
{
  using Slot = o2::calibration::TimeSlot<o2::trd::GainCalibHistos>;

 public:
  CalibratorGain() = default;
  ~CalibratorGain() final = default;

  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->getNEntries() >= mMinEntriesTotal; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tStart, TFType tEnd) final;

  /// (Re-)Creates a file "trd_calibgain.root". This lets continually fill
  /// a tree with the fit results and the residuals histograms.
  void createOutputFile();

  /// Close the output file. E.g. First write the tree to the file and let the
  /// smart pointers take care of closing the file.
  void closeOutputFile();

  const std::vector<o2::trd::CalGain>& getCcdbObjectVector() const { return mObjectVector; }
  std::vector<o2::ccdb::CcdbObjectInfo>& getCcdbObjectInfoVector() { return mInfoVector; }

  void initProcessing();

  /// Initialize the fit values once with the previous valid ones if they are
  /// available.
  void retrievePrev(o2::framework::ProcessingContext& pc);

 private:
  bool mInitDone{false};                                               ///< flag to avoid creating the TProfiles multiple times
  const TRDCalibParams& mParams{TRDCalibParams::Instance()};           ///< reference to calibration parameters
  size_t mMinEntriesTotal{mParams.minEntriesTotalGainCalib};           ///< minimum total number of angular deviations
  size_t mMinEntriesChamber{mParams.minEntriesChamberGainCalib};       ///< minimum number of angular deviations per chamber for accepting refitted value
  bool mEnableOutput{false};                                           ///< enable output of calibration fits and tprofiles in a root file instead of the ccdb
  std::unique_ptr<TFile> mOutFile{nullptr};                            ///< output file
  std::unique_ptr<TTree> mOutTree{nullptr};                            ///< output tree
  std::unique_ptr<TF1Convolution> mFconv;                              ///< for fitting convolution of landau*exp and gaussian
  std::unique_ptr<TF1> mFitFunction;                                   ///< fitting function for dEdx distribution
  std::array<float, constants::MAXCHAMBER> mFitResults;                ///< stores most probable value of dEdx from fit
  std::array<std::unique_ptr<TH1F>, constants::MAXCHAMBER> mdEdxhists; ///< stores dEdx hists
  std::vector<o2::ccdb::CcdbObjectInfo> mInfoVector;                   ///< vector of CCDB infos; each element is filled with CCDB description of accompanying CCDB calibration object
  std::vector<o2::trd::CalGain> mObjectVector;                         ///< vector of CCDB calibration objects; the extracted gain per chamber for given slot

  ClassDefOverride(CalibratorGain, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CALIBRATORGAIN_H
