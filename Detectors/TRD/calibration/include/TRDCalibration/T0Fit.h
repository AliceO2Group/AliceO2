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

/// \file   T0Fit.h
/// \brief  Fits the TRD PH distributions to extract the t0 value
/// \author Luisa Bergmann

#ifndef O2_TRD_T0FIT_H
#define O2_TRD_T0FIT_H

#include "DataFormatsTRD/T0FitHistos.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTRD/Constants.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsTRD/CalT0.h"
#include "TRDCalibration/CalibrationParams.h"

#include "Rtypes.h"
#include "TProfile.h"
#include "TF1.h"
#include "Fit/Fitter.h"
#include "TFile.h"
#include "TTree.h"

#include <array>
#include <cstdlib>
#include <memory>

namespace o2
{
namespace trd
{
//______________________________________________________________________________________________
struct ErfLandauChi2Functor {
  double operator()(const double* par) const;
  std::vector<float> x; ///< x-value (time-bin) of adc profile
  std::vector<float> y; ///< y-value (av. adc) for corresp. time-bin
  float lowerBoundFit;  ///< lower bound for fit
  float upperBoundFit;  ///< upper bound for fit
};

//______________________________________________________________________________________________
class T0Fit final : public o2::calibration::TimeSlotCalibration<o2::trd::T0FitHistos>
{
  using Slot = o2::calibration::TimeSlot<o2::trd::T0FitHistos>;

 public:
  T0Fit() = default;
  ~T0Fit() final = default;

  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->getNEntries() >= mParams.minEntriesTotalT0Fit; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tStart, TFType tEnd) final;

  /// (Re-)Creates a file "trd_t0fit.root". This lets continually fill
  /// a tree with the fit results.
  void createOutputFile();

  /// Close the output file. E.g. First write the tree to the file and let the
  /// smart pointers take care of closing the file.
  void closeOutputFile();

  const std::vector<o2::trd::CalT0>& getCcdbObjectVector() const { return mObjectVector; }
  std::vector<o2::ccdb::CcdbObjectInfo>& getCcdbObjectInfoVector() { return mInfoVector; }

  void initProcessing();

 private:
  bool mInitDone{false};                                         ///< flag to avoid creating output etc multiple times
  const TRDCalibParams& mParams{TRDCalibParams::Instance()};     ///< reference to calibration parameters
  bool mEnableOutput{false};                                     ///< enable output in a root file instead of the ccdb
  std::unique_ptr<TFile> mOutFile{nullptr};                      ///< output file
  std::unique_ptr<TTree> mOutTree{nullptr};                      ///< output tree
  ErfLandauChi2Functor mFitFunctor;                              ///< used for minimization process, provides chi2 estimate
  ROOT::Fit::Fitter mFitter;                                     ///< instance of the ROOT fitter
  std::array<double, 4> mParamsStart;                            ///< Starting parameters for fit
  std::unique_ptr<TF1> mFuncErfLandau;                           ///< helper function to calculate the t0 value after the fitting procedure
  float mDummyT0{-5};                                            ///< dummy value for t0, to be used if fit fails or not enough statistics
  std::array<float, o2::trd::constants::MAXCHAMBER> t0_chambers; ///< t0 values of the individual chambers
  float t0_average{-5};                                          ///< average t0 value across all chambers

  std::vector<o2::ccdb::CcdbObjectInfo> mInfoVector; ///< vector of CCDB infos; each element is filled with CCDB description of accompanying CCDB calibration object
  std::vector<o2::trd::CalT0> mObjectVector;         ///< vector of CCDB calibration objects; the extracted t0 per chamber and average for given slot

  std::unique_ptr<TProfile> adcProfIncl;                                            ///< profile that holds inclusive PH spectrum
  std::array<std::unique_ptr<TProfile>, o2::trd::constants::MAXCHAMBER> adcProfDet; ///< array of profiles for PH spectrum of each chamber

  ClassDefNV(T0Fit, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_T0FIT_H
