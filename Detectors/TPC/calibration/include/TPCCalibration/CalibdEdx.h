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
/// \brief This file provides the container used for time based residual dE/dx calibration.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBDEDX_H_
#define ALICEO2_TPC_CALIBDEDX_H_

#include <cstddef>
#include <gsl/span>
#include <string_view>

// o2 includes
#include "CommonDataFormat/TFIDInfo.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/CalibdEdxCorrection.h"
#include "DetectorsBase/Propagator.h"
#include "CommonUtils/TreeStreamRedirector.h"

// boost includes
#include <boost/histogram.hpp>
#include "THn.h"

namespace o2::tpc
{

/// Class that creates dE/dx histograms from a sequence of tracks objects
class CalibdEdx
{
 public:
  enum Axis {
    dEdx,
    Tgl,
    Snp,
    Sector,
    Stack,
    Charge,
    Size ///< Number of axes
  };

  // Interger histogram axis identifying the GEM stacks, without under and overflow bins.
  using IntAxis = boost::histogram::axis::integer<int, boost::histogram::use_default, boost::histogram::axis::option::none_t>;
  // Float axis to store data, without under and overflow bins.
  using FloatAxis = boost::histogram::axis::regular<float, boost::histogram::use_default, boost::histogram::use_default, boost::histogram::axis::option::none_t>;

  using TFIDInfo = o2::dataformats::TFIDInfo;

  // Histogram axes types
  using AxesType = std::tuple<
    FloatAxis, // dEdx
    FloatAxis, // Tgl
    FloatAxis, // Snp
    IntAxis,   // sector
    IntAxis,   // stackType
    IntAxis    // Charge
    >;

  using Hist = boost::histogram::histogram<AxesType>;

  /// copy ctor
  CalibdEdx(const CalibdEdx& other);

  /// \param angularBins number of bins for Tgl and Snp
  /// \param fitSnp enable Snp correction
  CalibdEdx(int dEdxBins = 60, float mindEdx = 20, float maxdEdx = 90, int angularBins = 36, bool fitSnp = false);

  void setCuts(const TrackCuts& cuts) { mCuts = cuts; }
  void setApplyCuts(bool apply) { mApplyCuts = apply; }
  bool getApplyCuts() const { return mApplyCuts; }

  /// \param minEntries per GEM stack to enable sector by sector correction. Below this value we only perform one fit per ROC type (IROC, OROC1, ...; no side nor sector information).
  void setSectorFitThreshold(int minEntries) { mSectorThreshold = minEntries; }
  /// \param minEntries per GEM stack to enable Tgl fit
  void set1DFitThreshold(int minEntries) { m1DThreshold = minEntries; }
  /// \param minEntries per GEM stack to enable Tgl and Snp fit
  /// has no effect if fitSnp = false
  void set2DFitThreshold(int minEntries) { m2DThreshold = minEntries; }

  int getSectorFitThreshold() const { return mSectorThreshold; }
  int getTglFitThreshold() const { return m1DThreshold; }
  int getSnpFitThreshold() const { return m2DThreshold; }

  /// \brief Params used to remove electron points from the fit.
  /// The fit to find the MIP peak will be performed \p passes times, from the second time
  /// and afterwords any points with dEdx values above the previous fit * (1 + \p cut) and blow
  /// previous fit * (1 - \p lowCutFactor * \p cut) will be cut out.
  /// \note you can set \p passes = 0 to disable this functionality
  void setElectronCut(float cut, int passes = 3, float lowCutFactor = 1.5)
  {
    mFitCut = cut;
    mFitPasses = passes;
    mFitLowCutFactor = lowCutFactor;
  }

  /// setting the material type for track propagation
  void setMaterialType(o2::base::Propagator::MatCorrType materialType) { mMatType = materialType; }

  /// Fill histograms using tracks data.
  void fill(const TrackTPC& tracks);
  void fill(const gsl::span<const TrackTPC>);
  void fill(const std::vector<TrackTPC>& tracks) { fill(gsl::span(tracks)); }

  void fill(const TFIDInfo& tfid, const gsl::span<const TrackTPC> tracks)
  {
    mTFID = tfid;
    fill(tracks);
  }
  void fill(const TFIDInfo& tfid, const std::vector<TrackTPC>& tracks) { fill(tfid, gsl::span(tracks)); }

  /// Add counts from another container.
  void merge(const CalibdEdx* other);

  /// Compute MIP position from dEdx histograms and save result in the correction container.
  /// To retrieve the correction call `CalibdEdx::getCalib()`
  void finalize();

  /// Return calib data histogram
  const Hist& getHist() const { return mHist; }
  /// Return calib data as a THn
  THnF* getRootHist() const;

  const CalibdEdxCorrection& getCalib() const { return mCalib; }

  /// calibration used during reconstruction
  void setCalibrationInput(const CalibdEdxCorrection& calib) { mCalibIn = calib; }
  const CalibdEdxCorrection& getCalibrationInput() const { return mCalibIn; }

  /// Return the number of hist entries of the gem stack with less statistics
  int minStackEntries() const;

  /// \brief Check if there are enough data to compute the calibration.
  /// \return false if any of the GEM stacks has less entries than minEntries
  bool hasEnoughData(float minEntries) const;

  /// Find the sector of a track at a given GEM stack type.
  static int findTrackSector(const TrackTPC& track, GEMstack, bool& ok);

  /// Print statistics info.
  void print() const;

  /// Save the histograms to a TTree.
  void writeTTree(std::string_view fileName) const;

  /// Enable debug output to file of the time slots calibrations outputs and dE/dx histograms
  void enableDebugOutput(std::string_view fileName);

  /// Disable debug output to file. Also writes and closes stored time slots.
  void disableDebugOutput();

  /// Write debug output to file
  void finalizeDebugOutput() const;

  /// \return if debug output is enabled
  bool hasDebugOutput() const { return static_cast<bool>(mDebugOutputStreamer); }

  constexpr static float MipScale = 1.0 / 50.0; ///< Inverse of target dE/dx value for MIPs

  constexpr static float scaleTgl(float tgl, GEMstack rocType) { return tgl / conf_dedx_corr::TglScale[rocType]; }
  constexpr static float recoverTgl(float scaledTgl, GEMstack rocType) { return scaledTgl * conf_dedx_corr::TglScale[rocType]; }

 private:
  // ATTENTION: Adjust copy constructor
  bool mFitSnp{};
  bool mApplyCuts{true};         ///< Wether or not to apply tracks cuts
  TrackCuts mCuts{0.3, 0.7, 60}; ///< MIP
  int mSectorThreshold = 1000;   ///< Minimum entries per stack to perform a sector by sector fit.
  int m1DThreshold = 500;        ///< Minimum entries per stack to perform a Tgl fit
  int m2DThreshold = 5000;       ///< Minimum entries per stack to perform a Snp fit
  float mFitCut = 0.2;           ///< dEdx cut value used to remove electron tracks
  float mFitLowCutFactor = 1.5;  ///< dEdx cut multiplier for the lower dE/dx range
  int mFitPasses = 3;            ///< number of fit passes used to remove electron tracks
  TFIDInfo mTFID{};              ///< current TFID

  Hist mHist;                     ///< dEdx multidimensional histogram
  CalibdEdxCorrection mCalib{};   ///< Calibration output
  CalibdEdxCorrection mCalibIn{}; ///< Calibration output

  o2::base::Propagator::MatCorrType mMatType{}; ///< material type for track propagation

  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugOutputStreamer; ///< Debug output streamer
  ClassDefNV(CalibdEdx, 4);
};

} // namespace o2::tpc

#endif
