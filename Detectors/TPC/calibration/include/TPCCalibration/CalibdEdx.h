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
/// \brief This file provides the container used for time based dE/dx calibration.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBDEDX_H_
#define ALICEO2_TPC_CALIBDEDX_H_

#include <array>
#include <cstddef>
#include <gsl/span>
#include <string_view>

// o2 includes
#include "TPCCalibration/CalibdEdxDataContainer.h"
#include "TPCCalibration/FastHisto.h"
#include "DataFormatsTPC/TrackCuts.h"

namespace o2::tpc
{

// forward declaration
class TrackTPC;

/// Class that creates dE/dx histograms from a sequence of tracks objects
class CalibdEdx
{
 public:
  using Hist = FastHisto<float>;
  using HistContainer = CalibdEdxDataContainer<Hist>;
  using CalibContainer = CalibdEdxDataContainer<float>;
  using Entries = std::array<float, HistContainer::totalStacks>;

  /// Default constructor
  CalibdEdx() = default;

  /// Constructor that enable tracks cuts
  CalibdEdx(unsigned int nBins, float mindEdx, float maxdEdx, const TrackCuts& cuts);

  /// Constructor that enable tracks cuts, and creates a TrackCuts internally
  CalibdEdx(unsigned int nBins, float mindEdx = 10, float maxdEdx = 100, float minP = 0.4, float maxP = 0.6, int minClusters = 60)
    : CalibdEdx(nBins, mindEdx, maxdEdx, {minP, maxP, static_cast<float>(minClusters)}) {}

  /// Fill histograms using tracks data
  void fill(const gsl::span<const TrackTPC> tracks);

  /// Add counts from other container
  void merge(const CalibdEdx* other);

  /// Compute MIP position from dEdx histograms, and save result in the calib container
  void finalise();

  const HistContainer& getHistos() const { return mHistos; }
  const CalibContainer& getCalib() const { return mCalib; }

  /// \brief Check if there are enough data to compute the calibration.
  /// \return false if any of the histograms has less entries than minEntries
  bool hasEnoughData(size_t minEntries) const;

  /// Print the number of entries in each histogram
  void print() const;

  void setApplyCuts(bool apply) { mApplyCuts = apply; }
  bool getApplyCuts() const { return mApplyCuts; }
  void setCuts(const TrackCuts& cuts) { mCuts = cuts; }

  /// \return number of entries in each histogram
  const Entries& getEntries() const { return mEntries; }

  /// Save the histograms to a file
  void dumpToFile(std::string_view fileName) const;

  /// Find the index of each GEM stacks the track crossed.
  static std::array<size_t, 4> findTrackStacks(const TrackTPC& track, bool& ok);

 private:
  static void mergeHistos(Hist& fist, const Hist& second);

  bool mApplyCuts{true}; ///< Whether or not to apply tracks cuts
  size_t mNBins{};       ///< Number of bins in each histogram
  TrackCuts mCuts;       ///< Cut class

  Entries mEntries{0};   ///< Number of entries in each histogram
  HistContainer mHistos; ///< Histogram container
  CalibContainer mCalib; ///< Calibration output container

  ClassDefNV(CalibdEdx, 1);
};

} // namespace o2::tpc
#endif
