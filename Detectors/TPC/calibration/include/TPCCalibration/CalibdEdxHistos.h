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

/// \file CalibdEdxHistos.h
/// \brief This file provides the container used for time based dE/dx calibration.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBDEDXHISTOS_H_
#define ALICEO2_TPC_CALIBDEDXHISTOS_H_

#include <array>
#include <cstddef>
#include <gsl/span>
#include <string_view>

// o2 includes
#include "TPCCalibration/CalibdEdxBase.h"
#include "TPCCalibration/FastHisto.h"
#include "DataFormatsTPC/TrackCuts.h"

namespace o2::tpc
{

// forward declaration
class TrackTPC;

/// Class that creates dE/dx histograms from a sequence of tracks objects
class CalibdEdxHistos : public CalibdEdxBase<FastHisto<float>>
{
 public:
  using Hist = FastHisto<float>;

  /// Default constructor
  CalibdEdxHistos() = default;

  /// Constructor that enable tracks cuts
  CalibdEdxHistos(unsigned int nBins, float mindEdx, float maxdEdx, const TrackCuts& cuts);

  /// Constructor that enable tracks cuts, and creates a TrackCuts internally
  CalibdEdxHistos(unsigned int nBins, float mindEdx = 10, float maxdEdx = 100, float minP = 0.4, float maxP = 0.6, int minClusters = 60)
    : CalibdEdxHistos(nBins, mindEdx, maxdEdx, {minP, maxP, static_cast<float>(minClusters)}) {}

  /// Fill histograms using tracks data
  void fill(const gsl::span<const TrackTPC> tracks);

  /// Add counts from other container
  void merge(const CalibdEdxHistos* other);

  /// Print the number of entries in each histogram
  void print() const;

  void setApplyCuts(bool apply) { mApplyCuts = apply; }
  bool getApplyCuts() const { return mApplyCuts; }
  void setCuts(const TrackCuts& cuts) { mCuts = cuts; }

  /// \return number of entries in each histogram
  const std::array<float, totalStacks>& getEntries() const { return mEntries; }

  /// Save the histograms to a file
  void dumpToFile(std::string_view fileName) const;

  /// Find the index of each GEM stacks the track crossed.
  static std::array<float, 4> findTrackStacks(const TrackTPC& track, bool& ok);

 private:
  static void mergeContainer(Container& fist, const Container& second);

  bool mApplyCuts{true}; ///< Whether or not to apply tracks cuts
  size_t mNBins{};       ///< Number of bins in each histogram
  TrackCuts mCuts;       ///< Cut class

  std::array<float, totalStacks> mEntries{0}; ///< Number of entries in each histogram

  ClassDefNV(CalibdEdxHistos, 1);
};

} // namespace o2::tpc
#endif
