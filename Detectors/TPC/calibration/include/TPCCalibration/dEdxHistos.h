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

/// \file dEdxHistos.h
/// \brief This file provides the container used for time based dE/dx calibration.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_DEDXHISTOS_H_
#define ALICEO2_TPC_DEDXHISTOS_H_

#include <array>
#include <gsl/span>
#include <string_view>

// o2 includes
#include "TPCCalibration/FastHisto.h"
#include "DataFormatsTPC/TrackCuts.h"

namespace o2::tpc
{

// forward declaration
class TrackTPC;

/// Class that creates dE/dx histograms from a sequence of tracks objects
class dEdxHistos
{
  using Hist = FastHisto<float>;

 public:
  /// Default constructor
  dEdxHistos() = default;

  /// Constructor that enable tracks cuts
  dEdxHistos(unsigned int nBins, const TrackCuts& cuts);

  /// Constructor that enable tracks cuts, and creates a TrackCuts internally
  dEdxHistos(unsigned int nBins, float minP = 0.4, float maxP = 0.6, int minClusters = 60)
    : dEdxHistos(nBins, {minP, maxP, static_cast<float>(minClusters)}) {}

  /// Fill histograms using tracks data
  void fill(const gsl::span<const TrackTPC> tracks);

  /// Add counts from other container
  void merge(const dEdxHistos* other);

  /// Print the number of entries in each histogram
  void print() const;

  void setApplyCuts(bool apply) { mApplyCuts = apply; }
  bool getApplyCuts() { return mApplyCuts; }
  void setCuts(const TrackCuts& cuts) { mCuts = cuts; }

  /// \return number of entries in the A side histogram
  double getASideEntries() const { return mEntries[0]; }

  /// \return number of entries in the C side histogram
  double getCSideEntries() const { return mEntries[1]; }

  /// Get the underlying histograms
  const std::array<Hist, 2>& getHists() const { return mHist; }

  /// Save the histograms to a file
  void dumpToFile(std::string_view fileName) const;

 private:
  bool mApplyCuts{true}; ///< Whether or not to apply tracks cuts
  TrackCuts mCuts;       ///< Cut class

  std::array<float, 2> mEntries{0, 0}; ///< Number of entries in each histogram
  std::array<Hist, 2> mHist;           ///< MIP position histograms, for TPC's A and C sides

  ClassDefNV(dEdxHistos, 1);
};

} // namespace o2::tpc
#endif
