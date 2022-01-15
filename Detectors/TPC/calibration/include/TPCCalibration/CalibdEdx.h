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

#include <cstddef>
#include <gsl/span>
#include <string_view>
#include <array>

// o2 includes
#include "DataFormatsTPC/TrackCuts.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/CalibdEdxCorrection.h"

// boost includes
#include <boost/histogram.hpp>
#include "THn.h"

namespace o2::tpc
{

// forward declaration
class TrackTPC;

/// Class that creates dE/dx histograms from a sequence of tracks objects
class CalibdEdx
{
 public:
  enum Axis {
    dEdx,
    Tgl,
    Snp,
    // Z,
    Sector,
    Stack,
    Charge,
    Size ///< Number of axes
  };

  // Interger histogram axis identifying the GEM stacks, without under and overflow bins.
  using IntAxis = boost::histogram::axis::integer<int, boost::histogram::use_default, boost::histogram::axis::option::none_t>;
  // Float axis to store data, without under and overflow bins.
  using FloatAxis = boost::histogram::axis::regular<float, boost::histogram::use_default, boost::histogram::use_default, boost::histogram::axis::option::none_t>;

  // Define histogram axes types
  // on changing the axis order also change the constructor and fill functions order in de .cxx
  // and the HistAxis enum
  using AxesType = std::tuple<
    FloatAxis, // dEdx
    FloatAxis, // Tgl
    FloatAxis, // Snp
    // FloatAxis, // Z
    IntAxis, // sector
    IntAxis, // stackType
    IntAxis  // Charge
    >;

  using Hist = boost::histogram::histogram<AxesType>;
  using FitCuts = std::array<int, 3>;

  CalibdEdx(int dEdxBins = 100, float mindEdx = 5, float maxdEdx = 70, int angularBins = 18, float maxTgl = 1.5);

  void setCuts(const TrackCuts& cuts) { mCuts = cuts; }
  void setApplyCuts(bool apply) { mApplyCuts = apply; }
  bool getApplyCuts() const { return mApplyCuts; }

  float getField() const { return mField; };
  void setField(float field) { mField = field; }

  FitCuts getFitCuts() const { return mFitCuts; }
  void setFitCuts(const FitCuts& cuts) { mFitCuts = cuts; }

  /// Fill histograms using tracks data.
  void fill(const TrackTPC& tracks);
  void fill(const gsl::span<const TrackTPC>);
  void fill(const std::vector<TrackTPC>& tracks) { fill(gsl::span(tracks)); }

  /// Add counts from other container.
  void merge(const CalibdEdx* other);

  /// Compute MIP position from dEdx histograms, and save result in the correction container.
  void finalize();

  /// Return calib data histogram
  const Hist& getHist() const { return mHist; }
  /// Return calib data as a THn
  THnF* getRootHist() const;

  const CalibdEdxCorrection& getCalib() const { return mCalib; }

  ///< Return the number of hist entries of the gem stack with less statistics
  int minStackEntries() const;

  /// \brief Check if there are enough data to compute the calibration.
  /// \return false if any of the GEM stacks has less entries than minEntries
  bool hasEnoughData(float minEntries) const;

  /// Find the sector of a track at a given GEM stack type.
  static int findTrackSector(const TrackTPC& track, GEMstack, bool& ok);

  /// Print the number of entries in each histogram.
  void print() const;

  /// Save the histograms to a TTree.
  void writeTTree(std::string_view fileName) const;

  constexpr static float mipScale = 1.0 / 50.0; ///< Inverse of target MIP dE/dx value

  constexpr static std::array<float, 4> tglScale{1.9, 1.5, 1.22, 1.02}; ///< Max Tgl values for each ROC type

 private:
  float mField = -5;                ///< Magnetic field in kG, used for track propagation
  bool mApplyCuts{true};            ///< Wether or not to apply tracks cuts
  TrackCuts mCuts{0.4, 0.6, 60};    ///< Cut values
  FitCuts mFitCuts{100, 500, 2500}; ///< Minimum entries per stack to perform a sector, 1D and 2D fit, for each GEM stack

  Hist mHist;                   ///< dEdx multidimensional histogram
  CalibdEdxCorrection mCalib{}; ///< Calibration output

  ClassDefNV(CalibdEdx, 1);
};

} // namespace o2::tpc

#endif
