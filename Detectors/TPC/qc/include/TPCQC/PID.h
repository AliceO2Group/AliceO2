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

///
/// @file   PID.h
/// @author Thomas Klemenz, thomas.klemenz@tum.de
///

#ifndef AliceO2_TPC_QC_PID_H
#define AliceO2_TPC_QC_PID_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <string_view>

// root includes
#include "TH1.h"

// NOTE
// required for backward compatibility, will be removed in the next PR
#include "TH2.h"

// o2 includes
#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

class TrackTPC;

namespace qc
{

/// @brief  PID quality control class
///
/// This class is used to extract PID related variables
/// from TrackTPC objects and store it in histograms.
///
/// origin: TPC
/// @author Thomas Klemenz, thomas.klemenz@tum.de
class PID
{
 public:
  /// \brief Constructor.
  PID() = default;

  /// bool extracts intormation from track and fills it to histograms
  /// @return true if information can be extracted and filled to histograms
  bool processTrack(const o2::tpc::TrackTPC& track);

  /// Initialize all histograms
  void initializeHistograms();

  /// Reset all histograms
  void resetHistograms();

  /// Dump results to a file
  void dumpToFile(std::string filename);

  std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>>& getMapOfHisto() { return mMapHist; }
  const std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>>& getMapOfHisto() const { return mMapHist; }

  // NOTE
  // we need these two function to make backward compatibility, The CI check trigger QC test and there these two functions are required
  // I will remove these two functions in the next PR once this PR is merged.
  /// get 1D histograms
  std::vector<TH1F>& getHistograms1D() { return mHist1D; }
  const std::vector<TH1F>& getHistograms1D() const { return mHist1D; }

  /// get 2D histograms
  std::vector<TH2F>& getHistograms2D() { return mHist2D; }
  const std::vector<TH2F>& getHistograms2D() const { return mHist2D; }

 private:
  std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>> mMapHist;
  // NOTE
  // same reason these two are need to remove the circular dependencies and make backward compatible for QC
  // I will remove these two functions in the next PR once this PR is merged.
  std::vector<TH1F> mHist1D{};
  std::vector<TH2F> mHist2D{};
  ClassDefNV(PID, 1)
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif