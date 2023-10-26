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
/// @file   Tracks.h
/// @author Stefan Heckel, sheckel@cern.ch
///

#ifndef AliceO2_TPC_QC_TRACKS_H
#define AliceO2_TPC_QC_TRACKS_H

#include <vector>
#include <string_view>
#include <string>
#include <unordered_map>
#include <memory>

// root includes
#include "TH1.h"
#include "TH1F.h"
#include "TH2F.h"

namespace o2
{
namespace tpc
{

class TrackTPC;

namespace qc
{

/// @brief Tracks quality control class
///
/// This class is used to extract track related variables
/// from TrackTPC objects and store it in histograms.
///
/// origin: TPC
/// @author Stefan Heckel, sheckel@cern.ch
class Tracks
{
 public:
  // default constructor
  Tracks() = default;

  /// bool extracts intormation from track and fills it to histograms
  /// @return true if information can be extracted and filled to histograms
  bool processTrack(const o2::tpc::TrackTPC& track);

  /// Initialize all histograms
  void initializeHistograms();

  /// Reset all histograms
  void resetHistograms();

  /// Function to be called at each endOfCycle
  void processEndOfCycle();

  /// Dump results to a file
  void dumpToFile(std::string_view filename);

  // To set the elementary track cuts
  void setTrackCuts(float AbsEta = 1.,
                    int nClusterCut = 60, float dEdxTot = 20, float cutPtForDCAr = 1.5, float samplingFractionDCAr = 0.1)
  {
    mCutAbsEta = AbsEta;
    mCutMinnCls = nClusterCut;
    mCutMindEdxTot = dEdxTot;
    mCutMinPtDCAr = cutPtForDCAr;
    mSamplingFractionDCAr = samplingFractionDCAr;
  }

  // Just for backward compatibility with crrent QC, temporary, will be removed in the next PR
  /// get 1D histograms
  std::vector<TH1F>& getHistograms1D() { return mHist1D; }
  const std::vector<TH1F>& getHistograms1D() const { return mHist1D; }

  // Just for backward compatibility with crrent QC, temporary, will be removed in the next PR
  /// get 2D histograms
  std::vector<TH2F>& getHistograms2D() { return mHist2D; }
  const std::vector<TH2F>& getHistograms2D() const { return mHist2D; }

  // Just for backward compatibility with crrent QC, temporary, will be removed in the next PR
  /// get ratios of 1D histograms
  std::vector<TH1F>& getHistogramRatios1D() { return mHistRatio1D; }
  const std::vector<TH1F>& getHistogramRatios1D() const { return mHistRatio1D; }

  /// get ratios of 1D histograms
  std::unordered_map<std::string, std::unique_ptr<TH1>>& getMapHist() { return mMapHist; }
  const std::unordered_map<std::string, std::unique_ptr<TH1>>& getMapHist() const { return mMapHist; }

 private:
  float mCutAbsEta = 1.f;             // Eta cut
  int mCutMinnCls = 60;               // minimum N clusters
  float mCutMindEdxTot = 20.f;        // dEdxTot min value
  float mCutMinPtDCAr = 1.5f;         // minimum pT for DCAr plots DCAr vs. phi, eta, nCluster
  float mSamplingFractionDCAr = 0.1f; // sampling rate for calculation of DCAr
  std::unordered_map<std::string, std::unique_ptr<TH1>> mMapHist;
  std::vector<TH1F> mHist1D{};      ///< Initialize vector of 1D histograms
  std::vector<TH2F> mHist2D{};      ///< Initialize vector of 2D histograms
  std::vector<TH1F> mHistRatio1D{}; ///< Initialize vector of ratios of 1D histograms

  ClassDefNV(Tracks, 2)
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif
