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
/// @file   TrackClusters.h
/// @author Laura Serksnyte
///

#ifndef AliceO2_TPC_QC_TRACKCLUSTERS_H
#define AliceO2_TPC_QC_TRACKCLUSTERS_H

// root includes
#include "TH1F.h"

// o2 includes
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/TrackTPC.h"

namespace o2::tpc
{
class TrackTPC;
struct ClusterNativeAccess;

namespace qc
{

/// @brief  Shared cluster and crossed rows TPC quality control task
class TrackClusters
{
 public:
  /// \brief Constructor.
  TrackClusters() = default;

  /// bool extracts intormation from track and fills it to histograms
  /// @return true if information can be extracted and filled to histograms
  bool processTrackAndClusters(const std::vector<o2::tpc::TrackTPC>* tracks, const o2::tpc::ClusterNativeAccess* clusterIndex, std::vector<o2::tpc::TPCClRefElem>* clusRefs);

  /// Initialize all histograms
  void initializeHistograms();

  /// Reset all histograms
  void resetHistograms();

  /// Dump results to a file
  void dumpToFile(std::string filename);

  // To set the elementary track cuts
  void setTrackClustersCuts(int minNCls = 60, float mindEdxTot = 10.0, float absEta = 1.)
  {
    mCutMinNCls = minNCls;
    mCutMindEdxTot = mindEdxTot;
    mCutAbsEta = absEta;
  }

  std::unordered_map<std::string, std::vector<std::unique_ptr<TH1>>>& getMapOfHisto() { return mMapHist; }
  const std::unordered_map<std::string, std::vector<std::unique_ptr<TH1>>>& getMapOfHisto() const { return mMapHist; }

 private:
  int mCutMinNCls = 60;        // minimum N clusters
  float mCutMindEdxTot = 10.f; // dEdxTot min value
  float mCutAbsEta = 1.f;      // AbsTgl max cut
  std::unordered_map<std::string, std::vector<std::unique_ptr<TH1>>> mMapHist;
  ClassDefNV(TrackClusters, 1)
};
} // namespace qc
} // namespace o2::tpc

#endif