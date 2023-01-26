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
/// @file   Clusters.h
/// @author
///

#ifndef AliceO2_TPC_CLUSTERS_H
#define AliceO2_TPC_CLUSTERS_H

// root includes
#include "TCanvas.h"

// o2 includes
#include "TPCBase/CalDet.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

class ClusterNative;

namespace qc
{

/// Keep QC information for Clusters related observables
///
class Clusters
{
 public:
  Clusters() = default;

  Clusters(std::string_view nclName) : mNClusters{nclName} {};

  template <class T>
  bool processCluster(const T& cluster, const o2::tpc::Sector sector, const int row);

  void fillADCValue(int cru, int rowInSector, int padInRow, int timeBin, float adcValue);

  void normalize(const float nHBFPerTF = 128);

  inline void analyse() { Clusters::normalize(); } // deprecated

  void denormalize();

  void reset();

  void merge(Clusters& clusters);

  void dumpToFile(std::string filename, int type = 0);

  const CalPad& getNClusters() const { return mNClusters; }
  const CalPad& getQMax() const { return mQMax; }
  const CalPad& getQTot() const { return mQTot; }
  const CalPad& getSigmaTime() const { return mSigmaTime; }
  const CalPad& getSigmaPad() const { return mSigmaPad; }
  const CalPad& getTimeBin() const { return mTimeBin; }
  const CalPad& getOccupancy() const { return mOccupancy; }

  CalPad& getNClusters() { return mNClusters; }
  CalPad& getQMax() { return mQMax; }
  CalPad& getQTot() { return mQTot; }
  CalPad& getSigmaTime() { return mSigmaTime; }
  CalPad& getSigmaPad() { return mSigmaPad; }
  CalPad& getTimeBin() { return mTimeBin; }
  CalPad& getOccupancy() { return mOccupancy; }

  void endTF() { ++mProcessedTFs; }

  size_t getProcessedTFs() { return mProcessedTFs; }

 private:
  CalPad mNClusters{"N_Clusters"};
  CalPad mQMax{"Q_Max"};
  CalPad mQTot{"Q_Tot"};
  CalPad mSigmaTime{"Sigma_Time"};
  CalPad mSigmaPad{"Sigma_Pad"};
  CalPad mTimeBin{"Time_Bin"};
  CalPad mOccupancy{"Occupancy"};
  size_t mProcessedTFs{0};
  bool mIsNormalized{false};

  ClassDefNV(Clusters, 1)
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif
