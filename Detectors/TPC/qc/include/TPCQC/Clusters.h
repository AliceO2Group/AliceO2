// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

//root includes
#include "TCanvas.h"

//o2 includes
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

  bool processCluster(const o2::tpc::ClusterNative& cluster, const o2::tpc::Sector sector, const int row);

  void analyse();

  void dumpToFile(std::string filename);

  const CalPad& getNClusters() const { return mNClusters; }
  const CalPad& getQMax() const { return mQMax; }
  const CalPad& getQTot() const { return mQTot; }
  const CalPad& getSigmaTime() const { return mSigmaTime; }
  const CalPad& getSigmaPad() const { return mSigmaPad; }
  const CalPad& getTimeBin() const { return mTimeBin; }

 private:
  CalPad mNClusters{"N_Clusters"};
  CalPad mQMax{"Q_Max"};
  CalPad mQTot{"Q_Tot"};
  CalPad mSigmaTime{"Sigma_Time"};
  CalPad mSigmaPad{"Sigma_Pad"};
  CalPad mTimeBin{"Time_Bin"};

  ClassDefNV(Clusters, 1)
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif