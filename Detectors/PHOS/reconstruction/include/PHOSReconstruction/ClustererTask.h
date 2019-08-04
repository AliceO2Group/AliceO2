// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClustererTask.h
/// \brief Definition of the PHOS cluster finder task

#ifndef ALICEO2_PHOS_CLUSTERERTASK
#define ALICEO2_PHOS_CLUSTERERTASK

#include "FairTask.h"

#include "PHOSReconstruction/Clusterer.h"

namespace o2
{

namespace phos
{

class Cluster;
class Digit;
class Clusterer;

class ClustererTask : public FairTask
{

 public:
  ClustererTask();
  ~ClustererTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;

 private:
  std::vector<Cluster>* mClustersArray = nullptr; ///< Array of clusters
  const std::vector<Digit>* mDigitsArray;         ///< Input array of digits
  Clusterer* mClusterer;                          ///< Clusterer to do the job
  ClassDefOverride(ClustererTask, 1);
};
} // namespace phos
} // namespace o2

#endif /* ALICEO2_PHOS_CLUSTERERTASK */
