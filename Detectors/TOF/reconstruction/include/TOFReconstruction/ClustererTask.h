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
/// \brief Definition of the TOF cluster finder task

#ifndef ALICEO2_TOF_CLUSTERERTASK
#define ALICEO2_TOF_CLUSTERERTASK

#include "FairTask.h"

#include "TOFReconstruction/Clusterer.h"

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace tof
{

class ClustererTask : public FairTask
{

  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

 public:
  ClustererTask(Bool_t useMCTruth = kTRUE);
  ~ClustererTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;

 private:
  DigitDataReader mReader; ///< Digit reader
  Clusterer mClusterer;    ///< Cluster finder

  std::vector<Cluster>* mClustersArray = nullptr;                           ///< Array of clusters
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; ///< MC labels for output
  MCLabelContainer const* mDigitMCTruth;                                    ///< Array for MCTruth information associated to digits

  ClassDefOverride(ClustererTask, 1);
};
} // namespace tof
} // namespace o2

#endif /* ALICEO2_TOF_CLUSTERERTASK */
