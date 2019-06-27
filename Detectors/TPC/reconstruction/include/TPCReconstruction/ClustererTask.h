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
/// \brief TPC Clusterer Task
/// \author Sebastian Klewin <sebastian.klewin@cern.ch>

#ifndef __ALICEO2__ClustererTask__
#define __ALICEO2__ClustererTask__

#include "FairTask.h"  // for FairTask, InitStatus
#include "Rtypes.h"    // for ClustererTask::Class, ClassDef, etc

#include "TPCBase/Digit.h"
#include "TPCReconstruction/HwClusterer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include <vector>
#include <memory>

namespace o2 {
namespace tpc{

class ClustererTask : public FairTask{

  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using OutputType = ClusterHardwareContainer8kb;

 public:
  /// Default constructor
  /// \param sectorid Sector to be processed
  ClustererTask(int sectorid = -1);

  /// Destructor
  ~ClustererTask() override = default;

  /// Initializes the clusterer and connects input and output container
  InitStatus Init() override;

  /// Clusterization
  void Exec(Option_t* option) override;

  /// Complete Clusterization
  void FinishTask() override;

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  void setContinuousReadout(bool isContinuous);

 private:
  bool mIsContinuousReadout = true; ///< Switch for continuous readout
  int mEventCount = 0;              ///< Event counter
  int mClusterSector = -1;          ///< Sector to be processed

  std::unique_ptr<HwClusterer> mHwClusterer; ///< Hw Clusterfinder instance

  // Digit arrays
  std::unique_ptr<const std::vector<Digit>> mDigitsArray;     ///< Array of TPC digits
  std::unique_ptr<const MCLabelContainer> mDigitMCTruthArray; ///< Array for MCTruth information associated to digits in mDigitsArrray

  // Cluster arrays
  std::unique_ptr<std::vector<OutputType>> mHwClustersArray; ///< Array of clusters found by Hw Clusterfinder
  std::unique_ptr<MCLabelContainer> mHwClustersMCTruthArray; ///< Array for MCTruth information associated to cluster in mHwClustersArrays

  ClassDefOverride(ClustererTask, 1)
};

inline
void ClustererTask::setContinuousReadout(bool isContinuous)
{
  mIsContinuousReadout = isContinuous;
}

}
}

#endif
