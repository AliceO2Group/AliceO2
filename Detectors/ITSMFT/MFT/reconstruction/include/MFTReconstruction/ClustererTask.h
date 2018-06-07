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
/// \brief Task driving the cluster finding from digits
/// \author bogdan.vulpescu@cern.ch
/// \date 03/05/2017

#ifndef ALICEO2_MFT_CLUSTERERTASK_H_
#define ALICEO2_MFT_CLUSTERERTASK_H_

#include "FairTask.h"

#include "MFTBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

class TClonesArray;

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace MFT
{
class EventHeader;
class ClustererTask : public FairTask
{
  using DigitPixelReader = o2::ITSMFT::DigitPixelReader;
  using Clusterer = o2::ITSMFT::Clusterer;
  using Cluster = o2::ITSMFT::Cluster;
  using MCTruth = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

 public:
  ClustererTask(bool useMC = true);
  ~ClustererTask() override;

  InitStatus Init() override;
  void Exec(Option_t* opt) override;

 private:
  bool mUseMCTruth = true;                             ///< flag to use MCtruth if available
  const o2::ITSMFT::GeometryTGeo* mGeometry = nullptr; ///< ITS OR MFT upgrade geometry
  DigitPixelReader mReader;                            ///< Pixel reader
  Clusterer mClusterer;                                ///< Cluster finder

  std::vector<Cluster> mClustersArray;                       //!< Array of clusters
  std::vector<Cluster>* mClustersArrayPtr = &mClustersArray; //!< Array of clusters pointer
  MCTruth mClsLabels;                                        //! MC labels
  MCTruth* mClsLabelsPtr = &mClsLabels;                      //! MC labels pointer

  ClassDefOverride(ClustererTask, 1);
};
}
}

#endif
