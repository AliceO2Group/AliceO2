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
/// \brief Definition of the ITS cluster finder task

#ifndef ALICEO2_ITS_CLUSTERERTASK
#define ALICEO2_ITS_CLUSTERERTASK

#include "FairTask.h" 

#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <memory>

namespace o2
{
class MCCompLabel;
namespace dataformats
{
  template<typename T>
  class MCTruthContainer;
}
 
namespace ITS
{
  
class ClustererTask : public FairTask
{
  using Clusterer = o2::ITSMFT::Clusterer;
  using Cluster = o2::ITSMFT::Cluster;
  using CompCluster = o2::ITSMFT::CompCluster;
  using CompClusterExt = o2::ITSMFT::CompClusterExt;
  using MCTruth = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

 public:
  ClustererTask(bool useMC = true, bool raw = false);
  ~ClustererTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;
  Clusterer& getClusterer() { return mClusterer; }
  void run(const std::string inpName, const std::string outName, bool entryPerROF = true);
  void setSelfManagedMode(bool v) { mSelfManagedMode = v; }
  bool isSelfManagedMode() const { return mSelfManagedMode; }
  void attachFairManagerIO();
  o2::ITSMFT::PixelReader* getReader() const { return (o2::ITSMFT::PixelReader*)mReader; }

 private:
  bool mSelfManagedMode = false;                          ///< manages itself input output
  bool mRawDataMode = false;                              ///< input from raw data or MC digits
  bool mUseMCTruth = true;                                ///< flag to use MCtruth if available
  o2::ITSMFT::PixelReader* mReader = nullptr;             ///< Pointer on the relevant Pixel reader
  std::unique_ptr<o2::ITSMFT::DigitPixelReader> mReaderMC;                            ///< reader for MC data
  std::unique_ptr<o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS>> mReaderRaw; ///< reader for raw data

  const o2::ITSMFT::GeometryTGeo* mGeometry = nullptr;    ///< ITS OR MFT upgrade geometry
  Clusterer mClusterer;                                   ///< Cluster finder

  std::vector<Cluster> mFullClus;               //!< vector of full clusters
  std::vector<Cluster>* mFullClusPtr = nullptr; //!< vector of full clusters pointer

  std::vector<CompClusterExt> mCompClus;               //!< vector of compact clusters
  std::vector<CompClusterExt>* mCompClusPtr = nullptr; //!< vector of compact clusters pointer

  MCTruth mClsLabels;                                        //! MC labels
  MCTruth* mClsLabelsPtr = nullptr;                          //! MC labels pointer (optional)

  ClassDefOverride(ClustererTask, 1)
};
}
}

#endif /* ALICEO2_ITS_CLUSTERERTASK */
