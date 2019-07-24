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

#include "MFTBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <memory>

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace mft
{

class ClustererTask
{
  using Clusterer = o2::itsmft::Clusterer;
  using Cluster = o2::itsmft::Cluster;
  using CompCluster = o2::itsmft::CompCluster;
  using CompClusterExt = o2::itsmft::CompClusterExt;
  using MCTruth = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

 public:
  ClustererTask(bool useMC = true, bool raw = false);
  ~ClustererTask();

  void Init();
  Clusterer& getClusterer() { return mClusterer; }
  void run(const std::string inpName, const std::string outName, bool entryPerROF = true);
  void setSelfManagedMode(bool v) { mSelfManagedMode = v; }
  bool isSelfManagedMode() const { return mSelfManagedMode; }
  o2::itsmft::PixelReader* getReader() const { return (o2::itsmft::PixelReader*)mReader; }
  void loadDictionary(std::string fileName) { mClusterer.loadDictionary(fileName); }

 private:
  bool mSelfManagedMode = false;                                                      ///< manages itself input output
  bool mRawDataMode = false;                                                          ///< input from raw data or MC digits
  bool mUseMCTruth = true;                                                            ///< flag to use MCtruth if available
  o2::itsmft::PixelReader* mReader = nullptr;                                         ///< Pointer on the relevant Pixel reader
  std::unique_ptr<o2::itsmft::DigitPixelReader> mReaderMC;                            ///< reader for MC data
  std::unique_ptr<o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingMFT>> mReaderRaw; ///< reader for raw data

  const o2::itsmft::GeometryTGeo* mGeometry = nullptr; ///< ITS OR MFT upgrade geometry
  Clusterer mClusterer;                                ///< Cluster finder

  std::vector<Cluster> mFullClus;               //!< vector of full clusters
  std::vector<Cluster>* mFullClusPtr = nullptr; //!< vector of full clusters pointer

  std::vector<CompClusterExt> mCompClus;               //!< vector of compact clusters
  std::vector<CompClusterExt>* mCompClusPtr = nullptr; //!< vector of compact clusters pointer

  std::vector<o2::itsmft::ROFRecord> mROFRecVec;               //!< vector of ROFRecord references
  std::vector<o2::itsmft::ROFRecord>* mROFRecVecPtr = nullptr; //!< vector of ROFRecord references pointer

  MCTruth mClsLabels;               //! MC labels
  MCTruth* mClsLabelsPtr = nullptr; //! MC labels pointer (optional)

  ClassDefNV(ClustererTask, 1);
};
} // namespace mft
} // namespace o2

#endif
