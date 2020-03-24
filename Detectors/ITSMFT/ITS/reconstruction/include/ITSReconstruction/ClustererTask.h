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

#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
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
#include <limits>

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace its
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
  void run(const std::string inpName, const std::string outName);
  o2::itsmft::PixelReader* getReader() const { return (o2::itsmft::PixelReader*)mReader; }

  void loadDictionary(std::string fileName) { mClusterer.loadDictionary(fileName); }

  void writeTree(std::string basename, int i);
  void setMaxROframe(int max) { maxROframe = max; }
  int getMaxROframe() const { return maxROframe; }

  void setPatterns() { mClusterer.setPatterns(&mPatterns); }

 private:
  int maxROframe = std::numeric_limits<int>::max();                                   ///< maximal number of RO frames per a file
  bool mRawDataMode = false;                                                          ///< input from raw data or MC digits
  bool mUseMCTruth = true;                                                            ///< flag to use MCtruth if available
  o2::itsmft::PixelReader* mReader = nullptr;                                         ///< Pointer on the relevant Pixel reader
  std::unique_ptr<o2::itsmft::DigitPixelReader> mReaderMC;                            ///< reader for MC data
  std::unique_ptr<o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS>> mReaderRaw; ///< reader for raw data

  const o2::itsmft::GeometryTGeo* mGeometry = nullptr; ///< ITS OR MFT upgrade geometry
  Clusterer mClusterer;                                ///< Cluster finder

  std::vector<Cluster> mFullClus;               //!< vector of full clusters

  std::vector<CompClusterExt> mCompClus;               //!< vector of compact clusters

  std::vector<o2::itsmft::ROFRecord> mROFRecVec;               //!< vector of ROFRecord references

  MCTruth mClsLabels;               //! MC labels

  std::vector<o2::itsmft::ClusterTopology> mPatterns;

  ClassDefNV(ClustererTask, 2);
};
} // namespace its
} // namespace o2

#endif /* ALICEO2_ITS_CLUSTERERTASK */
