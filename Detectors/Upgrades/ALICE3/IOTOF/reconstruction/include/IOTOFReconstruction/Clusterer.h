// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Definition of the IOTOF cluster finder

#ifndef ALICEO2_IOTOF_CLUSTERER_H
#define ALICEO2_IOTOF_CLUSTERER_H

#include "DataFormatsIOTOF/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsIOTOF/Cluster.h"
#include "IOTOFSimulation/DPLDigitizerParam.h"
#include "IOTOFReconstruction/ClustererParam.h"
#include "IOTOFReconstruction/TopologyClassifier.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <gsl/span>
#include <vector>
#include <array>
#include <memory>
#include <cstring>
#include <utility>

namespace o2::iotof
{

class GeometryTGeo;

class Clusterer
{
 public:
  static constexpr int MaxLabels = 10;

  using Digit = o2::iotof::Digit;
  using DigROFRecord = o2::itsmft::ROFRecord;
  using DigMC2ROFRecord = o2::itsmft::MC2ROFRecord;
  using ClusterTruth = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using ConstDigitTruth = o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>;
  using Label = o2::MCCompLabel;

  //----------------------------------------------
  struct ClustererThread {
    Clusterer* mParent = nullptr;
    // Column buffers data members in TRK, for now not needed in TF3

    // Further struct members in TRK, for now not needed in TF3

    std::array<Label, MaxLabels> mLabelsBuff; ///< MC label buffer for one cluster

    // per-thread output (accumulated, then merged back by caller)
    std::vector<Cluster> mClusters;
    std::vector<uint16_t> mPatterns;
    ClusterTruth mLabels;

    // Further reset column buffer in TRK, not included for now in TF3
    TopologyClassifier mClsTopoClassifier; //! Convert the cluster topology to the corresponding entry in the dictionary.

    void fetchMCLabels(uint32_t digID, const ConstDigitTruth* labelsDig, int& nfilled);
    void findClustersSingleHit(gsl::span<const Digit> digits, uint32_t digitIdx,
                               const ConstDigitTruth* labelsDigPtr, ClusterTruth* labelsClusPtr);
    void findClustersMultipleHits(gsl::span<const Digit> digits, gsl::span<const uint32_t> digitIdxs,
                                  const ConstDigitTruth* labelsDigPtr, ClusterTruth* labelsClusPtr);
    void processChip(gsl::span<const Digit> digits, int chipFirst, int chipN,
                     std::vector<Cluster>* clustersOut, std::vector<unsigned char>* patternsOut,
                     const ConstDigitTruth* labelsDigPtr, ClusterTruth* labelsClusPtr);
    void writeTopologiesToFile(const char* filename);

    explicit ClustererThread(Clusterer* par = nullptr) : mParent(par) {}
    ClustererThread(const ClustererThread&) = delete;
    ClustererThread& operator=(const ClustererThread&) = delete;
  };
  //----------------------------------------------

  virtual void process(gsl::span<const Digit> digits,
                       gsl::span<const DigROFRecord> digitROFs,
                       std::vector<o2::iotof::Cluster>& clusters,
                       std::vector<unsigned char>& patterns,
                       std::vector<o2::itsmft::ROFRecord>& clusterROFs,
                       const ConstDigitTruth* digitLabels = nullptr,
                       ClusterTruth* clusterLabels = nullptr,
                       gsl::span<const DigMC2ROFRecord> digMC2ROFs = {},
                       std::vector<o2::itsmft::MC2ROFRecord>* clusterMC2ROFs = nullptr);

  // ///< load the dictionary of cluster topologies
  // void loadDictionary(const std::string& fileName) { mPattIdConverter.loadDictionary(fileName); }
  // void setDictionary(const TopologyDictionary* dict) { mPattIdConverter.setDictionary(dict); }
  // const TopologyDictionary& getDictionary() const { return mPattIdConverter.getDictionary(); }
  // auto& getPattIdConverter() const { return mPattIdConverter; }

 protected:
  std::unique_ptr<ClustererThread> mThread;
  std::vector<int> mSortIdx; ///< reusable per-ROF sort buffer
};

} // namespace o2::iotof

#endif
