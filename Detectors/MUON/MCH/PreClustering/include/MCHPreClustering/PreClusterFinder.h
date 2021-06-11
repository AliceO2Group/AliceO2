// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @since 2016-10-20
/// @author P. Pillot
/// @brief Class to group the fired pads into preclusters

#ifndef ALICEO2_MCH_PRECLUSTERFINDER_H_
#define ALICEO2_MCH_PRECLUSTERFINDER_H_

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <memory>

#include <gsl/span>

#include "DataFormatsMCH/Digit.h"
#include "MCHBase/PreCluster.h"

namespace o2
{
namespace mch
{

class PreClusterFinder
{
 public:
  PreClusterFinder();
  ~PreClusterFinder();

  PreClusterFinder(const PreClusterFinder&) = delete;
  PreClusterFinder& operator=(const PreClusterFinder&) = delete;
  PreClusterFinder(PreClusterFinder&&) = delete;
  PreClusterFinder& operator=(PreClusterFinder&&) = delete;

  void init();
  void deinit();
  void reset();

  void loadDigits(gsl::span<const Digit> digits);
  void loadDigit(const Digit& digit);

  int discardHighOccupancy(bool perDE, bool perEvent);

  int run();

  void getPreClusters(std::vector<o2::mch::PreCluster>& preClusters, std::vector<Digit>& digits);

 private:
  struct DetectionElement;

  struct PreCluster {
    uint16_t firstPad; // index of first associated pad in the orderedPads array
    uint16_t lastPad;  // index of last associated pad in the orderedPads array
    float area[2][2];  // 2D area containing the precluster
    bool useMe;        // false if precluster already merged to another one
    bool storeMe;      // true if precluster to be saved (merging result)
  };

  void reset(int deIndex);

  void preClusterizeRecursive();
  void addPad(DetectionElement& de, uint16_t iPad, PreCluster& cluster);

  int mergePreClusters();
  void mergePreClusters(PreCluster& cluster, std::vector<std::unique_ptr<PreCluster>> preClusters[2],
                        int nPreClusters[2], DetectionElement& de, int iPlane, PreCluster*& mergedCluster);
  PreCluster* usePreClusters(PreCluster* cluster, DetectionElement& de);
  void mergePreClusters(PreCluster& cluster1, PreCluster& cluster2, DetectionElement& de);

  bool areOverlapping(PreCluster& cluster1, PreCluster& cluster2, DetectionElement& de, float precision);

  void createMapping();

  static constexpr int SNDEs = 156; ///< number of DEs

  std::vector<std::unique_ptr<DetectionElement>> mDEs; ///< internal mapping
  std::unordered_map<int, int> mDEIndices{};           ///< maps DE indices from DE IDs

  int mNPreClusters[SNDEs][2]{};                                     ///< number of preclusters in each cathods of each DE
  std::vector<std::unique_ptr<PreCluster>> mPreClusters[SNDEs][2]{}; ///< preclusters in each cathods of each DE
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PRECLUSTERFINDER_H_
