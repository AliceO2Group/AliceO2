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

#include "MCHBase/DigitBlock.h"
#include "MCHBase/Mapping.h"

namespace o2
{
namespace mch
{

class PreClusterFinder
{
 public:
  struct PreCluster {
    uint16_t firstPad; // index of first associated pad in the orderedPads array
    uint16_t lastPad;  // index of last associated pad in the orderedPads array
    float area[2][2];  // 2D area containing the precluster
    bool useMe;        // false if precluster already merged to another one
    bool storeMe;      // true if precluster to be saved (merging result)
  };

  PreClusterFinder() = default;
  ~PreClusterFinder() = default;

  PreClusterFinder(const PreClusterFinder&) = delete;
  PreClusterFinder& operator=(const PreClusterFinder&) = delete;
  PreClusterFinder(PreClusterFinder&&) = delete;
  PreClusterFinder& operator=(PreClusterFinder&&) = delete;

  void init(std::string& fileName);
  void deinit();
  void reset();

  void loadDigits(const DigitStruct* digits, uint32_t nDigits);

  int run();

  int getNDEWithPreClusters(int& nUsedDigits);
  bool hasPreClusters(int iDE);
  int getNPreClusters(int iDE, int iPlane);
  const PreCluster* getPreCluster(int iDE, int iPlane, int iCluster);
  const DigitStruct* getDigit(int iDE, uint16_t iOrderedPad);

  /// return the number of detection elements in the internal structure
  static constexpr int getNDEs() { return SNDEs; }
  int getDEId(int iDE);

 private:
  struct DetectionElement {
    std::unique_ptr<Mapping::MpDE> mapping; // mapping of this DE including the list of pads
    std::vector<const DigitStruct*> digits; // list of pointers to digits (not owner)
    uint16_t nFiredPads[2];                 // number of fired pads on each plane
    std::vector<uint16_t> firedPads[2];     // indices of fired pads on each plane
    uint16_t nOrderedPads[2];               // current number of fired pads in the following arrays
    std::vector<uint16_t> orderedPads[2];   // indices of fired pads ordered after preclustering and merging
  };

  /// Return detection element ID part of the unique ID
  int detectionElementId(uint32_t uid) { return uid & 0xFFF; }

  /// Return the cathode part of the unique ID
  int cathode(uint32_t uid) { return (uid & 0x40000000) >> 30; }

  void preClusterizeRecursive();
  void addPad(DetectionElement& de, uint16_t iPad, PreCluster& cluster);

  int mergePreClusters();
  void mergePreClusters(PreCluster& cluster, std::vector<std::unique_ptr<PreCluster>> preClusters[2],
                        int nPreClusters[2], DetectionElement& de, int iPlane, PreCluster*& mergedCluster);
  PreCluster* usePreClusters(PreCluster* cluster, DetectionElement& de);
  void mergePreClusters(PreCluster& cluster1, PreCluster& cluster2, DetectionElement& de);

  bool areOverlapping(PreCluster& cluster1, PreCluster& cluster2, DetectionElement& de, float precision);

  void readMapping(const char* fileName);

  static constexpr int SNDEs = 156; ///< number of DEs

  DetectionElement mDEs[SNDEs]{};            ///< internal mapping
  std::unordered_map<int, int> mDEIndices{}; ///< maps DE indices from DE IDs

  int mNPreClusters[SNDEs][2]{};                                     ///< number of preclusters in each cathods of each DE
  std::vector<std::unique_ptr<PreCluster>> mPreClusters[SNDEs][2]{}; ///< preclusters in each cathods of each DE
};

//_________________________________________________________________________________________________
inline int PreClusterFinder::getNDEWithPreClusters(int& nUsedDigits)
{
  /// return number of DEs with fired pads

  int nFiredDE(0);
  nUsedDigits = 0;

  for (int iDE = 0; iDE < SNDEs; ++iDE) {

    DetectionElement& de(mDEs[iDE]);

    if (de.nOrderedPads[1] > 0) {
      nUsedDigits += de.nOrderedPads[1];
      ++nFiredDE;
    }
  }

  return nFiredDE;
}

//_________________________________________________________________________________________________
inline bool PreClusterFinder::hasPreClusters(int iDE)
{
  /// return true if this DE contains preclusters
  assert(iDE >= 0 && iDE < SNDEs);
  return (mDEs[iDE].nOrderedPads[1] > 0);
}

//_________________________________________________________________________________________________
inline int PreClusterFinder::getNPreClusters(int iDE, int iPlane)
{
  /// return number of preclusters in plane "iPlane" of DE "iDE"
  assert(iDE >= 0 && iDE < SNDEs && iPlane >= 0 && iPlane < 2);
  return mNPreClusters[iDE][iPlane];
}

//_________________________________________________________________________________________________
inline const PreClusterFinder::PreCluster* PreClusterFinder::getPreCluster(int iDE, int iPlane, int iCluster)
{
  /// return the preclusters "iCluster" in plane "iPlane" of DE "iDE"
  assert(iDE >= 0 && iDE < SNDEs && iPlane >= 0 && iPlane < 2 && iCluster >= 0 &&
         iCluster < mNPreClusters[iDE][iPlane]);
  return mPreClusters[iDE][iPlane][iCluster].get();
}

//_________________________________________________________________________________________________
inline const DigitStruct* PreClusterFinder::getDigit(int iDE, uint16_t iOrderedPad)
{
  /// return the digit associated to the pad registered at the index "iOrderedPad".
  /// This index must be in the range [firstPad, lastPad] associated to a precluster to be stored.
  assert(iDE >= 0 && iDE < SNDEs && iOrderedPad >= 0 && iOrderedPad < mDEs[iDE].nOrderedPads[1]);
  DetectionElement& de(mDEs[iDE]);
  return de.digits[de.mapping->pads[de.orderedPads[1][iOrderedPad]].iDigit];
}

//_________________________________________________________________________________________________
inline int PreClusterFinder::getDEId(int iDE)
{
  /// return the unique ID of the DE "iDE"
  assert(iDE >= 0 && iDE < SNDEs);
  return mDEs[iDE].mapping->uid;
}

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PRECLUSTERFINDER_H_
