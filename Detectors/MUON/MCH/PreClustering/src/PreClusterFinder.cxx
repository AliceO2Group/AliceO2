// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHPreClustering/PreClusterFinder.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

#include <fairmq/Tools.h>
#include <FairMQLogger.h>

#include "PreClusterFinderMapping.h"

namespace o2
{
namespace mch
{

struct PreClusterFinder::DetectionElement {
  std::unique_ptr<Mapping::MpDE> mapping; // mapping of this DE including the list of pads
  std::vector<const Digit*> digits;       // list of pointers to digits (not owner)
  uint16_t nFiredPads[2];                 // number of fired pads on each plane
  std::vector<uint16_t> firedPads[2];     // indices of fired pads on each plane
  uint16_t nOrderedPads[2];               // current number of fired pads in the following arrays
  std::vector<uint16_t> orderedPads[2];   // indices of fired pads ordered after preclustering and merging
};

using namespace std;

//_________________________________________________________________________________________________
PreClusterFinder::PreClusterFinder() : mDEs{}
{
  /// default constructor: prepare the internal mapping structures
  for (auto i = 0; i < SNDEs; i++) {
    mDEs.emplace_back(new DetectionElement);
  }
}

//_________________________________________________________________________________________________
PreClusterFinder::~PreClusterFinder() = default;

//_________________________________________________________________________________________________
void PreClusterFinder::init()
{
  /// load the mapping and fill the internal structures

  createMapping();

  for (int iDE = 0; iDE < SNDEs; ++iDE) {
    for (int iPlane = 0; iPlane < 2; ++iPlane) {
      mPreClusters[iDE][iPlane].reserve(100);
    }
  }
}

//_________________________________________________________________________________________________
void PreClusterFinder::deinit()
{
  /// clear the internal structure
  reset();
  mDEIndices.clear();
}

//_________________________________________________________________________________________________
void PreClusterFinder::reset()
{
  /// reset fired pad and precluster information
  for (int iDE = 0; iDE < SNDEs; ++iDE) {
    reset(iDE);
  }
}

//_________________________________________________________________________________________________
void PreClusterFinder::reset(int deIndex)
{
  /// reset fired pad and precluster information of this DE

  Mapping::MpPad* pad(nullptr);
  DetectionElement& de(*(mDEs[deIndex]));

  // loop over planes
  for (int iPlane = 0; iPlane < 2; ++iPlane) {

    // clear number of preclusters
    mNPreClusters[deIndex][iPlane] = 0;

    // loop over fired pads
    for (int iFiredPad = 0; iFiredPad < de.nFiredPads[iPlane]; ++iFiredPad) {

      pad = &de.mapping->pads[de.firedPads[iPlane][iFiredPad]];
      pad->iDigit = 0;
      pad->useMe = false;
    }

    // clear number of fired pads
    de.nFiredPads[iPlane] = 0;
  }

  // clear ordered number of fired pads
  de.nOrderedPads[0] = 0;
  de.nOrderedPads[1] = 0;
}

//_________________________________________________________________________________________________
void PreClusterFinder::loadDigits(gsl::span<const Digit> digits)
{
  /// fill the Mapping::MpDE structure with fired pads
  for (const auto& digit : digits) {
    loadDigit(digit);
  }
}

//_________________________________________________________________________________________________
void PreClusterFinder::loadDigit(const Digit& digit)
{
  /// fill the Mapping::MpDE structure with fired pad

  int deIndex = mDEIndices[digit.getDetID()];
  assert(deIndex >= 0 && deIndex < SNDEs);

  DetectionElement& de(*(mDEs[deIndex]));

  uint16_t iPad = digit.getPadID();
  int iPlane = (iPad < de.mapping->nPads[0]) ? 0 : 1;

  // register this digit
  uint16_t iDigit = de.nFiredPads[0] + de.nFiredPads[1];
  if (iDigit >= de.digits.size()) {
    de.digits.push_back(&digit);
  } else {
    de.digits[iDigit] = &digit;
  }
  de.mapping->pads[iPad].iDigit = iDigit;
  de.mapping->pads[iPad].useMe = true;

  // set this pad as fired
  if (de.nFiredPads[iPlane] < de.firedPads[iPlane].size()) {
    de.firedPads[iPlane][de.nFiredPads[iPlane]] = iPad;
  } else {
    de.firedPads[iPlane].push_back(iPad);
  }
  ++de.nFiredPads[iPlane];
}

//_________________________________________________________________________________________________
int PreClusterFinder::discardHighOccupancy(bool perDE, bool perEvent)
{
  /// discard high-occupancy (noisy) DE and/or event

  static constexpr double maxOccupancy = 0.2;
  static constexpr int maxHighOccupancyDE = 4;

  if (!perDE && !perEvent) {
    return 0;
  }

  // discard DE with high occupancy in either bending or non-bending plane on stations 3-4-5
  int nDigits(0);
  int nRemovedDigits(0);
  int nHighOccupancyDE(0);
  for (int iDE = 0; iDE < SNDEs; ++iDE) {
    DetectionElement& de(*(mDEs[iDE]));
    nDigits += de.nFiredPads[0] + de.nFiredPads[1];
    if (de.mapping->uid >= 500 &&
        (de.nFiredPads[0] > maxOccupancy * de.mapping->nPads[0] ||
         de.nFiredPads[1] > maxOccupancy * de.mapping->nPads[1])) {
      ++nHighOccupancyDE;
      if (perDE) {
        nRemovedDigits += de.nFiredPads[0] + de.nFiredPads[1];
        reset(iDE);
      }
    }
  }

  // discard events with too many high-occupancy DE
  if (perEvent && nHighOccupancyDE > maxHighOccupancyDE) {
    nRemovedDigits = nDigits;
    reset();
  }

  return nRemovedDigits;
}

//_________________________________________________________________________________________________
int PreClusterFinder::run()
{
  /// preclusterize each cathod separately then merge them
  preClusterizeRecursive();
  return mergePreClusters();
}

//_________________________________________________________________________________________________
void PreClusterFinder::getPreClusters(std::vector<o2::mch::PreCluster>& preClusters, std::vector<Digit>& digits)
{
  /// add the preclusters and associated digits at the end of the input vectors
  /// the existing preclusters and digits are not touched, so the corresponding indices are preserved
  /// however, iterators, pointers and references might be invalidated in case the vectors are resized

  for (int iDE = 0, nDEs = SNDEs; iDE < nDEs; ++iDE) {

    DetectionElement& de(*(mDEs[iDE]));
    if (de.nOrderedPads[1] == 0) {
      continue;
    }

    for (int iPlane = 0; iPlane < 2; ++iPlane) {
      for (int iCluster = 0; iCluster < mNPreClusters[iDE][iPlane]; ++iCluster) {

        PreCluster* cluster = mPreClusters[iDE][iPlane][iCluster].get();
        if (!cluster->storeMe) {
          continue;
        }

        // add this precluster
        uint32_t firstDigit = digits.size();
        uint32_t nDigits = cluster->lastPad - cluster->firstPad + 1;
        preClusters.push_back({firstDigit, nDigits});

        // add the digits of this precluster
        for (uint16_t iOrderedPad = cluster->firstPad; iOrderedPad <= cluster->lastPad; ++iOrderedPad) {
          digits.emplace_back(*de.digits[de.mapping->pads[de.orderedPads[1][iOrderedPad]].iDigit]);
        }
      }
    }
  }
}

//_________________________________________________________________________________________________
void PreClusterFinder::preClusterizeRecursive()
{
  /// preclusterize both planes of every DE using recursive algorithm

  PreCluster* cluster(nullptr);
  uint16_t iPad(0);

  // loop over DEs
  for (int iDE = 0; iDE < SNDEs; ++iDE) {

    DetectionElement& de(*(mDEs[iDE]));

    // loop over planes
    for (int iPlane = 0; iPlane < 2; ++iPlane) {

      // loop over fired pads
      for (int iFiredPad = 0; iFiredPad < de.nFiredPads[iPlane]; ++iFiredPad) {

        iPad = de.firedPads[iPlane][iFiredPad];

        if (de.mapping->pads[iPad].useMe) {

          // create the precluster if needed
          if (mNPreClusters[iDE][iPlane] >= mPreClusters[iDE][iPlane].size()) {
            mPreClusters[iDE][iPlane].push_back(std::make_unique<PreCluster>());
          }

          // get the precluster
          cluster = mPreClusters[iDE][iPlane][mNPreClusters[iDE][iPlane]].get();
          ++mNPreClusters[iDE][iPlane];

          // reset its content
          cluster->area[0][0] = 1.e6;
          cluster->area[0][1] = -1.e6;
          cluster->area[1][0] = 1.e6;
          cluster->area[1][1] = -1.e6;
          cluster->useMe = true;
          cluster->storeMe = false;

          // add the pad and its fired neighbours recusively
          cluster->firstPad = de.nOrderedPads[0];
          addPad(de, iPad, *cluster);
        }
      }
    }
  }
}

//_________________________________________________________________________________________________
void PreClusterFinder::addPad(DetectionElement& de, uint16_t iPad, PreCluster& cluster)
{
  /// add the given MpPad and its fired neighbours (recursive method)

  Mapping::MpPad* pads(de.mapping->pads.get());

  // add the given pad
  Mapping::MpPad& pad(pads[iPad]);
  if (de.nOrderedPads[0] < de.orderedPads[0].size()) {
    de.orderedPads[0][de.nOrderedPads[0]] = iPad;
  } else {
    de.orderedPads[0].push_back(iPad);
  }
  cluster.lastPad = de.nOrderedPads[0];
  ++de.nOrderedPads[0];
  if (pad.area[0][0] < cluster.area[0][0]) {
    cluster.area[0][0] = pad.area[0][0];
  }
  if (pad.area[0][1] > cluster.area[0][1]) {
    cluster.area[0][1] = pad.area[0][1];
  }
  if (pad.area[1][0] < cluster.area[1][0]) {
    cluster.area[1][0] = pad.area[1][0];
  }
  if (pad.area[1][1] > cluster.area[1][1]) {
    cluster.area[1][1] = pad.area[1][1];
  }

  pad.useMe = false;

  // loop over its neighbours
  for (int iNeighbour = 0; iNeighbour < pad.nNeighbours; ++iNeighbour) {

    if (pads[pad.neighbours[iNeighbour]].useMe) {

      // add the pad to the precluster
      addPad(de, pad.neighbours[iNeighbour], cluster);
    }
  }
}

//_________________________________________________________________________________________________
int PreClusterFinder::mergePreClusters()
{
  /// merge overlapping preclusters on every DE
  /// return the total number of preclusters after merging

  PreCluster* cluster(nullptr);
  int nPreClusters(0);

  // loop over DEs
  for (int iDE = 0; iDE < SNDEs; ++iDE) {

    DetectionElement& de(*(mDEs[iDE]));

    // loop over preclusters of one plane
    for (int iCluster = 0; iCluster < mNPreClusters[iDE][0]; ++iCluster) {

      if (!mPreClusters[iDE][0][iCluster]->useMe) {
        continue;
      }

      cluster = mPreClusters[iDE][0][iCluster].get();
      cluster->useMe = false;

      // look for overlapping preclusters in the other plane
      PreCluster* mergedCluster(nullptr);
      mergePreClusters(*cluster, mPreClusters[iDE], mNPreClusters[iDE], de, 1, mergedCluster);

      // add the current one
      if (!mergedCluster) {
        mergedCluster = usePreClusters(cluster, de);
      } else {
        mergePreClusters(*mergedCluster, *cluster, de);
      }

      ++nPreClusters;
    }

    // loop over preclusters of the other plane
    for (int iCluster = 0; iCluster < mNPreClusters[iDE][1]; ++iCluster) {

      if (!mPreClusters[iDE][1][iCluster]->useMe) {
        continue;
      }

      // all remaining preclusters have to be stored
      usePreClusters(mPreClusters[iDE][1][iCluster].get(), de);

      ++nPreClusters;
    }
  }

  return nPreClusters;
}

//_________________________________________________________________________________________________
void PreClusterFinder::mergePreClusters(PreCluster& cluster, std::vector<std::unique_ptr<PreCluster>> preClusters[2],
                                        int nPreClusters[2], DetectionElement& de, int iPlane,
                                        PreCluster*& mergedCluster)
{
  /// merge preclusters on the given plane overlapping with the given one (recursive method)

  // overlap precision in cm: positive(negative) = increase(decrease) precluster size
  constexpr float overlapPrecision = -1.e-4;

  PreCluster* cluster2(nullptr);

  // loop over preclusters in the given plane
  for (int iCluster = 0; iCluster < nPreClusters[iPlane]; ++iCluster) {

    if (!preClusters[iPlane][iCluster]->useMe) {
      continue;
    }

    cluster2 = preClusters[iPlane][iCluster].get();
    if (Mapping::areOverlapping(cluster.area, cluster2->area, overlapPrecision) &&
        areOverlapping(cluster, *cluster2, de, overlapPrecision)) {

      cluster2->useMe = false;

      // look for new overlapping preclusters in the other plane
      mergePreClusters(*cluster2, preClusters, nPreClusters, de, (iPlane + 1) % 2, mergedCluster);

      // store overlapping preclusters and merge them
      if (!mergedCluster) {
        mergedCluster = usePreClusters(cluster2, de);
      } else {
        mergePreClusters(*mergedCluster, *cluster2, de);
      }
    }
  }
}

//_________________________________________________________________________________________________
PreClusterFinder::PreCluster* PreClusterFinder::usePreClusters(PreCluster* cluster, DetectionElement& de)
{
  /// use this precluster as a new merged precluster

  uint16_t firstPad = de.nOrderedPads[1];

  // move the fired pads
  for (int iOrderPad = cluster->firstPad; iOrderPad <= cluster->lastPad; ++iOrderPad) {

    if (de.nOrderedPads[1] < de.orderedPads[1].size()) {
      de.orderedPads[1][de.nOrderedPads[1]] = de.orderedPads[0][iOrderPad];
    } else {
      de.orderedPads[1].push_back(de.orderedPads[0][iOrderPad]);
    }

    ++de.nOrderedPads[1];
  }

  cluster->firstPad = firstPad;
  cluster->lastPad = de.nOrderedPads[1] - 1;

  cluster->storeMe = true;

  return cluster;
}

//_________________________________________________________________________________________________
void PreClusterFinder::mergePreClusters(PreCluster& cluster1, PreCluster& cluster2, DetectionElement& de)
{
  /// merge precluster2 into precluster1

  // move the fired pads
  for (int iOrderPad = cluster2.firstPad; iOrderPad <= cluster2.lastPad; ++iOrderPad) {

    if (de.nOrderedPads[1] < de.orderedPads[1].size()) {
      de.orderedPads[1][de.nOrderedPads[1]] = de.orderedPads[0][iOrderPad];
    } else {
      de.orderedPads[1].push_back(de.orderedPads[0][iOrderPad]);
    }

    ++de.nOrderedPads[1];
  }

  cluster1.lastPad = de.nOrderedPads[1] - 1;
}

//_________________________________________________________________________________________________
bool PreClusterFinder::areOverlapping(PreCluster& cluster1, PreCluster& cluster2, DetectionElement& de, float precision)
{
  /// check if the two preclusters overlap
  /// precision in cm: positive = increase pad size / negative = decrease pad size

  // loop over all pads of the precluster1
  for (int iOrderPad1 = cluster1.firstPad; iOrderPad1 <= cluster1.lastPad; ++iOrderPad1) {

    // loop over all pads of the precluster2
    for (int iOrderPad2 = cluster2.firstPad; iOrderPad2 <= cluster2.lastPad; ++iOrderPad2) {

      if (Mapping::areOverlapping(de.mapping->pads[de.orderedPads[0][iOrderPad1]].area,
                                  de.mapping->pads[de.orderedPads[0][iOrderPad2]].area, precision)) {
        return true;
      }
    }
  }

  return false;
}

//_________________________________________________________________________________________________
void PreClusterFinder::createMapping()
{
  /// Fill the internal mapping structures

  auto tStart = std::chrono::high_resolution_clock::now();

  std::vector<std::unique_ptr<Mapping::MpDE>> mpDEs = Mapping::createMapping();

  if (mpDEs.size() != SNDEs) {
    throw runtime_error("invalid mapping");
  }

  mDEIndices.reserve(SNDEs);

  for (int iDE = 0; iDE < SNDEs; ++iDE) {

    DetectionElement& de(*(mDEs[iDE]));

    de.mapping = std::move(mpDEs[iDE]);

    mDEIndices.emplace(de.mapping->uid, iDE);

    int initialSize = (de.mapping->nPads[0] / 10 + de.mapping->nPads[1] / 10); // 10 % occupancy

    de.digits.reserve(initialSize);
    de.nOrderedPads[0] = 0;
    de.orderedPads[0].reserve(initialSize);
    de.nOrderedPads[1] = 0;
    de.orderedPads[1].reserve(initialSize);

    for (int iPlane = 0; iPlane < 2; ++iPlane) {
      de.nFiredPads[iPlane] = 0;
      de.firedPads[iPlane].reserve(de.mapping->nPads[iPlane] / 10); // 10% occupancy
    }
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "create mapping in: " << std::chrono::duration<double, std::milli>(tEnd - tStart).count() << " ms";
}

} // namespace mch
} // namespace o2
