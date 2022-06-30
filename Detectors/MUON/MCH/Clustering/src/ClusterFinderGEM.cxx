// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFinderGEM.cxx
/// \brief Definition of a class to reconstruct clusters with the original MLEM algorithm
///
/// The original code is in AliMUONClusterFinderMLEM and associated classes.
/// It has been re-written in an attempt to simplify it without changing the results.
///
/// \author Philippe Pillot, Subatech

#include "MCHClustering/ClusterFinderGEM.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

#include <TH2I.h>
#include <TAxis.h>
#include <TMath.h>
#include <TRandom.h>

#include <FairLogger.h>

// GG
#include "PadOriginal.h"
#include "ClusterOriginal.h"
#include "MCHBase/MathiesonOriginal.h"
#include "mathiesonFit.h"

#define VERBOSE 0

namespace o2
{
namespace mch
{

//_________________________________________________________________________________________________
ClusterFinderGEM::ClusterFinderGEM()
  : mMathiesons(std::make_unique<MathiesonOriginal[]>(2)), mPreCluster(std::make_unique<ClusterOriginal>())
{
  /// default constructor

  // Mathieson function for station 1
  mMathiesons[0].setPitch(0.21);
  mMathiesons[0].setSqrtKx3AndDeriveKx2Kx4(0.7000);
  mMathiesons[0].setSqrtKy3AndDeriveKy2Ky4(0.7550);

  // Mathieson function for other stations
  mMathiesons[1].setPitch(0.25);
  mMathiesons[1].setSqrtKx3AndDeriveKx2Kx4(0.7131);
  mMathiesons[1].setSqrtKy3AndDeriveKy2Ky4(0.7642);
  // GG
  // init Mathieson
  initMathieson();
  nPads = 0;
  xyDxy = nullptr;
  cathode = nullptr;
  saturated = nullptr;
  padCharge = nullptr;
  DEId = -1;
  currentBC = 0xFFFFFFFF;
  currentOrbit = 0xFFFFFFFF;
  currentPreClusterID = 0;
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::init(int _mode)
{
  /// initialize the clustering
  // ??? Not used
  mode = _mode;
}
//_________________________________________________________________________________________________
void ClusterFinderGEM::deinit()
{
  // std::cout << "  [GEM] deinit " << std::endl;
  /// deinitialize the Original clustering
}

//_________________________________________________________________________________________________
ClusterFinderGEM::~ClusterFinderGEM() = default;
/*
ClusterFinderGEM::~ClusterFinderGEM()
{
  // std::cout << "  [GEM] Delete " << std::endl;
  // GG invalid
  if ( pOriginalClusterDump != 0) {
    delete [] pOriginalClusterDump;
  }
  if ( pGEMClusterDump != 0) {
    delete [] pGEMClusterDump;
  }
}
*/

//_________________________________________________________________________________________________
void ClusterFinderGEM::reset()
{

  // std::cout << "  [GEM] Reset hits/mClusters.size=" << mClusters.size() << std::endl;

  // GEM part
  nPads = 0;
  DEId = -1;
  if (xyDxy != nullptr) {
    delete[] xyDxy;
    xyDxy = nullptr;
  };
  if (cathode != nullptr) {
    delete[] cathode;
    cathode = nullptr;
  };
  if (padCharge != nullptr) {
    delete[] padCharge;
    padCharge = nullptr;
  };
  if (saturated != nullptr) {
    delete[] saturated;
    saturated = nullptr;
  };
  // Inv ??? freeMemoryPadProcessing();
  mClusters.clear();
  mUsedDigits.clear();
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::dumpPreCluster(ClusterDump* dumpFile, gsl::span<const Digit> digits, uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster)
{
  /// reset the precluster with the pads converted from the input digits
  // GG mPrecluster defined in other MCHPreClustering/PreClusterFinder.h
  mSegmentation = &mapping::segmentation(digits[0].getDetID());

  // GG
  // Allocation
  nPads = digits.size();
  double xyDxy[nPads * 4];
  Mask_t saturated[nPads];
  Mask_t cathode[nPads];
  double padCharge[nPads];
  // use to store in the dump file
  uint32_t padADC[nPads];
  DEId = digits[0].getDetID();
  double* xPad = getX(xyDxy, nPads);
  double* yPad = getY(xyDxy, nPads);
  double* dxPad = getDX(xyDxy, nPads);
  double* dyPad = getDY(xyDxy, nPads);

  for (int iDigit = 0; iDigit < digits.size(); ++iDigit) {
    const auto& digit = digits[iDigit];
    int padID = digit.getPadID();

    double x = mSegmentation->padPositionX(padID);
    double y = mSegmentation->padPositionY(padID);
    double dx = mSegmentation->padSizeX(padID) / 2.;
    double dy = mSegmentation->padSizeY(padID) / 2.;
    uint32_t adc = digit.getADC();
    float charge(0.);
    std::memcpy(&charge, &adc, sizeof(adc));
    bool isSaturated = digit.isSaturated();
    int plane = mSegmentation->isBendingPad(padID) ? 0 : 1;

    if (charge <= 0.) {
      throw std::runtime_error("The precluster contains a digit with charge <= 0");
    }

    // GG
    // Initialisation for GEM processing
    xPad[iDigit] = x;
    yPad[iDigit] = y;
    dxPad[iDigit] = dx;
    dyPad[iDigit] = dy;
    padCharge[iDigit] = charge;
    saturated[iDigit] = isSaturated;
    cathode[iDigit] = plane;
    padADC[iDigit] = adc;
  }
  // GG Store in dump file
  uint32_t N = digits.size();
  // std::cout << "  [GEM] Dump PreCluster " << dumpFile->getName() << ", size=" << N << std::endl;

  if ((N != 0)) {
    // Replace 0, by 0xFFFFFFFF
    uint32_t header[6] = {(uint32_t)(bunchCrossing), (orbit), iPreCluster, (0), N, (uint32_t)(DEId)};
    dumpFile->dumpUInt32(0, 6, header);
    dumpFile->dumpFloat64(0, N, xPad);
    dumpFile->dumpFloat64(0, N, yPad);
    dumpFile->dumpFloat64(0, N, dxPad);
    dumpFile->dumpFloat64(0, N, dyPad);
    dumpFile->dumpFloat64(0, N, padCharge);
    dumpFile->dumpInt16(0, N, saturated);
    dumpFile->dumpInt16(0, N, cathode);
    dumpFile->dumpUInt32(0, N, padADC);
  }
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::dumpClusterResults(ClusterDump* dumpFile, const std::vector<Cluster>& clusters, size_t startIdx, uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster)
{
  // Dump result
  // Dump hits from iNewCluster to mClusters.size()
  uint32_t N = clusters.size() - startIdx;
  // struct ClusterStruct {
  //  float x;             ///< cluster position along x
  //  float y;             ///< cluster position along y
  //  float z;             ///< cluster position along z
  //  float ex;            ///< cluster resolution along x
  //  float ey;            ///< cluster resolution along y
  //  uint32_t uid;        ///< cluster unique ID
  //  uint32_t firstDigit; ///< index of first associated digit in the ordered vector of digits
  //  uint32_t nDigits;    ///< number of digits attached to this cluster
  //
  // Header
  // std::cout << "  [GEM] Dump Cluster " << dumpFile->getName() << ", size=" << N << " start=" << startIdx << std::endl;

  uint32_t header[6] = {(uint32_t)(bunchCrossing), (orbit), iPreCluster, (0), N, 0};
  dumpFile->dumpUInt32(0, 6, header);
  //
  double x[N], y[N];
  double ex[N], ey[N];
  uint32_t uid[N];
  uint32_t firstDigit[N], nDigits[N];
  int i = 0;
  for (int n = startIdx; n < clusters.size(); ++n, i++) {
    Cluster hit = clusters[n];
    x[i] = hit.x;
    y[i] = hit.y;
    ex[i] = hit.ex;
    ey[i] = hit.ey;
    uid[i] = hit.uid;
    firstDigit[i] = hit.firstDigit;
    nDigits[i] = hit.nDigits;
  }
  if (N > 0) {
    dumpFile->dumpFloat64(0, N, x);
    dumpFile->dumpFloat64(0, N, y);
    dumpFile->dumpFloat64(0, N, ex);
    dumpFile->dumpFloat64(0, N, ey);
    dumpFile->dumpUInt32(0, N, uid);
    dumpFile->dumpUInt32(0, N, firstDigit);
    dumpFile->dumpUInt32(0, N, nDigits);
    // dumpFile->flush();
  }
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::fillGEMInputData(gsl::span<const Digit>& digits, uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster)
{
  /// reset the precluster with the pads converted from the input digits
  // GG mPrecluster defined in other MCHPreClustering/PreClusterFinder.h
  mPreCluster->clear();
  mSegmentation = &mapping::segmentation(digits[0].getDetID());

  // GG
  // Allocation
  nPads = digits.size();
  xyDxy = new double[nPads * 4];
  saturated = new Mask_t[nPads];
  cathode = new Mask_t[nPads];
  padCharge = new double[nPads];
  // use to store in the dump file
  uint32_t padADC[nPads];
  DEId = digits[0].getDetID();
  double* xPad = getX(xyDxy, nPads);
  double* yPad = getY(xyDxy, nPads);
  double* dxPad = getDX(xyDxy, nPads);
  double* dyPad = getDY(xyDxy, nPads);

  for (int iDigit = 0; iDigit < digits.size(); ++iDigit) {
    const auto& digit = digits[iDigit];
    int padID = digit.getPadID();

    double x = mSegmentation->padPositionX(padID);
    double y = mSegmentation->padPositionY(padID);
    double dx = mSegmentation->padSizeX(padID) / 2.;
    double dy = mSegmentation->padSizeY(padID) / 2.;
    uint32_t adc = digit.getADC();
    float charge(0.);
    std::memcpy(&charge, &adc, sizeof(adc));
    bool isSaturated = digit.isSaturated();
    int plane = mSegmentation->isBendingPad(padID) ? 0 : 1;

    if (charge <= 0.) {
      throw std::runtime_error("The precluster contains a digit with charge <= 0");
    }

    mPreCluster->addPad(x, y, dx, dy, charge, isSaturated, plane, iDigit, PadOriginal::kZero);
    // GG
    // Initialisation for GEM processing
    xPad[iDigit] = x;
    yPad[iDigit] = y;
    dxPad[iDigit] = dx;
    dyPad[iDigit] = dy;
    padCharge[iDigit] = charge;
    saturated[iDigit] = isSaturated;
    cathode[iDigit] = plane;
    padADC[iDigit] = adc;
  }
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::setClusterResolution(Cluster& cluster) const
{
  /// set the cluster resolution in both directions depending on whether its position
  /// lies on top of a fired digit in both planes or not (e.g. mono-cathode)

  if (cluster.getChamberId() < 4) {

    // do not consider mono-cathode clusters in stations 1 and 2
    cluster.ex = SDefaultClusterResolution;
    cluster.ey = SDefaultClusterResolution;

  } else {

    // find pads below the cluster
    int padIDNB(-1), padIDB(-1);
    bool padsFound = mSegmentation->findPadPairByPosition(cluster.x, cluster.y, padIDB, padIDNB);

    // look for these pads (if any) in the list of digits associated to this cluster
    auto itPadNB = mUsedDigits.end();
    if (padsFound || mSegmentation->isValid(padIDNB)) {
      itPadNB = std::find_if(mUsedDigits.begin() + cluster.firstDigit, mUsedDigits.end(),
                             [padIDNB](const Digit& digit) { return digit.getPadID() == padIDNB; });
    }
    auto itPadB = mUsedDigits.end();
    if (padsFound || mSegmentation->isValid(padIDB)) {
      itPadB = std::find_if(mUsedDigits.begin() + cluster.firstDigit, mUsedDigits.end(),
                            [padIDB](const Digit& digit) { return digit.getPadID() == padIDB; });
    }

    // set the cluster resolution accordingly
    cluster.ex = (itPadNB == mUsedDigits.end()) ? SBadClusterResolution : SDefaultClusterResolution;
    cluster.ey = (itPadB == mUsedDigits.end()) ? SBadClusterResolution : SDefaultClusterResolution;
  }
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::findClusters(gsl::span<const Digit> digits,
                                    uint16_t bunchCrossing, uint32_t orbit, uint32_t iPreCluster)
{
  /// reconstruct the clusters from the list of digits of one precluster
  /// reconstructed clusters and associated digits are added to the internal lists
  // std::cout << "  [GEM] preCluster digit size=" << digits.size() << std::endl;
  // skip preclusters with only 1 digit
  if (digits.size() < 2) {
    return;
  }
  uint32_t nPreviousCluster = mClusters.size();
  if (VERBOSE > 0) {
    printf("----------------------------------------\n");
    std::cout << "  [GEM] PreCluster BC=" << bunchCrossing
              << ", orbit = " << orbit
              << "iPreCluster,  = " << iPreCluster
              << std::endl;
    printf("----------------------------------------\n");
  }
  // std::cout << "  [GEM] hits/mClusters.size=" << mClusters.size() << std::endl;

  // set the Mathieson function to be used
  // GG Inv  mMathieson = (digits[0].getDetID() < 300) ? &mMathiesons[0] : &mMathiesons[1];

  // reset the current precluster being processed
  // GG Set the Precluster (pads , ....)
  fillGEMInputData(digits, bunchCrossing, orbit, iPreCluster);

  // GG process clusters
  int chId = DEId / 100;
  int nbrOfHits = clusterProcess(xyDxy, cathode, saturated, padCharge, chId, nPads);
  double theta[nbrOfHits * 5];
  Group_t thetaToGroup[nbrOfHits];
  collectTheta(theta, thetaToGroup, nbrOfHits);
  // std::cout << "  [GEM] Seeds found by GEM " << nbrOfHits << " / nPads = " << nPads << std::endl;
  double* muX = getMuX(theta, nbrOfHits);
  double* muY = getMuY(theta, nbrOfHits);
  double* w = getW(theta, nbrOfHits);
  // Find subClusters with there seeds
  //
  Group_t padToCathGrp[nPads];
  collectPadToCathGroup(padToCathGrp, nPads);
  // Take care the number of groups can be !=
  // between thetaToGroup
  Group_t nCathGrp = vectorMaxShort(padToCathGrp, nPads);
  int nPadStored = 0;
  // Index of the first store digit of the group
  uint32_t firstDigit;
  // For all the cath groups
  for (int g = 1; g < (nCathGrp + 1); g++) {
    int i = 0;
    firstDigit = mUsedDigits.size();
    for (const auto& pad : *mPreCluster) {
      // Store the pad belonging to the cath-group
      if (padToCathGrp[i] == g) {
        mUsedDigits.emplace_back(digits[pad.digitIndex()]);
        nPadStored++;
      }
      i++;
    }
    // n digits for the considered group
    uint32_t nDigits = mUsedDigits.size() - firstDigit;
    // For all the seeds or hits
    for (int s = 0; s < nbrOfHits; s++) {
      // Store the hit belonging to the cath-group
      if (thetaToGroup[s] == g) {
        double x = muX[s];
        double y = muY[s];
        /*
          // ??? Do something for the mathieson factors
          double dx, dy;
          // ??? value of chID
          // ??? To do later
          if (chId <= 4) {
            dx = SDefaultClusterResolution;
            dy = SDefaultClusterResolution;
          } else {
            // Find the associated pads
            // set the cluster resolution accordingly
            // cluster.ex = (NBPad) ? SBadClusterResolution : SDefaultClusterResolution;
            // cluster.ey = (BPad) ? SBadClusterResolution : SDefaultClusterResolution;
            // ???
            dx = SDefaultClusterResolution;
            dy = SDefaultClusterResolution;
          }
           */
        uint32_t uid = Cluster::buildUniqueId(digits[0].getDetID() / 100 - 1, digits[0].getDetID(), thetaToGroup[s]);
        mClusters.push_back({
          static_cast<float>(x), static_cast<float>(y), 0.0, // x, y, z
          static_cast<float>(0), static_cast<float>(0),      // x, y resolution
          uid,                                               // uid
          firstDigit, nDigits                                // firstDigit, nDigits
        });
        setClusterResolution(mClusters[mClusters.size() - 1]);
      }
    }
  }

  // std::cout << "  [GEM] Finished preCluster " << digits.size() << std::endl;
}

} // namespace mch
} // namespace o2
