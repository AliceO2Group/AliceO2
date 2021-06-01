// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <FairMQLogger.h>

#include "PadOriginal.h"
#include "ClusterOriginal.h"
/* Inv ???
#include "MathiesonOriginal.h"
*/
// GG
#include "mathiesonFit.h"

namespace o2
{
namespace mch
{
    
//_________________________________________________________________________________________________
ClusterFinderGEM::ClusterFinderGEM()
   : mPreCluster(std::make_unique<ClusterOriginal>())
/*
  : mMathiesons(std::make_unique<MathiesonOriginal[]>(2)),
*/
{
  /// default constructor

  // Mathieson function for station 1
  /* Inv ???
  mMathiesons[0].setPitch(0.21);
  mMathiesons[0].setSqrtKx3AndDeriveKx2Kx4(0.7000);
  mMathiesons[0].setSqrtKy3AndDeriveKy2Ky4(0.7550);

  // Mathieson function for other stations
  mMathiesons[1].setPitch(0.25);
  mMathiesons[1].setSqrtKx3AndDeriveKx2Kx4(0.7131);
  mMathiesons[1].setSqrtKy3AndDeriveKy2Ky4(0.7642);
  */
  // GG
  // init Mathieson
  initMathieson();
  nPads = 0;
  xyDxy=0;
  cathode=0;
  saturated=0;
  padCharge=0;
  DEId=-1;
  // GG Init/open dump files
  pClusterDump = new ClusterDump("EMRun2.dat", 0);
  currentBC    = 0xFFFFFFFF;
  currentOrbit = 0xFFFFFFFF;
  currentPreClusterID=0;
  
}

//_________________________________________________________________________________________________
// GG ClusterFinderGEM::~ClusterFinderGEM() = default;
ClusterFinderGEM::~ClusterFinderGEM() {
  delete pClusterDump;
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::init()
{
  /// initialize the clustering
  // GG before to start the workflow
  mPreClusterFinder.init();
 }

//_________________________________________________________________________________________________
void ClusterFinderGEM::deinit()
{
  /// deinitialize the clustering
  /// GG in case of errore in the workflow
  mPreClusterFinder.deinit();
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::reset()
{
  /// reset the list of reconstructed clusters and associated digits
  // GG Between 2 diff. time frames
  mClusters.clear();
  mUsedDigits.clear();
  // GG
  nPads = 0;
  DEId = -1;
  if ( xyDxy != 0)     { delete[] xyDxy;     xyDxy=0; };
  if ( cathode != 0)   { delete[] cathode;   cathode=0; };
  if ( padCharge != 0) { delete[] padCharge; padCharge=0; };
  if ( saturated != 0) { delete[] saturated; saturated=0; };
  freeMemoryPadProcessing();
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::initPreCluster(gsl::span<const Digit>& digits, uint16_t bunchCrossing, uint32_t orbit, uint32_t iROF)
{
  /// reset the precluster with the pads converted from the input digits
  // GG mPrecluster defined in other MCHPreClustering/PreClusterFinder.h
  mPreCluster->clear();
  mSegmentation = &mapping::segmentation(digits[0].getDetID());

  // GG 
  // Allocation
  nPads = digits.size();
  xyDxy = new double[nPads*4];
  saturated = new Mask_t[nPads];
  cathode   = new Mask_t[nPads];
  padCharge = new double[nPads];
  // use to store in the dump file
  uint32_t padADC[nPads];
  // std::cout << "DeId: " << digits[0].getDetID() << std::endl;  
  DEId = digits[0].getDetID();
  double *xPad = getX( xyDxy, nPads);
  double *yPad = getY( xyDxy, nPads);
  double *dxPad = getDX( xyDxy, nPads);
  double *dyPad = getDY( xyDxy, nPads);
  
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
    xPad[iDigit] = x; yPad[iDigit] = y; dxPad[iDigit] = dx; dyPad[iDigit] = dy;
    padCharge[iDigit] = charge;
    saturated[iDigit] = isSaturated;
    cathode[iDigit] = plane;
    padADC[iDigit] = adc;
  }
  // GG Store in dump file
  uint32_t N = digits.size();
  if ( digits.size() != 0) {
    // Replace 0, by 0xFFFFFFFF
    uint32_t header[6] = { (uint32_t) (bunchCrossing),  (orbit), iROF, (0), N, (uint32_t) (DEId)};
    pClusterDump->dumpUInt32(0, 6, header );
    pClusterDump->dumpFloat64(0, N, xPad);    
    pClusterDump->dumpFloat64(0, N, yPad);
    pClusterDump->dumpFloat64(0, N, dxPad);
    pClusterDump->dumpFloat64(0, N, dyPad);
    pClusterDump->dumpFloat64(0, N, padCharge );
    pClusterDump->dumpInt16(0, N, saturated );
    pClusterDump->dumpInt16(0, N, cathode );
    pClusterDump->dumpUInt32(0, N, padADC );
  }
     
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::findClusters( gsl::span<const Digit> digits, 
                                     uint16_t bunchCrossing, uint32_t orbit, uint32_t iROF, bool samePreCluster)
{
  /// reconstruct the clusters from the list of digits of one precluster
  /// reconstructed clusters and associated digits are added to the internal lists
  std::cout << "New preCluster size=" << digits.size() << std::endl;
  // skip preclusters with only 1 digit
  if (digits.size() < 2) {
    return;
  }

  // set the Mathieson function to be used
  // GG Inv ??? mMathieson = (digits[0].getDetID() < 300) ? &mMathiesons[0] : &mMathiesons[1];

  // reset the current precluster being processed
  // GG Set the Precluster (pads , ....)
  initPreCluster(digits, bunchCrossing, orbit, iROF);
  // std::cout << "Init preCluster " << digits.size() << std::endl;
  
  // GG process clusters
  int chId = DEId / 100;
  int nbrOfHits = clusterProcess( xyDxy, cathode, saturated, padCharge, chId, nPads);
  double theta[nbrOfHits*5];
  Group_t thetaToGroup[nbrOfHits];
  collectTheta( theta, thetaToGroup, nbrOfHits);
  std::cout << "Seeds found by GEM " << nbrOfHits << " / nPads = " << nPads << std::endl;
  double *muX = getMuX( theta, nbrOfHits);
  double *muY = getMuY( theta, nbrOfHits);
  double *w = getW( theta, nbrOfHits);
  // Find subClusters with there seeds
  // 
  
  /*
  Loop sur les groupes >= 1
   voir si groupe associé au cathodes marchent bien et voir ordre
   append les usedDigits du groupe
   mUsedDigits.emplace_back( digits[ idPad] );
   next groupe
  */
  Group_t padToCathGrp[nPads];
  collectPadToCathGroup( padToCathGrp, nPads );
  // GG ???????????????????
  // Take care the number of groups can be !=
  // between thetaToGroup
  Group_t nCathGrp = vectorMaxShort( padToCathGrp, nPads);
  int nPadStored = 0;
  // Index of the first store digit of the group
  uint32_t firstDigit;
  // For all the cath groups
  for (int g = 1; g < (nCathGrp+1); g++) {
    int i=0;    
    firstDigit = mUsedDigits.size();
    for (const auto& pad : *mPreCluster) {
      // Store the pad belonging to the cath-group
      if( padToCathGrp[i] == g) {
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
      if ( thetaToGroup[s] == g) {
        double x  = muX[s];
        double y  = muY[s];
        // ??? Do something for the mathieson factors
        double dx, dy;
        // ??? value of chID
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
        uint32_t uid = ClusterStruct::buildUniqueId(digits[0].getDetID() / 100 - 1, digits[0].getDetID(), thetaToGroup[s]);
        mClusters.push_back({
          static_cast<float>(x),  static_cast<float>(y), 0.0, // x, y, z
          static_cast<float>(dx), static_cast<float>(dy),     // x, y resolution
          uid,                                                // uid 
          firstDigit, nDigits                                 // firstDigit, nDigits
       });
      }
    }    
  }
  ////////////////////////////
  /* Invalid
  // Original version
  std::cout << "Results preCluster " << digits.size() << std::endl;
  int nGroups = 1;
  int nSeeds[nGroups] = {1};
  for (int g = 0; g < nGroups; g++) {
    uint32_t firstDigit = 0;
    // Copy digit used in the cluster g
    for (const auto& pad : *mPreCluster) {
      if (pad.isReal()) {
        mUsedDigits.emplace_back(digits[pad.digitIndex()]);
      }
    }    
    uint32_t nDigits = mUsedDigits.size() - firstDigit;
    for (int s = 0; s < nSeeds[g]; s++) {
      double x  = mPreCluster->pad(0).x();
      double y  = mPreCluster->pad(0).y();
      double dx = mPreCluster->pad(0).dx();
      double dy = mPreCluster->pad(0).dy();
      int seedId = s;
      uint32_t uid = ClusterStruct::buildUniqueId(digits[0].getDetID() / 100 - 1, digits[0].getDetID(), seedId);
      mClusters.push_back({
        static_cast<float>(x),  static_cast<float>(y), 0.0, // x, y, z
        static_cast<float>(dx), static_cast<float>(dy),     // x, y resolution
        uid,                                                // uid 
        firstDigit, nDigits                                 // firstDigit, nDigits
       });
    }
  }
  */
  std::cout << "Finished preCluster " << digits.size() << std::endl;

}



//_________________________________________________________________________________________________
void ClusterFinderGEM::processPreCluster()
{
  /// builds an array of pixel and extract clusters from it
  std::cout << "Test processPreCluster GG: to remove ???" << std::endl; 
}



} // namespace mch
} // namespace o2
