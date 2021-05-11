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
}

//_________________________________________________________________________________________________
ClusterFinderGEM::~ClusterFinderGEM() = default;

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

// GG To keep, use at the workflow level
//_________________________________________________________________________________________________
void ClusterFinderGEM::reset()
{
  /// reset the list of reconstructed clusters and associated digits
  // GG Between 2 diff. time frames
  mClusters.clear();
  mUsedDigits.clear();
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::initPreCluster(gsl::span<const Digit>& digits)
{
  /// reset the precluster with the pads converted from the input digits
  // GG mPrecluster defined in other MCHPreClustering/PreClusterFinder.h
  mPreCluster->clear();

  mSegmentation = &mapping::segmentation(digits[0].getDetID());

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
  }
}

//_________________________________________________________________________________________________
void ClusterFinderGEM::findClusters(gsl::span<const Digit> digits)
{
  /// reconstruct the clusters from the list of digits of one precluster
  /// reconstructed clusters and associated digits are added to the internal lists
  std::cout << "New preCluster " << digits.size() << std::endl;
  // skip preclusters with only 1 digit
  if (digits.size() < 2) {
    return;
  }

  // set the Mathieson function to be used
  // Inv ??? mMathieson = (digits[0].getDetID() < 300) ? &mMathiesons[0] : &mMathiesons[1];

  // reset the current precluster being processed
  // GG Set the Precluster (pads , ....)
  std::cout << "Init preCluster " << digits.size() << std::endl;
  initPreCluster(digits);

  // Find subClusters with there seeds
  // 
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
  std::cout << "Finished preCluster " << digits.size() << std::endl;

}



//_________________________________________________________________________________________________
void ClusterFinderGEM::processPreCluster()
{
  /// builds an array of pixel and extract clusters from it
  std::cout << "Test GG" << std::endl; 

}



} // namespace mch
} // namespace o2
