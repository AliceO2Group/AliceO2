// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPCUpgradeHwClusterFinder.cxx
/// \brief HwClusterFinder for the TPC


#include "TPCReconstruction/HwClusterFinder.h"
#include "TPCReconstruction/ClusterContainer.h"
#include "TPCReconstruction/Cluster.h"

#include "FairLogger.h"


using namespace o2::TPC;


//________________________________________________________________________
HwClusterFinder::HwClusterFinder(
    unsigned short cru, unsigned short row,
    short padOffset, unsigned short pads, unsigned short timebins,
    float diffThreshold, float chargeThreshold,
    bool requirePositiveCharge)
  : mRequirePositiveCharge(requirePositiveCharge)
  , mRequireNeighbouringPad(false)//true)
  , mRequireNeighbouringTimebin(true)
  , mAssignChargeUnique(false)//true)
  , mPadOffset(padOffset)
  , mCRU(cru)
  , mRow(row)
  , mPads(pads)
  , mTimebins(timebins)
  , mClusterSizePads(5)
  , mClusterSizeTime(5)
  , mDiffThreshold(diffThreshold)   // not yet used
  , mChargeThreshold(chargeThreshold)
  , mGlobalTimeOfLast(0)
  , mTimebinsAfterLastProcessing(0)
  , mNextCF()
  , mData()   // couldn't find a way to initialize a vector of unique_ptr's here
  , mTmpCluster(5, std::vector<MiniDigit>(5, MiniDigit()))
  , mClusterContainer()
  , mClusterDigitIndices()
{
  if (mPads < mClusterSizePads) {
    LOG(ERROR) << "Given width in pad direction is smaller than cluster size in pad direction."
      << " Width in pad direction was increased to cluster size " << mClusterSizePads << FairLogger::endl;
    mPads = mClusterSizePads;
  }
  if (mTimebins < mClusterSizeTime) {
    LOG(ERROR) << "Given width in time direction is smaller than cluster size in time direction."
      << " Width in time direction was increased to cluster size " << mClusterSizeTime << FairLogger::endl;
    mTimebins = mClusterSizeTime;
  }

  for (short t = 0; t < mTimebins; ++t)
    mData.emplace_back(std::make_unique<std::vector<MiniDigit>>(pads, MiniDigit()));

}

//________________________________________________________________________
void HwClusterFinder::addZeroTimebin(unsigned globalTime, int length)
{
  addTimebin((*mData.begin())->begin() /* some iterator, is not used */,globalTime,length,true);
}

//________________________________________________________________________
void HwClusterFinder::printLocalStorage()
{
  for (short t = 0; t < mTimebins; ++t){
    LOG(DEBUG) << "t " << t << ":\t";
    for (auto &digi : *mData[t])
      LOG(DEBUG) << digi.charge << "\t";
    LOG(DEBUG) << FairLogger::endl;
  }
  LOG(DEBUG) << FairLogger::endl;
}

//________________________________________________________________________
bool HwClusterFinder::findCluster()
{
  mTimebinsAfterLastProcessing = 0;
  int foundNclusters = 0;

  //
  // Set region to look in for peaks, max. array size +-2 in both dimensions
  // to have enough space for 5x5 clusters with peak in inner region.
  //
  //        pMin  pMax
  //        |     |
  //        V     V
  //    o o o o o o o o  <- (mTimebins-1) for itterating through array, mGlobalTimeOfLast
  //    o o o o o o o o
  //    o o|o¯o¯o¯o|o o  <- tMax
  //    o o|o o o o|o o
  //    o o|o_o_o_o|o o  <- tMin
  //    o o o o o o o o
  //    o o o o o o o o  <- 0
  //
  //

  //
  // In time direction
  //
  short tMax = (mTimebins-1) - 2;
  short tMin = 2;

  //
  // In pad direction
  //
  int pMin = 2;
  int pMax = (mPads-1)-2;

  float qMax;
  float qTot;
  float charge;
  float meanP, meanT;
  float sigmaP, sigmaT;
  short minP, minT;
  short maxP, maxT;
  short deltaP, deltaT;
  //
  // peak finding
  //
  short t,p,tt,pp;
  for (t=tMin; t<=tMax; ++t) {
    for (p=pMin; p<=pMax; ++p) {
        //printf("t:%d, p:%d\n",t,p);
      //
      // find peak in 3x3 matrix
      //
      //    --->  pad direction
      //    o o o o o    |
      //    o i i i o    |
      //    o i C i o    V Time direction
      //    o i i i o
      //    o o o o o
      //
      if (mData[t  ]->at(p).charge < mChargeThreshold) continue;

      // Require at least one neighboring time bin with signal
      if (mRequireNeighbouringTimebin   && (mData[t-1]->at(p  ).charge <= 0 && mData[t+1]->at(p  ).charge <= 0)) continue;
      // Require at least one neighboring pad with signal
      if (mRequireNeighbouringPad       && (mData[t  ]->at(p-1).charge <= 0 && mData[t  ]->at(p+1).charge <= 0)) continue;

      // check for local maximum
      if (mData[t-1]->at(p  ).charge >=  mData[t]->at(p).charge) continue;
      if (mData[t+1]->at(p  ).charge >   mData[t]->at(p).charge) continue;
      if (mData[t  ]->at(p-1).charge >=  mData[t]->at(p).charge) continue;
      if (mData[t  ]->at(p+1).charge >   mData[t]->at(p).charge) continue;
      if (mData[t-1]->at(p-1).charge >=  mData[t]->at(p).charge) continue;
      if (mData[t+1]->at(p+1).charge >   mData[t]->at(p).charge) continue;
      if (mData[t+1]->at(p-1).charge >   mData[t]->at(p).charge) continue;
      if (mData[t-1]->at(p+1).charge >=  mData[t]->at(p).charge) continue;
//      printf("##\n");
//      printf("## cluster found at t=%d, p=%d (in row %d of CRU %d)\n",t,p,mRow,mCRU);
//      printf("##\n");
//      printCluster(t,p);

      //
      // cluster was found!!
      //

      // prepare temp storage
      for (tt=0; tt<mClusterSizeTime; ++tt) {
        for (pp=0; pp<mClusterSizePads; ++pp){
          mTmpCluster[tt][pp] = MiniDigit();
        }
      }

      //
      // Cluster peak (C) and surrounding inner 3x3 matrix (i) is always
      // used taken for the found cluster
      //
      for (tt=1; tt<4; ++tt) {
        for (pp=1; pp<4; ++pp) {
          if ( mRequirePositiveCharge && mData[t+(tt-2)]->at(p+(pp-2)).charge < 0) continue;
          mTmpCluster[tt][pp] = mData[t+(tt-2)]->at(p+(pp-2));
//          mData[t+(tt-2)][p+(pp-2)] = 0;
        }
      }

      //
      // The outer cells of the 5x5 matrix (o) are taken only if the
      // neighboring inner cell (i) has a signal above threshold.
      //

      //
      // The cells of the "inner cross" have here only 1 neighbour.
      // [t]
      //  0         o
      //  1         i
      //  2     o i C i o
      //  3         i
      //  4         o
      //
      //    [p] 0 1 2 3 4

      //                  o                       i                                mTmpCluster[t][p]
      if(chargeForCluster(mData[t-2]->at(p  ).charge, mData[t-1]->at(p  ).charge)) mTmpCluster[0][2] = mData[t-2]->at(p  );   // t-X -> older
      if(chargeForCluster(mData[t+2]->at(p  ).charge, mData[t+1]->at(p  ).charge)) mTmpCluster[4][2] = mData[t+2]->at(p  );   // t+X -> newer
      if(chargeForCluster(mData[t  ]->at(p-2).charge, mData[t  ]->at(p-1).charge)) mTmpCluster[2][0] = mData[t  ]->at(p-2);
      if(chargeForCluster(mData[t  ]->at(p+2).charge, mData[t  ]->at(p+1).charge)) mTmpCluster[2][4] = mData[t  ]->at(p+2);


      // The cells of the corners have 3 neighbours.
      //    o o   o o
      //    o i   i o
      //        C
      //    o i   i o
      //    o o   o o

      // bottom left
      if(chargeForCluster(mData[t+1]->at(p-2).charge,mData[t+1]->at(p-1).charge)) mTmpCluster[3][0] = mData[t+1]->at(p-2);
      if(chargeForCluster(mData[t+2]->at(p-2).charge,mData[t+1]->at(p-1).charge)) mTmpCluster[4][0] = mData[t+2]->at(p-2);
      if(chargeForCluster(mData[t+2]->at(p-1).charge,mData[t+1]->at(p-1).charge)) mTmpCluster[4][1] = mData[t+2]->at(p-1);
      // bottom right
      if(chargeForCluster(mData[t+2]->at(p+1).charge,mData[t+1]->at(p+1).charge)) mTmpCluster[4][3] = mData[t+2]->at(p+1);
      if(chargeForCluster(mData[t+2]->at(p+2).charge,mData[t+1]->at(p+1).charge)) mTmpCluster[4][4] = mData[t+2]->at(p+2);
      if(chargeForCluster(mData[t+1]->at(p+2).charge,mData[t+1]->at(p+1).charge)) mTmpCluster[3][4] = mData[t+1]->at(p+2);
      // top right
      if(chargeForCluster(mData[t-1]->at(p+2).charge,mData[t-1]->at(p+1).charge)) mTmpCluster[1][4] = mData[t-1]->at(p+2);
      if(chargeForCluster(mData[t-2]->at(p+2).charge,mData[t-1]->at(p+1).charge)) mTmpCluster[0][4] = mData[t-2]->at(p+2);
      if(chargeForCluster(mData[t-2]->at(p+1).charge,mData[t-1]->at(p+1).charge)) mTmpCluster[0][3] = mData[t-2]->at(p+1);
      // top left
      if(chargeForCluster(mData[t-2]->at(p-1).charge,mData[t-1]->at(p-1).charge)) mTmpCluster[0][1] = mData[t-2]->at(p-1);
      if(chargeForCluster(mData[t-2]->at(p-2).charge,mData[t-1]->at(p-1).charge)) mTmpCluster[0][0] = mData[t-2]->at(p-2);
      if(chargeForCluster(mData[t-1]->at(p-2).charge,mData[t-1]->at(p-1).charge)) mTmpCluster[1][0] = mData[t-1]->at(p-2);

      //
      // calculate cluster Properties
      //

      qMax = mTmpCluster[2][2].charge;
      qTot = 0;
      meanP = 0;
      meanT = 0;
      sigmaP = 0;
      sigmaT = 0;
      minP = mClusterSizePads;
      minT = mClusterSizeTime;
      maxP = 0;
      maxT = 0;
      mClusterDigitIndices.emplace_back();
      for (tt = 0; tt < mClusterSizeTime; ++tt) {
        deltaT = tt - mClusterSizeTime/2;
        for (pp = 0; pp < mClusterSizePads; ++pp) {
          deltaP = pp - mClusterSizePads/2;

          charge = mTmpCluster[tt][pp].charge;
          if (charge > 0 || mTmpCluster[tt][pp].event >= 0)
            mClusterDigitIndices.back().emplace_back(std::make_pair(mTmpCluster[tt][pp].index, mTmpCluster[tt][pp].event));

          qTot += charge;

          meanP += charge * static_cast<float>(deltaP);
          meanT += charge * static_cast<float>(deltaT);

          sigmaP += charge * static_cast<float>(deltaP)*static_cast<float>(deltaP);
          sigmaT += charge * static_cast<float>(deltaT)*static_cast<float>(deltaT);

          if (charge > 0) {
            minP = std::min(minP,pp); maxP = std::max(maxP,pp);
            minT = std::min(minT,tt); maxT = std::max(maxT,tt);
          }
        }
      }

      if (qTot > 0) {
        meanP  /= qTot;
        meanT  /= qTot;
        sigmaP /= qTot;
        sigmaT /= qTot;

        sigmaP = std::sqrt(sigmaP - (meanP*meanP));
        sigmaT = std::sqrt(sigmaT - (meanT*meanT));

        meanP += p+mPadOffset;
        meanT += mGlobalTimeOfLast-(mTimebins-1)+t;
      }

      ++foundNclusters;
      mClusterContainer.emplace_back(mCRU, mRow, qTot, qMax, meanP, sigmaP, meanT, sigmaT);

      if (mAssignChargeUnique) {
        if (p < (pMin+4)) {
          // If the cluster peak is in one of the 6 leftmost pads, the Cluster Finder
          // on the left has to know about it to ignore the already used pads.
          if (auto next = mNextCF.lock()) {
            next->clusterAlreadyUsed(t,p+mPadOffset);//,mTmpCluster);
          }
        }


        //
        // subtract found cluster from storage
        //
        // TODO: really nexessary?? or just set to 0
        for (tt=0; tt<5; ++tt) {
          for (pp=0; pp<5; ++pp) {
            //mData[t+(tt-2)][p+(pp-2)].charge -= mTmpCluster[tt][pp];
            mData[t+(tt-2)]->at(p+(pp-2)).clear();
          }
        }
      }
    }
  }

  if (foundNclusters > 0) return true;
  return false;
}

//________________________________________________________________________
bool HwClusterFinder::chargeForCluster(float outerCharge, float innerCharge)
{
  //printf("%.2f - %.2f = %.f compared to %.2f)?\n",toCompare,*charge,toCompare-*charge,-mDiffThreshold);
  if ((mRequirePositiveCharge && (outerCharge > 0)) &&
      (innerCharge > mChargeThreshold)) {
    return true;
  }
  return false;
}

//________________________________________________________________________
// TODO: really nexessary?? or just set to 0
void HwClusterFinder::clusterAlreadyUsed(short time, short pad)
{
  short localPad = pad - mPadOffset;

  short t,p;
  for (t=time-2; t<=time+2; ++t){
    if (t < 0 || t >= mTimebins) continue;
    for (p=localPad-2; p<=localPad+2; ++p){
      if (p < 0 || p >= mPads) continue;

      //mData[t][p].charge -= cluster[t-time+2][p-localPad+2].charge;
      mData[t]->at(p).clear();
    }
  }
}

//________________________________________________________________________
void HwClusterFinder::reset(unsigned globalTimeAfterReset)
{
  for(auto &tb : mData)
    for (auto &digi : *tb) digi.clear();

  mGlobalTimeOfLast = globalTimeAfterReset;
}

//________________________________________________________________________
void HwClusterFinder::printCluster(short time, short pad)
{
  short t,p;
  for (t = time-2; t <= time+2; ++t) {
    LOG(DEBUG) << "t " << t << ":\t";
    for (p = pad-2; p <= pad+2; ++p) {
      LOG(DEBUG) << mData[t]->at(p).charge << "\t";
    }
    LOG(DEBUG) << FairLogger::endl;
  }
  LOG(DEBUG) << FairLogger::endl;
}
