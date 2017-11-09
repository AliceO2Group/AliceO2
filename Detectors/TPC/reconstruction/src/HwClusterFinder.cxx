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
    short cru, short row, short id,
    short padOffset, short pads, short timebins,
    float diffThreshold, float chargeThreshold,
    bool requirePositiveCharge)
  : mGlobalTimeOfLast(0)
  , mTimebinsAfterLastProcessing(0)
  , mCRU(cru)
  , mRow(row)
  , mId(id)
  , mPadOffset(padOffset)
  , mPads(pads)
  , mTimebins(timebins)
  , mClusterSizePads(5)
  , mClusterSizeTime(5)
  , mDiffThreshold(diffThreshold)
  , mChargeThreshold(chargeThreshold)
  , mRequirePositiveCharge(requirePositiveCharge)
  , mRequireNeighbouringPad(false)//true)
  , mRequireNeighbouringTimebin(true)
  , mAutoProcessing(false)
  , mAssignChargeUnique(false)//true)
  , mData(nullptr)
  , tmpCluster(nullptr)
  , mZeroTimebin(nullptr)
  , mNextCF(nullptr)
{
  if (mPads < mClusterSizePads) {
    LOG(ERROR) << "Given width in pad direction is smaller than cluster size in pad direction."
      << " width in pad direction was increased to cluster size." << FairLogger::endl;
    mPads = mClusterSizePads;
  }
  if (mTimebins < mClusterSizeTime) {
    LOG(ERROR) << "Given width in time direction is smaller than cluster size in time direction."
      << " width in time direction was increased to cluster size." << FairLogger::endl;
    mTimebins = mClusterSizeTime;
  }

  if (mTimebins*mPads > 64) {
    LOG(WARNING) << "Bins in pad direction X bins in time direction is larger than 64." << FairLogger::endl;
  }

  short t,p;
  mData = new MiniDigit*[mTimebins];
  for (t = 0; t < mTimebins; ++t){
    mData[t] = new MiniDigit[mPads];
//    for (p = 0; p < mPads; ++p){
//      mData[t][p] = MiniDigit();
//    }
  }

  tmpCluster = new MiniDigit*[mClusterSizeTime];
  for (t=0; t<mClusterSizeTime; ++t) {
    tmpCluster[t] = new MiniDigit[mClusterSizePads];
  }
}

//________________________________________________________________________
HwClusterFinder::HwClusterFinder(const HwClusterFinder& other)
  : mGlobalTimeOfLast(other.mGlobalTimeOfLast)
  , mTimebinsAfterLastProcessing(other.mTimebinsAfterLastProcessing)
  , mCRU(other.mCRU)
  , mRow(other.mRow)
  , mId(other.mId)
  , mPadOffset(other.mPadOffset)
  , mPads(other.mPads)
  , mTimebins(other.mTimebins)
  , mClusterSizePads(other.mClusterSizePads)
  , mClusterSizeTime(other.mClusterSizeTime)
  , mDiffThreshold(other.mDiffThreshold)
  , mChargeThreshold(other.mChargeThreshold)
  , mRequirePositiveCharge(other.mRequirePositiveCharge)
  , mRequireNeighbouringPad(other.mRequireNeighbouringPad)
  , mRequireNeighbouringTimebin(other.mRequireNeighbouringTimebin)
  , mAutoProcessing(other.mAutoProcessing)
  , mAssignChargeUnique(other.mAssignChargeUnique)
  , clusterContainer(other.clusterContainer)
  , clusterDigitIndices(other.clusterDigitIndices)
  , mNextCF(other.mNextCF)
{
  short t,p;
  mData = new MiniDigit*[mTimebins];
  for (t = 0; t < mTimebins; ++t){
    mData[t] = new MiniDigit[mPads];
    for (p = 0; p < mPads; ++p){
      mData[t][p] = other.mData[t][p];
    }
  }

  for (t=0; t<mClusterSizeTime; ++t) {
    tmpCluster[t] = new MiniDigit[mClusterSizePads];
    for (p=0; p<mClusterSizePads; ++p){
      tmpCluster[t][p] = other.tmpCluster[t][p];
    }
  }
}

//________________________________________________________________________
HwClusterFinder::~HwClusterFinder()
{
  short t;
  for (t = 0; t < mTimebins; ++t){
    delete [] mData[t];
  }
  delete [] mData;
  delete [] mZeroTimebin;

  for (t=0; t<mClusterSizeTime; ++t) {
    delete [] tmpCluster[t];
  }
  delete [] tmpCluster;
}

//________________________________________________________________________
//bool HwClusterFinder::AddTimebins(int nBins, float** timebins, unsigned globalTimeOfLast, int length)
//{
//  bool ret = false;
//  for(short n=0; n<nBins; ++n){
//    ret = ret | !(AddTimebin(timebins[n],globalTimeOfLast,length));
//  }
//  return !ret;
//}

//________________________________________________________________________
void HwClusterFinder::AddZeroTimebin(unsigned globalTime, int length)
{
  if (mZeroTimebin == nullptr) {
    mZeroTimebin = new MiniDigit[length];
    for (short i = 0; i < length; ++i) mZeroTimebin[i] = MiniDigit();
  }
  bool ret = AddTimebin(mZeroTimebin,globalTime,length);
}

//________________________________________________________________________
void HwClusterFinder::PrintLocalStorage()
{
  short t,p;
  for (t = 0; t < mTimebins; ++t){
  printf("t %d:\t",t);
    for (p = 0; p < mPads; ++p){
      printf("%.2f\t", mData[t][p]);
    }
    printf("\n");
  }
  printf("\n");
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
  short delTm = -2;
  short delTp = 2;

  //
  // In pad direction
  //
  int pMin = 2;
  int pMax = (mPads-1)-2;
  short delPm = -2;
  short delPp = 2;

  double qMax;
  double qTot;
  double charge;
  double meanP, meanT;
  double sigmaP, sigmaT;
  short minP, minT;
  short maxP, maxT;
  short deltaP, deltaT;
  short clusterSize;
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
      if (mData[t  ][p  ].charge < mChargeThreshold) continue;

      // Require at least one neighboring time bin with signal
      if (mRequireNeighbouringTimebin   && (mData[t-1][p  ].charge + mData[t+1][p  ].charge <= 0)) continue;
      // Require at least one neighboring pad with signal
      if (mRequireNeighbouringPad       && (mData[t  ][p-1].charge + mData[t  ][p+1].charge <= 0)) continue;

      // check for local maximum
      if (mData[t-1][p  ].charge >=  mData[t][p].charge) continue;
      if (mData[t+1][p  ].charge >   mData[t][p].charge) continue;
      if (mData[t  ][p-1].charge >=  mData[t][p].charge) continue;
      if (mData[t  ][p+1].charge >   mData[t][p].charge) continue;
      if (mData[t-1][p-1].charge >=  mData[t][p].charge) continue;
      if (mData[t+1][p+1].charge >   mData[t][p].charge) continue;
      if (mData[t+1][p-1].charge >   mData[t][p].charge) continue;
      if (mData[t-1][p+1].charge >=  mData[t][p].charge) continue;
//      printf("##\n");
//      printf("## cluster found at t=%d, p=%d (in CF %d in row %d of CRU %d)\n",t,p,mId,mRow,mCRU);
//      printf("##\n");
//      printCluster(t,p);

      //
      // cluster was found!!
      //

      // prepare temp storage
      for (tt=0; tt<mClusterSizeTime; ++tt) {
        for (pp=0; pp<mClusterSizePads; ++pp){
          tmpCluster[tt][pp] = MiniDigit();
        }
      }

      //
      // Cluster peak (C) and surrounding inner 3x3 matrix (i) is always
      // used taken for the found cluster
      //
      for (tt=1; tt<4; ++tt) {
        for (pp=1; pp<4; ++pp) {
          if ( mRequirePositiveCharge && mData[t+(tt-2)][p+(pp-2)].charge < 0) continue;
          tmpCluster[tt][pp] = mData[t+(tt-2)][p+(pp-2)];
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

    //tmpCluster[t][p]
      tmpCluster[0][2] = chargeForCluster(&mData[t-2][p  ],&mData[t-1][p  ]);   // t-X -> older
      tmpCluster[4][2] = chargeForCluster(&mData[t+2][p  ],&mData[t+1][p  ]);   // t+X -> newer
      tmpCluster[2][0] = chargeForCluster(&mData[t  ][p-2],&mData[t  ][p-1]);
      tmpCluster[2][4] = chargeForCluster(&mData[t  ][p+2],&mData[t  ][p+1]);


      // The cells of the corners have 3 neighbours.
      //    o o   o o
      //    o i   i o
      //        C
      //    o i   i o
      //    o o   o o

      // bottom left
      tmpCluster[3][0] = chargeForCluster(&mData[t+1][p-2],&mData[t+1][p-1]);
      tmpCluster[4][0] = chargeForCluster(&mData[t+2][p-2],&mData[t+1][p-1]);
      tmpCluster[4][1] = chargeForCluster(&mData[t+2][p-1],&mData[t+1][p-1]);
      // bottom right
      tmpCluster[4][3] = chargeForCluster(&mData[t+2][p+1],&mData[t+1][p+1]);
      tmpCluster[4][4] = chargeForCluster(&mData[t+2][p+2],&mData[t+1][p+1]);
      tmpCluster[3][4] = chargeForCluster(&mData[t+1][p+2],&mData[t+1][p+1]);
      // top right
      tmpCluster[1][4] = chargeForCluster(&mData[t-1][p+2],&mData[t-1][p+1]);
      tmpCluster[0][4] = chargeForCluster(&mData[t-2][p+2],&mData[t-1][p+1]);
      tmpCluster[0][3] = chargeForCluster(&mData[t-2][p+1],&mData[t-1][p+1]);
      // top left
      tmpCluster[0][1] = chargeForCluster(&mData[t-2][p-1],&mData[t-1][p-1]);
      tmpCluster[0][0] = chargeForCluster(&mData[t-2][p-2],&mData[t-1][p-1]);
      tmpCluster[1][0] = chargeForCluster(&mData[t-1][p-2],&mData[t-1][p-1]);

      //
      // calculate cluster Properties
      //

      qMax = tmpCluster[2][2].charge;
      qTot = 0;
      meanP = 0;
      meanT = 0;
      sigmaP = 0;
      sigmaT = 0;
      minP = mClusterSizePads;
      minT = mClusterSizeTime;
      maxP = 0;
      maxT = 0;
      clusterSize = 0;
      clusterDigitIndices.emplace_back();
      for (tt = 0; tt < mClusterSizeTime; ++tt) {
        deltaT = tt - mClusterSizeTime/2;
        for (pp = 0; pp < mClusterSizePads; ++pp) {
          deltaP = pp - mClusterSizePads/2;

          charge = tmpCluster[tt][pp].charge;
          if (charge > 0 || tmpCluster[tt][pp].event >= 0)
            clusterDigitIndices.back().emplace_back(std::make_pair(tmpCluster[tt][pp].index, tmpCluster[tt][pp].event));

          qTot += charge;

          meanP += charge * deltaP;
          meanT += charge * deltaT;

          sigmaP += charge * deltaP*deltaP;
          sigmaT += charge * deltaT*deltaT;

          if (charge > 0) {
            minP = std::min(minP,pp); maxP = std::max(maxP,pp);
            minT = std::min(minT,tt); maxT = std::max(maxT,tt);
          }
        }
      }

      clusterSize = (maxP-minP+1)*10 + (maxT-minT+1);

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
//      clusterContainer.emplace_back(mCRU, mRow, mClusterSizePads, mClusterSizeTime, tmpCluster,p+mPadOffset,mGlobalTimeOfLast-(mTimebins-1)+t);
      clusterContainer.emplace_back(mCRU, mRow, qTot, qMax, meanP, sigmaP, meanT, sigmaT);

      if (mAssignChargeUnique) {
        if (p < (pMin+4)) {
          // If the cluster peak is in one of the 6 leftmost pads, the Cluster Finder
          // on the left has to know about it to ignore the already used pads.
          if (mNextCF != nullptr) mNextCF->clusterAlreadyUsed(t,p+mPadOffset,tmpCluster);
        }


        //
        // subtract found cluster from storage
        //
        // TODO: really nexessary?? or just set to 0
        for (tt=0; tt<5; ++tt) {
          for (pp=0; pp<5; ++pp) {
            //mData[t+(tt-2)][p+(pp-2)].charge -= tmpCluster[tt][pp];
            mData[t+(tt-2)][p+(pp-2)].clear();// = MiniDigit();
          }
        }
      }
    }
  }

  if (foundNclusters > 0) return true;
  return false;
}

//________________________________________________________________________
HwClusterFinder::MiniDigit HwClusterFinder::chargeForCluster(MiniDigit* charge, MiniDigit* toCompare)
{
  //printf("%.2f - %.2f = %.f compared to %.2f)?\n",toCompare,*charge,toCompare-*charge,-mDiffThreshold);
  if ((mRequirePositiveCharge && (charge->charge > 0)) &
      (toCompare->charge > mDiffThreshold)) {//mChargeThreshold)) {
    //printf("\tyes\n");
    return *charge;
  } else {
    //printf("\tno\n");
    return MiniDigit();
  }
}

//________________________________________________________________________
void HwClusterFinder::setNextCF(HwClusterFinder* nextCF)
{
  if (nextCF == nullptr) {
    LOG(WARNING) << R"(Got "nullptrd" as neighboring Cluster Finder.)" << FairLogger::endl;
    return;
  }
  if (mNextCF != nullptr) {
      LOG(WARNING) << "This Cluster Finder ("
        << "CRU " << mCRU << ", row " << mRow << ",pad offset " << mPadOffset
        << ") had already a neighboring CF set ("
        << "CRU " << mNextCF->getCRU() << ", row " << mNextCF->getRow() << ",pad offset " << mNextCF->getPadOffset()
        << "). It will be replaced with a new one ("
        << "CRU " << nextCF->getCRU() << ", row " << nextCF->getRow() << ",pad offset " << nextCF->getPadOffset()
        << ")." << FairLogger::endl;
  }
  mNextCF = nextCF;

  return;
}

//________________________________________________________________________
// TODO: really nexessary?? or just set to 0
void HwClusterFinder::clusterAlreadyUsed(short time, short pad, MiniDigit** cluster)
{
  short localPad = pad - mPadOffset;

  short t,p;
  for (t=time-2; t<=time+2; ++t){
    if (t < 0 || t >= mTimebins) continue;
    for (p=localPad-2; p<=localPad+2; ++p){
      if (p < 0 || p >= mPads) continue;

      //mData[t][p].charge -= cluster[t-time+2][p-localPad+2].charge;
      mData[t][p].clear();// = MiniDigit();
    }
  }
}

//________________________________________________________________________
void HwClusterFinder::reset(unsigned globalTimeAfterReset)
{
  short t,p;
  for (t = 0; t < mTimebins; ++t){
    for (p = 0; p < mPads; ++p){
      mData[t][p].clear();// = MiniDigit();
    }
  }

  mGlobalTimeOfLast = globalTimeAfterReset;
}

//________________________________________________________________________
void HwClusterFinder::printCluster(short time, short pad)
{
  short t,p;
  for (t = time-2; t <= time+2; ++t) {
    printf("%d\t\t",t);
    for (p = pad-2; p <= pad+2; ++p) {
      printf("%.2f\t", mData[t][p].charge);
    }
    printf("\n");
  }
}
