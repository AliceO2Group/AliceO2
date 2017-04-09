/// \file AliTPCUpgradeHwClusterFinder.cxx
/// \brief HwClusterFinder for the TPC


//#include "TPCSimulation/HwClusterer.h"
#include "TPCSimulation/HwClusterFinder.h"
#include "TPCSimulation/ClusterContainer.h"
#include "TPCSimulation/HwCluster.h"

#include "FairLogger.h"
#include "TClonesArray.h"


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
  , mRequireNeighbouringTimebin(false)//true)
  , mAutoProcessing(false)
  , mAssignChargeUnique(true)
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
  mData = new float*[mTimebins];
  for (t = 0; t < mTimebins; t++){
    mData[t] = new float[mPads]; 
    for (p = 0; p < mPads; p++){
      mData[t][p] = 0;
    }
  }

  tmpCluster = new float*[mClusterSizeTime];
  for (t=0; t<mClusterSizeTime; t++) {
    tmpCluster[t] = new float[mClusterSizePads];
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
  , mNextCF(other.mNextCF)
{
  short t,p;
  mData = new float*[mTimebins];
  for (t = 0; t < mTimebins; t++){
    mData[t] = new float[mPads]; 
    for (p = 0; p < mPads; p++){
      mData[t][p] = other.mData[t][p];
    }
  }

  for (t=0; t<mClusterSizeTime; t++) {
    tmpCluster[t] = new float[mClusterSizePads];
    for (p=0; p<mClusterSizePads; p++){
      tmpCluster[t][p] = other.tmpCluster[t][p];
    }
  }
}

//________________________________________________________________________
HwClusterFinder::~HwClusterFinder()
{
  short t;
  for (t = 0; t < mTimebins; t++){
    delete [] mData[t];
  }
  delete [] mData;
  delete [] mZeroTimebin;

  for (t=0; t<mClusterSizeTime; t++) {
    delete [] tmpCluster[t];
  }
  delete [] tmpCluster;
}

//________________________________________________________________________
bool HwClusterFinder::AddTimebin(float* timebin, unsigned globalTime, int length)
{
//  printf("adding the following timebin(%u)\t",globalTime);
//  for (int i=0; i<length; i++){
//    printf("%.2f\t",timebin[i]);
//  }
//  printf("\n");

  mGlobalTimeOfLast = globalTime;
  mTimebinsAfterLastProcessing++;

  //
  // reordering of the local arrays
  //
  float* data0 = mData[0];
  short t,p;
  for (t = 0; t<mTimebins-1; t++) {
    mData[t] = mData[t+1];
  }
  mData[mTimebins-1] = data0;

    //printf("%d\n",mGlobalTimeOfLast);
  if (length != mPads) {
    if (length < mPads) {
//      LOG(INFO) << "Number of pads in timebin (" << length << ") doesn't correspond to setting (" << mPads << "), "
//                << "filling remaining with 0." << FairLogger::endl;
      for (p = 0; p < length; p++){
        mData[mTimebins-1][p] = timebin[p];
      }
      for (p = length; p < mPads; p++){
        mData[mTimebins-1][p] = 0;
      }
    } else {
//      LOG(INFO) << "Number of pads in timebin (" << length << ") doesn't correspond to setting (" << mPads << "), "
//                << "ignoring last ones." << FairLogger::endl;
      for (p = 0; p < mPads; p++){
        mData[mTimebins-1][p] = timebin[p];
      }
    }
  } else {
    for (p = 0; p < mPads; p++){
      mData[mTimebins-1][p] = timebin[p];
    }
  }

  if (mAutoProcessing & (mTimebinsAfterLastProcessing >= (mTimebins -2 -2))) findCluster();
  return true;
}

//________________________________________________________________________
bool HwClusterFinder::AddTimebins(int nBins, float** timebins, unsigned globalTimeOfLast, int length)
{
  bool ret = false;
  for(short n=0; n<nBins; n++){
    ret = ret | !(AddTimebin(timebins[n],globalTimeOfLast,length));
  }
  return !ret;
}

//________________________________________________________________________
void HwClusterFinder::AddZeroTimebin(unsigned globalTime, int length)
{
  if (mZeroTimebin == nullptr) {
    mZeroTimebin = new float[length];
    for (short i = 0; i < length; i++) mZeroTimebin[i] = 0;
  }
  bool ret = AddTimebin(mZeroTimebin,globalTime,length);
}

//________________________________________________________________________
void HwClusterFinder::PrintLocalStorage()
{
  short t,p;
  for (t = 0; t < mTimebins; t++){
  printf("t %d:\t",t);
    for (p = 0; p < mPads; p++){
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

  //
  // peak finding
  //
  short t,p,tt,pp;
  for (t=tMin; t<=tMax; t++) {
    for (p=pMin; p<=pMax; p++) {
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
      if (mData[t  ][p  ] < mChargeThreshold) continue;

      // Require at least one neighboring time bin with signal
      if (mRequireNeighbouringTimebin   && (mData[t-1][p  ] + mData[t+1][p  ] <= 0)) continue;
      // Require at least one neighboring pad with signal
      if (mRequireNeighbouringPad       && (mData[t  ][p-1] + mData[t  ][p+1] <= 0)) continue;

      // check for local maximum
      if (mData[t-1][p  ] >=  mData[t][p]) continue;
      if (mData[t+1][p  ] >   mData[t][p]) continue;
      if (mData[t  ][p-1] >=  mData[t][p]) continue;
      if (mData[t  ][p+1] >   mData[t][p]) continue;
      if (mData[t-1][p-1] >=  mData[t][p]) continue;
      if (mData[t+1][p+1] >   mData[t][p]) continue;
      if (mData[t+1][p-1] >   mData[t][p]) continue;
      if (mData[t-1][p+1] >=  mData[t][p]) continue;
//      printf("##\n");
//      printf("## cluster found at t=%d, p=%d (in CF %d in row %d of CRU %d)\n",t,p,mId,mRow,mCRU);
//      printf("##\n");
//      printCluster(t,p);
        
      //
      // cluster was found!!
      //
      
      // prepare temp storage
      for (tt=0; tt<mClusterSizeTime; tt++) {
        for (pp=0; pp<mClusterSizePads; pp++){
          tmpCluster[tt][pp] = 0;
        }
      }

      //
      // Cluster peak (C) and surrounding inner 3x3 matrix (i) is always
      // used taken for the found cluster
      //
      float charge;
      for (tt=1; tt<4; tt++) {
        for (pp=1; pp<4; pp++) {
          charge = mData[t+(tt-2)][p+(pp-2)];
          if ( mRequirePositiveCharge && charge < 0) continue;
          tmpCluster[tt][pp] = charge;
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

//      if ((mCRU == 179 && mRow == 1 && p+mPadOffset == 103 && mGlobalTimeOfLast-(mTimebins-1)+t == 170)/* ||
//          (mCRU == 256 && mRow == 10 &&  p+mPadOffset == 27 && mGlobalTimeOfLast-(mTimebins-1)+t == 181)*/ ) {
//        PrintLocalStorage();
//      }

      HwCluster cl(mCRU, mRow, mClusterSizePads, mClusterSizeTime, tmpCluster,p+mPadOffset,mGlobalTimeOfLast-(mTimebins-1)+t);
      foundNclusters++;
      clusterContainer.push_back(cl);

      if (mAssignChargeUnique) {
        if (p < (pMin+4)) { 
          // If the cluster peak is in one of the 6 leftmost pads, the Cluster Finder
          // on the left has to know about it to ignore the already used pads.
          if (mNextCF != nullptr) mNextCF->clusterAlreadyUsed(t,p+mPadOffset,tmpCluster);
        }
        

        //
        // subtract found cluster from storage
        //
        for (tt=0; tt<5; tt++) {
          for (pp=0; pp<5; pp++) {
            mData[t+(tt-2)][p+(pp-2)] -= tmpCluster[tt][pp];
          }
        }
      }
    }
  }

  if (foundNclusters > 0) return true;
  return false;
}

//________________________________________________________________________
float HwClusterFinder::chargeForCluster(float* charge, float* toCompare)
{
  //printf("%.2f - %.2f = %.f compared to %.2f)?\n",toCompare,*charge,toCompare-*charge,-mDiffThreshold);
  if ((mRequirePositiveCharge && (*charge > 0)) &
      (*toCompare > mDiffThreshold)) {//mChargeThreshold)) {
    //printf("\tyes\n");
    return *charge;
  } else {
    //printf("\tno\n");
    return 0;
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
void HwClusterFinder::clusterAlreadyUsed(short time, short pad, float** cluster)
{
  short localPad = pad - mPadOffset;

  short t,p;
  for (t=time-2; t<=time+2; t++){
    if (t < 0 || t >= mTimebins) continue;
    for (p=localPad-2; p<=localPad+2; p++){
      if (p < 0 || p >= mPads) continue;
        
      mData[t][p] -= cluster[t-time+2][p-localPad+2];
    }
  }
}

//________________________________________________________________________
void HwClusterFinder::reset(unsigned globalTimeAfterReset)
{
  short t,p;
  for (t = 0; t < mTimebins; t++){
    for (p = 0; p < mPads; p++){
      mData[t][p] = 0;
    }
  }

  mGlobalTimeOfLast = globalTimeAfterReset;
}

//________________________________________________________________________
void HwClusterFinder::printCluster(short time, short pad)
{
  short t,p;
  for (t = time-2; t <= time+2; t++) {
    printf("%d\t\t",t);
    for (p = pad-2; p <= pad+2; p++) {
      printf("%.2f\t", mData[t][p]);
    }
    printf("\n");
  }
}
