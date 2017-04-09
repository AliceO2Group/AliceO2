/// \file Cluster.cxx
/// \brief Implementation of the ITSMFT cluster

#include "ITSMFTReconstruction/Cluster.h"
#include "FairLogger.h"

#include <TMath.h>
#include <TString.h>

#include <cstdlib>

using namespace o2::ITSMFT;

ClassImp(o2::ITSMFT::Cluster)

//_____________________________________________________
Cluster::Cluster()
  : mTracks{ -1, -1, -1 },
    mX(0),
    mY(0),
    mZ(0),
    mSigmaY2(0),
    mSigmaZ2(0),
    mSigmaYZ(0),
    mVolumeId(0),
    mCharge(0),
    mRecoInfo(0),
    mNxNzN(0)
#ifdef _ClusterTopology_
    ,
    mPatternNRows(0),
    mPatternNCols(0),
    mPatternMinRow(0),
    mPatternMinCol(0)
#endif
{
// default constructor
#ifdef _ClusterTopology_
  memset(mPattern, 0, kMaxPatternBytes * sizeof(UChar_t));
#endif
}

//_____________________________________________________
Cluster::~Cluster()
{
  // default destructor
}

//_____________________________________________________
Cluster::Cluster(const Cluster& cluster)
  : FairTimeStamp(cluster),
    mTracks{ cluster.mTracks[0], cluster.mTracks[1], cluster.mTracks[2] },
    mX(cluster.mX),
    mY(cluster.mY),
    mZ(cluster.mZ),
    mSigmaY2(cluster.mSigmaY2),
    mSigmaZ2(cluster.mSigmaZ2),
    mSigmaYZ(cluster.mSigmaYZ),
    mVolumeId(cluster.mVolumeId),
    mCharge(cluster.mCharge),
    mRecoInfo(cluster.mRecoInfo),
    mNxNzN(cluster.mNxNzN)
#ifdef _ClusterTopology_
    ,
    mPatternNRows(cluster.mPatternNRows),
    mPatternNCols(cluster.mPatternNCols),
    mPatternMinRow(cluster.mPatternMinRow),
    mPatternMinCol(cluster.mPatternMinCol)
#endif
{
// copy constructor
#ifdef _ClusterTopology_
  memcpy(mPattern, cluster.mPattern, kMaxPatternBytes * sizeof(UChar_t));
#endif
}

#ifdef _ClusterTopology_
//______________________________________________________________________________
void Cluster::resetPattern()
{
  // reset pixels pattern
  memset(mPattern, 0, kMaxPatternBytes * sizeof(UChar_t));
}

//______________________________________________________________________________
Bool_t Cluster::testPixel(UShort_t row, UShort_t col) const
{
  // test if pixel at relative row,col is fired
  int nbits = row * getPatternColSpan() + col;
  if (nbits >= kMaxPatternBits)
    return kFALSE;
  int bytn = nbits >> 3; // 1/8
  int bitn = nbits % 8;
  return (mPattern[bytn] & (0x1 << bitn)) != 0;
  //
}

//______________________________________________________________________________
void Cluster::setPixel(UShort_t row, UShort_t col, Bool_t fired)
{
  // test if pixel at relative row,col is fired
  int nbits = row * getPatternColSpan() + col;
  if (nbits >= kMaxPatternBits)
    return;
  int bytn = nbits >> 3; // 1/8
  int bitn = nbits % 8;
  if (nbits >= kMaxPatternBits)
    exit(1);
  if (fired)
    mPattern[bytn] |= (0x1 << bitn);
  else
    mPattern[bytn] &= (0xff ^ (0x1 << bitn));
  //
}

//______________________________________________________________________________
void Cluster::setPatternRowSpan(UShort_t nr, Bool_t truncated)
{
  // set pattern span in rows, flag if truncated
  mPatternNRows = kSpanMask & nr;
  if (truncated)
    mPatternNRows |= kTruncateMask;
}

//______________________________________________________________________________
void Cluster::setPatternColSpan(UShort_t nc, Bool_t truncated)
{
  // set pattern span in columns, flag if truncated
  mPatternNCols = kSpanMask & nc;
  if (truncated)
    mPatternNCols |= kTruncateMask;
}

#endif

//______________________________________________________________________________
Bool_t Cluster::hasCommonTrack(const Cluster* cl) const
{
  // check if clusters have common tracks
  int lbi, lbj;
  for (int i = 0; i < 3; i++) {
    if ((lbi = getLabel(i)) < 0)
      break;
    for (int j = 0; j < 3; j++) {
      if ((lbj = cl->getLabel(j)) < 0)
        break;
      if (lbi == lbj)
        return kTRUE;
    }
  }
  return kFALSE;
}
