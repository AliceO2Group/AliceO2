/// \file Cluster.h
/// \brief Definition of the ITS cluster
#ifndef ALICEO2_ITS_CLUSTER_H
#define ALICEO2_ITS_CLUSTER_H

#include "Rtypes.h" // for Double_t, ULong_t, etc

#include "FairTimeStamp.h" // for FairTimeStamp
#include "ITSBase/GeometryTGeo.h"

class TGeoHMatrix;

// uncomment this to have cluster topology in stored
//#define _ClusterTopology_

#define CLUSTER_VERSION 2

namespace AliceO2
{
namespace ITS
{
/// \class Cluster
/// \brief Cluster class for the ITS
///

class Cluster : public FairTimeStamp
{
 public:
  enum { // frame in which the track is currently defined
    kUsed = BIT(14),
    kShared = BIT(15),
    kFrameLoc = BIT(16),
    kFrameTrk = BIT(17),
    kFrameGlo = BIT(18),
    kFrameBits = kFrameLoc | kFrameTrk | kFrameGlo,
    kSplit = BIT(19)
  };
  //
  enum SortMode_t {       // various modes
    kSortIdLocXZ = BIT(0) // sort according to ID, then X,Z of local frame
    ,
    kSortIdTrkYZ = BIT(1) // sort according to ID, then Y,Z of tracking frame
    ,
    kSortBits = kSortIdLocXZ | kSortIdTrkYZ
  };
  enum {
    kOffsNZ = 0,
    kMaskNZ = 0xff,
    kOffsNX = 8,
    kMaskNX = 0xff,
    kOffsNPix = 16,
    kMaskNPix = 0x1ff,
    kOffsClUse = 25,
    kMaskClUse = 0x7f
  };
//
#ifdef _ClusterTopology_
  enum { kMaxPatternBits = 32 * 16, kMaxPatternBytes = kMaxPatternBits / 8, kSpanMask = 0x7fff,
         kTruncateMask = 0x8000 };
#endif
 public:
  Cluster();
  Cluster(const Cluster& cluster);
  virtual ~Cluster();

  //****** Basic methods ******************
  void setLabel(Int_t lab, Int_t i)
  {
    if (i >= 0 && i < 3)
      mTracks[i] = lab;
  }
  void setX(Float_t x) { mX = x; }
  void setY(Float_t y) { mY = y; }
  void setZ(Float_t z) { mZ = z; }
  void setSigmaY2(Float_t sigy2) { mSigmaY2 = sigy2; }
  void setSigmaZ2(Float_t sigz2) { mSigmaZ2 = sigz2; }
  void setSigmaYZ(Float_t sigyz) { mSigmaYZ = sigyz; };
  void setVolumeId(UShort_t id) { mVolumeId = id; }
  void increaseClusterUsage()
  {
    if (TestBit(kUsed))
      SetBit(kShared);
    else
      SetBit(kUsed);
  }

  Int_t getLabel(Int_t i) const { return mTracks[i]; }
  Float_t getX() const { return mX; }
  Float_t getY() const { return mY; }
  Float_t getZ() const { return mZ; }
  Float_t getSigmaY2() const { return mSigmaY2; }
  Float_t getSigmaZ2() const { return mSigmaZ2; }
  Float_t getSigmaYZ() const { return mSigmaYZ; }
  UShort_t getVolumeId() const { return mVolumeId; }
  //**************************************

  Int_t getLayer() const { return sGeom->getLayer(mVolumeId); }
  //
  Bool_t isFrameLoc() const { return TestBit(kFrameLoc); }
  Bool_t isFrameGlo() const { return TestBit(kFrameGlo); }
  Bool_t isFrameTrk() const { return TestBit(kFrameTrk); }
  //
  Bool_t isSplit() const { return TestBit(kSplit); }
  //
  void setFrameLoc()
  {
    ResetBit(kFrameBits);
    SetBit(kFrameLoc);
  }
  void setFrameGlo()
  {
    ResetBit(kFrameBits);
    SetBit(kFrameGlo);
  }
  void setFrameTrk()
  {
    ResetBit(kFrameTrk);
    SetBit(kFrameTrk);
  }
  //
  void setSplit(Bool_t v = kTRUE) { SetBit(kSplit, v); }
  //
  void goToFrameGlo();
  void goToFrameLoc();
  void goToFrameTrk();
  void getLocalXYZ(Float_t xyz[3]) const;
  void getTrackingXYZ(Float_t xyz[3]) const;
  //
  void setNxNzN(UChar_t nx, UChar_t nz, UShort_t n)
  {
    mNxNzN = ((n & kMaskNPix) << kOffsNPix) + ((nx & kMaskNX) << kOffsNX) + ((nz & kMaskNZ) << kOffsNZ);
  }
  void setClusterUsage(Int_t n);
  void modifyClusterUsage(Bool_t used = kTRUE) { used ? incClusterUsage() : decreaseClusterUsage(); }
  void incClusterUsage()
  {
    setClusterUsage(getClusterUsage() + 1);
    increaseClusterUsage();
  }
  void decreaseClusterUsage();
  Int_t getNx() const { return (mNxNzN >> kOffsNX) & kMaskNX; }
  Int_t getNz() const { return (mNxNzN >> kOffsNZ) & kMaskNZ; }
  Int_t getNPix() const { return (mNxNzN >> kOffsNPix) & kMaskNPix; }
  Int_t getClusterUsage() const { return (mNxNzN >> kOffsClUse) & kMaskClUse; }
  //
  void setQ(UShort_t q) { mCharge = q; }
  Int_t getQ() const { return mCharge; }
  //
  virtual void print(Option_t* option = "") const;
  virtual const TGeoHMatrix* getTracking2LocalMatrix() const;
  virtual TGeoHMatrix* getMatrix(Bool_t original = kFALSE) const;
  virtual Bool_t getGlobalXYZ(Float_t xyz[3]) const;
  virtual Bool_t getGlobalCov(Float_t cov[6]) const;
  virtual Bool_t getXRefPlane(Float_t& xref) const;
  virtual Bool_t getXAlphaRefPlane(Float_t& x, Float_t& alpha) const;
  //
  virtual Bool_t isSortable() const { return kTRUE; }
  virtual Bool_t isEqual(const TObject* obj) const;
  virtual Int_t Compare(const TObject* obj) const;
  //
  UShort_t getRecoInfo() const { return mRecoInfo; }
  void setRecoInfo(UShort_t v)
  {
    mRecoInfo = v;
    modifyClusterUsage(v > 0);
  }
  //
  Bool_t hasCommonTrack(const Cluster* cl) const;
  //
  static void setGeom(GeometryTGeo* gm) { sGeom = gm; }
  static void setSortMode(SortMode_t md)
  {
    sMode &= ~kSortBits;
    sMode |= md;
  }
  static UInt_t getSortMode() { return sMode & kSortBits; }
  static UInt_t getMode() { return sMode; }
  static SortMode_t sortModeIdTrkYZ() { return kSortIdTrkYZ; }
  static SortMode_t sortModeIdLocXZ() { return kSortIdLocXZ; }
//
#ifdef _ClusterTopology_
  Int_t getPatternRowSpan() const { return mPatternNRows & kSpanMask; }
  Int_t getPatternColSpan() const { return mPatternNCols & kSpanMask; }
  Bool_t isPatternRowsTruncated() const { return mPatternNRows & kTruncateMask; }
  Bool_t isPatternColsTruncated() const { return mPatternNRows & kTruncateMask; }
  Bool_t isPatternTruncated() const { return isPatternRowsTruncated() || isPatternColsTruncated(); }
  void setPatternRowSpan(UShort_t nr, Bool_t truncated);
  void setPatternColSpan(UShort_t nc, Bool_t truncated);
  void setPatternMinRow(UShort_t row) { mPatternMinRow = row; }
  void setPatternMinCol(UShort_t col) { mPatternMinCol = col; }
  void resetPattern();
  Bool_t testPixel(UShort_t row, UShort_t col) const;
  void setPixel(UShort_t row, UShort_t col, Bool_t fired = kTRUE);
  void getPattern(UChar_t patt[kMaxPatternBytes])
  {
    for (Int_t i = 0; i < kMaxPatternBytes; i++)
      patt[i] = mPattern[i];
  }
  Int_t getPatternMinRow() const { return mPatternMinRow; }
  Int_t getPatternMinCol() const { return mPatternMinCol; }
#endif
  //
 protected:
  //
  Cluster& operator=(const Cluster& cluster);

  Int_t mTracks[3];   ///< MC labels
  Float_t mX;         ///< X of the cluster in the tracking c.s.
  Float_t mY;         ///< Y of the cluster in the tracking c.s.
  Float_t mZ;         ///< Z of the cluster in the tracking c.s.
  Float_t mSigmaY2;   ///< Sigma Y square of cluster
  Float_t mSigmaZ2;   ///< Sigma Z square of cluster
  Float_t mSigmaYZ;   ///< Non-diagonal element of cov.matrix
  UShort_t mVolumeId; ///< Volume ID of the detector element

  UShort_t mCharge;           ///<  charge (for MC studies only)
  UShort_t mRecoInfo;         //!< space reserved for reco time manipulations
  Int_t mNxNzN;               ///< effective cluster size in X (1st byte) and Z (2nd byte) directions
                              ///< and total Npix(next 9 bits). last 7 bits are used for clusters usage counter
  static UInt_t sMode;        //!< general mode (sorting mode etc)
  static GeometryTGeo* sGeom; //!< pointer on the geometry data

#ifdef _ClusterTopology_
  UShort_t mPatternNRows;             ///< pattern span in rows
  UShort_t mPatternNCols;             ///< pattern span in columns
  UShort_t mPatternMinRow;            ///< pattern start row
  UShort_t mPatternMinCol;            ///< pattern start column
  UChar_t mPattern[kMaxPatternBytes]; ///< cluster topology
  //
  ClassDef(Cluster, CLUSTER_VERSION + 1)
#else
  ClassDef(Cluster, CLUSTER_VERSION)
#endif
};
//______________________________________________________
inline void Cluster::decreaseClusterUsage()
{
  // decrease cluster usage counter
  int n = getClusterUsage();
  if (n)
    setClusterUsage(--n);
  //
}

//______________________________________________________
inline void Cluster::setClusterUsage(Int_t n)
{
  // set cluster usage counter
  mNxNzN &= ~(kMaskClUse << kOffsClUse);
  mNxNzN |= (n & kMaskClUse) << kOffsClUse;
  if (n < 2)
    SetBit(kShared, kFALSE);
  if (!n)
    SetBit(kUsed, kFALSE);
}
}
}

#endif /* ALICEO2_ITS_CLUSTER_H */
