/// \file Cluster.h
/// \brief Cluster structure for ITS
#ifndef ALICEO2_ITS_CLUSTER_H
#define ALICEO2_ITS_CLUSTER_H

#include "Rtypes.h"                             // for Double_t, ULong_t, etc

#include "FairTimeStamp.h"                      // for FairTimeStamp

class TGeoHMatrix;

// uncomment this to have cluster topology in stored
//#define _ClusterTopology_  

#define CLUSTER_VERSION 2 

namespace AliceO2{
  namespace ITS{
    
    /// \class Cluster
    /// \brief Cluster class for the ITS
    ///

  class UpgradeGeometryTGeo;
   
class Cluster : public FairTimeStamp 
{
 public:
  enum { // frame in which the track is currently defined
     kUsed      = BIT(14)
    ,kShared    = BIT(15)
    ,kFrameLoc  = BIT(16)
    ,kFrameTrk  = BIT(17)
    ,kFrameGlo  = BIT(18)
    ,kFrameBits = kFrameLoc|kFrameTrk|kFrameGlo
    ,kSplit     = BIT(19)
  };
  //
  enum SortMode_t { // various modes
    kSortIdLocXZ  = BIT(0)    // sort according to ID, then X,Z of local frame
    ,kSortIdTrkYZ = BIT(1)    // sort according to ID, then Y,Z of tracking frame
    ,kSortBits = kSortIdLocXZ|kSortIdTrkYZ
  };
  enum {kOffsNZ=0,kMaskNZ=0xff,kOffsNX=8,kMaskNX=0xff,kOffsNPix=16,kMaskNPix=0x1ff,kOffsClUse=25,kMaskClUse=0x7f};
  //
#ifdef _ClusterTopology_
  enum {kMaxPatternBits=32*16, kMaxPatternBytes=kMaxPatternBits/8,
        kSpanMask=0x7fff,kTruncateMask=0x8000};
#endif
 public:
  Cluster();
  Cluster(const Cluster& cluster);
  virtual ~Cluster();

  //****** Basic methods ******************
  void     SetLabel(Int_t lab,Int_t i) {
    if (i>=0 && i<3) fTracks[i] = lab;
  }
  void     SetX(Float_t x) {fX = x;}
  void     SetY(Float_t y) {fY = y;}
  void     SetZ(Float_t z) {fZ = z;}
  void     SetSigmaY2(Float_t sigy2) {fSigmaY2 = sigy2;}
  void     SetSigmaZ2(Float_t sigz2) {fSigmaZ2 = sigz2;}
  void     SetSigmaYZ(Float_t sigyz) {fSigmaYZ = sigyz;};
  void     SetVolumeId(UShort_t id)  {fVolumeId = id;}

  void   IncreaseClusterUsage() {
    if (TestBit(kUsed)) SetBit(kShared);
    else SetBit(kUsed);
  }
  
  Int_t    GetLabel(Int_t i) const {return fTracks[i];}
  Float_t  GetX()            const {return fX;}
  Float_t  GetY()            const {return fY;}
  Float_t  GetZ()            const {return fZ;}
  Float_t  GetSigmaY2()      const {return fSigmaY2;}
  Float_t  GetSigmaZ2()      const {return fSigmaZ2;}
  Float_t  GetSigmaYZ()      const {return fSigmaYZ;}
  UShort_t GetVolumeId()     const {return fVolumeId;}
  //**************************************
  
  //
  Bool_t  IsFrameLoc()         const {return TestBit(kFrameLoc);}
  Bool_t  IsFrameGlo()         const {return TestBit(kFrameGlo);}
  Bool_t  IsFrameTrk()         const {return TestBit(kFrameTrk);}
  //
  Bool_t  IsSplit()            const {return TestBit(kSplit);}
  //
  void    SetFrameLoc()              {ResetBit(kFrameBits); SetBit(kFrameLoc);}
  void    SetFrameGlo()              {ResetBit(kFrameBits); SetBit(kFrameGlo);}
  void    SetFrameTrk()              {ResetBit(kFrameTrk);  SetBit(kFrameTrk);}
  //
  void    SetSplit(Bool_t v=kTRUE)   {SetBit(kSplit,v);}
  //
  void    GoToFrameGlo();
  void    GoToFrameLoc();
  void    GoToFrameTrk();
  void    GetLocalXYZ(Float_t xyz[3])                       const;
  void    GetTrackingXYZ(Float_t xyz[3])                    const; 
  //
  void    SetNxNzN(UChar_t nx,UChar_t nz,UShort_t n) {fNxNzN = ((n&kMaskNPix)
<<kOffsNPix) + ((nx&kMaskNX)<<kOffsNX) + ((nz&kMaskNZ)<<kOffsNZ);}
  void    SetClUsage(Int_t n);
  void    ModClUsage(Bool_t used=kTRUE) {used ? IncClUsage() : DecClUsage();}
  void    IncClUsage() {SetClUsage(GetClUsage()+1); IncreaseClusterUsage();}
  void    DecClUsage();
  Int_t   GetNx()      const {return (fNxNzN>>kOffsNX)&kMaskNX;}
  Int_t   GetNz()      const {return (fNxNzN>>kOffsNZ)&kMaskNZ;}
  Int_t   GetNPix()    const {return (fNxNzN>>kOffsNPix)&kMaskNPix;}
  Int_t   GetClUsage() const {return (fNxNzN>>kOffsClUse)&kMaskClUse;}
  //
  void    SetQ(UShort_t q)                                        {fCharge = q;}
  Int_t   GetQ()                                            const {return fCharge;}
  //
  virtual void                 Print(Option_t* option = "") const;
  virtual const TGeoHMatrix*   GetTracking2LocalMatrix()           const;
  virtual TGeoHMatrix*         GetMatrix(Bool_t original = kFALSE) const;
  virtual Bool_t               GetGlobalXYZ(Float_t xyz[3]) const;
  virtual Bool_t               GetGlobalCov(Float_t cov[6]) const;
  virtual Bool_t               GetXRefPlane(Float_t &xref)  const;
  //
  virtual Bool_t               IsSortable()                 const {return kTRUE;}
  virtual Bool_t               IsEqual(const TObject* obj)  const;
  virtual Int_t                Compare(const TObject* obj)  const;
  //
  UShort_t                     GetRecoInfo()                const {return fRecoInfo;}
  void                         SetRecoInfo(UShort_t v)            {fRecoInfo = v; ModClUsage(v>0);}
  //
  Bool_t  HasCommonTrack(const Cluster* cl)          const;
  //
  static  void                 SetGeom(UpgradeGeometryTGeo* gm) {fgGeom = gm;}
  static  void                 SetSortMode(SortMode_t md)   {fgMode &= ~kSortBits; fgMode |= md;}
  static  UInt_t               GetSortMode()                {return fgMode&kSortBits;}
  static  UInt_t               GetMode()                    {return fgMode;}
  static  SortMode_t           SortModeIdTrkYZ()            {return kSortIdTrkYZ;}
  static  SortMode_t           SortModeIdLocXZ()            {return kSortIdLocXZ;}
  //
#ifdef _ClusterTopology_
  Int_t    GetPatternRowSpan()                       const  {return fPatternNRows&kSpanMask;}
  Int_t    GetPatternColSpan()                       const  {return fPatternNCols&kSpanMask;}
  Bool_t   IsPatternRowsTruncated()                  const  {return fPatternNRows&kTruncateMask;}
  Bool_t   IsPatternColsTruncated()                  const  {return fPatternNRows&kTruncateMask;}
  Bool_t   IsPatternTruncated()                      const  {return IsPatternRowsTruncated()||IsPatternColsTruncated();}
  void     SetPatternRowSpan(UShort_t nr, Bool_t truncated);
  void     SetPatternColSpan(UShort_t nc, Bool_t truncated);
  void     SetPatternMinRow(UShort_t row)            {fPatternMinRow = row;}
  void     SetPatternMinCol(UShort_t col)            {fPatternMinCol = col;}
  void     ResetPattern();
  Bool_t   TestPixel(UShort_t row,UShort_t col)      const;
  void     SetPixel(UShort_t row,UShort_t col, Bool_t fired=kTRUE);
  void     GetPattern(UChar_t patt[kMaxPatternBytes]) {for(Int_t i=0; i<kMaxPatternBytes; i++) patt[i]=fPattern[i];}
  Int_t    GetPatternMinRow()                        const {return fPatternMinRow;}
  Int_t    GetPatternMinCol()                        const {return fPatternMinCol;}

#endif
  //
 protected:
  //
  Cluster &operator=(const Cluster& cluster);
  
  Int_t fTracks[3];   // MC labels
  Float_t  fX;        // X of the cluster in the tracking c.s.
  Float_t  fY;        // Y of the cluster in the tracking c.s.
  Float_t  fZ;        // Z of the cluster in the tracking c.s.
  Float_t  fSigmaY2;  // Sigma Y square of cluster
  Float_t  fSigmaZ2;  // Sigma Z square of cluster
  Float_t  fSigmaYZ;  // Non-diagonal element of cov.matrix
  UShort_t fVolumeId; // Volume ID of the detector element
  
  UShort_t                fCharge;        //  charge (for MC studies only)
  UShort_t                fRecoInfo;      //! space reserved for reco time manipulations
  Int_t                   fNxNzN;         //  effective cluster size in X (1st byte) and Z (2nd byte) directions 
                                          //  and total Npix(next 9 bits). last 7 bits are used for clusters usage counter
  static UInt_t           fgMode;         //! general mode (sorting mode etc)
  static UpgradeGeometryTGeo* fgGeom;         //! pointer on the geometry data

#ifdef  _ClusterTopology_
  UShort_t fPatternNRows;                 // pattern span in rows
  UShort_t fPatternNCols;                 // pattern span in columns
  UShort_t fPatternMinRow;                // pattern start row
  UShort_t fPatternMinCol;                // pattern start column
  UChar_t  fPattern[kMaxPatternBytes];    // cluster topology
  //
  ClassDef(Cluster,CLUSTER_VERSION+1)
#else
  ClassDef(Cluster,CLUSTER_VERSION)
#endif
    };
//______________________________________________________
inline void Cluster::DecClUsage() {
  // decrease cluster usage counter
  int n=GetClUsage(); 
  if (n) SetClUsage(--n);
  //
}

//______________________________________________________
inline void Cluster::SetClUsage(Int_t n) {
  // set cluster usage counter
  fNxNzN &= ~(kMaskClUse<<kOffsClUse);
  fNxNzN |= (n&kMaskClUse)<<kOffsClUse;
  if (n<2) SetBit(kShared,kFALSE);
  if (!n)  SetBit(kUsed,kFALSE);
}

  }
}

#endif /* ALICEO2_ITS_CLUSTER_H */
