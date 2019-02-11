
// Author: ruben.shahoyan@cern.ch   20/03/2007

///////////////////////////////////////////////////////////////////////////////////
//                                                                               //
//  Wrapper for the set of mag.field parameterizations by Chebyshev polinomials  //
//  To obtain the field in cartesian coordinates/components use                  //
//    Field(double* xyz, double* bxyz);                                          //
//  For cylindrical coordinates/components:                                      //
//    FieldCyl(double* rphiz, double* brphiz)                                    //
//                                                                               //
//  The solenoid part is parameterized in the volume  R<500, -550<Z<550 cm       //
//                                                                               //
//  The region R<423 cm,  -343.3<Z<481.3 for 30kA and -343.3<Z<481.3 for 12kA    //
//  is parameterized using measured data while outside the Tosca calculation     //
//  is used (matched to data on the boundary of the measurements)                //
//                                                                               //
//  Two options are possible:                                                    //
//  1) _BRING_TO_BOUNDARY_ is defined in the AliCheb3D:                          //
//     If the querried point is outside of the validity region then the field    //
//     at the closest point on the fitted surface is returned.                   //
//  2) _BRING_TO_BOUNDARY_ is not defined in the AliCheb3D:                      //
//     If the querried point is outside of the validity region the return        //
//     value for the field components are set to 0.                              //
//                                                                               //
//  To obtain the field integral in the TPC region from given point to nearest   //
//  cathod plane (+- 250 cm) use:                                                //
//  GetTPCInt(double* xyz, double* bxyz);  for Cartesian frame                   //
//  or                                                                           //
//  GetTPCIntCyl(Double_t *rphiz, Double_t *b); for Cylindrical frame            //
//                                                                               //
//                                                                               //
//  The units are kiloGauss and cm.                                              //
//                                                                               //
///////////////////////////////////////////////////////////////////////////////////

#ifndef ALIMAGWRAPCHEB_H
#define ALIMAGWRAPCHEB_H

#include <TMath.h>
#include <TNamed.h>
#include <TObjArray.h>
#include <TStopwatch.h>
#include "AliCheb3D.h"

#ifndef _MAGCHEB_CACHE_
#define _MAGCHEB_CACHE_  // use to spead up, but then Field calls are not thread safe
#endif

#ifndef _USE_FAST_ATAN2_
#define _USE_FAST_ATAN2_ // use approximate atan2 calculation, still precision is ~10^-4 of nominal field
#endif

class TSystem;
class TArrayF;
class TArrayI;

class AliMagWrapCheb: public TNamed
{
 public:
  AliMagWrapCheb();
  AliMagWrapCheb(const AliMagWrapCheb& src);
  ~AliMagWrapCheb() {Clear();}
  //
  void       CopyFrom(const AliMagWrapCheb& src);
  AliMagWrapCheb& operator=(const AliMagWrapCheb& rhs);
  virtual void Clear(const Option_t * = "");
  //
  Int_t      GetNParamsSol()                              const {return fNParamsSol;}
  Int_t      GetNSegZSol()                                const {return fNZSegSol;}
  Float_t*   GetSegZSol()                                 const {return fSegZSol;}
  //
  Int_t      GetNParamsTPCInt()                           const {return fNParamsTPC;}
  Int_t      GetNSegZTPCInt()                             const {return fNZSegTPC;}
  //
  Int_t      GetNParamsTPCRatInt()                        const {return fNParamsTPCRat;}
  Int_t      GetNSegZTPCRatInt()                          const {return fNZSegTPCRat;}
  //
  Int_t      GetNParamsDip()                              const {return fNParamsDip;}
  Int_t      GetNSegZDip()                                const {return fNZSegDip;}
  //
  Float_t    GetMaxZ()                                    const {return GetMaxZSol();}
  Float_t    GetMinZ()                                    const {return fParamsDip ? GetMinZDip() : GetMinZSol();}
  //
  Float_t    GetMinZSol()                                 const {return fMinZSol;}
  Float_t    GetMaxZSol()                                 const {return fMaxZSol;}
  Float_t    GetMaxRSol()                                 const {return fMaxRSol;}
  //
  Float_t    GetMinZDip()                                 const {return fMinZDip;}
  Float_t    GetMaxZDip()                                 const {return fMaxZDip;}
  //
  Float_t    GetMinZTPCInt()                              const {return fMinZTPC;}
  Float_t    GetMaxZTPCInt()                              const {return fMaxZTPC;}
  Float_t    GetMaxRTPCInt()                              const {return fMaxRTPC;}
  //
  Float_t    GetMinZTPCRatInt()                           const {return fMinZTPCRat;}
  Float_t    GetMaxZTPCRatInt()                           const {return fMaxZTPCRat;}
  Float_t    GetMaxRTPCRatInt()                           const {return fMaxRTPCRat;}
  //
  AliCheb3D* GetParamSol(Int_t ipar)                      const {return (AliCheb3D*)fParamsSol->UncheckedAt(ipar);}
  AliCheb3D* GetParamTPCRatInt(Int_t ipar)                const {return (AliCheb3D*)fParamsTPCRat->UncheckedAt(ipar);}
  AliCheb3D* GetParamTPCInt(Int_t ipar)                   const {return (AliCheb3D*)fParamsTPC->UncheckedAt(ipar);}
  AliCheb3D* GetParamDip(Int_t ipar)                      const {return (AliCheb3D*)fParamsDip->UncheckedAt(ipar);}
  //
  
  Int_t GetDipSegmentsForZSlice(int zid, TObjArray& arr)  const;
  Float_t* GetDipZSegArray()                              const {return fSegZDip;}

  virtual void Print(Option_t * = "")                     const;
  //
  virtual void Field(const Double_t *xyz, Double_t *b)    const;
  Double_t     GetBz(const Double_t *xyz)                 const;
  //
  void FieldCyl(const Double_t *rphiz, Double_t  *b)      const;  
  void GetTPCInt(const Double_t *xyz, Double_t *b)        const;
  void GetTPCIntCyl(const Double_t *rphiz, Double_t *b)   const;
  void GetTPCRatInt(const Double_t *xyz, Double_t *b)     const;
  void GetTPCRatIntCyl(const Double_t *rphiz, Double_t *b) const;
  //
  Int_t       FindSolSegment(const Double_t *xyz)         const; 
  Int_t       FindTPCSegment(const Double_t *xyz)         const; 
  Int_t       FindTPCRatSegment(const Double_t *xyz)      const; 
  Int_t       FindDipSegment(const Double_t *xyz)         const; 
  static void CylToCartCylB(const Double_t *rphiz, const Double_t *brphiz,Double_t *bxyz);
  static void CylToCartCartB(const Double_t *xyz,  const Double_t *brphiz,Double_t *bxyz);
  static void CartToCylCartB(const Double_t *xyz,  const Double_t *bxyz,  Double_t *brphiz);
  static void CartToCylCylB(const Double_t *rphiz, const Double_t *bxyz,  Double_t *brphiz);
  static void CartToCyl(const Double_t *xyz,  Double_t *rphiz);
  static void CylToCart(const Double_t *rphiz,Double_t *xyz);
  //
#ifdef  _INC_CREATION_ALICHEB3D_                          // see AliCheb3D.h for explanation
  void         LoadData(const char* inpfile);
  //
  AliMagWrapCheb(const char* inputFile);
  void       SaveData(const char* outfile)                const;
  Int_t      SegmentDimension(Float_t** seg,const TObjArray* par,int npar, int dim, 
			      Float_t xmn,Float_t xmx,Float_t ymn,Float_t ymx,Float_t zmn,Float_t zmx);
  //
  void       AddParamSol(const AliCheb3D* param);
  void       AddParamTPCInt(const AliCheb3D* param);
  void       AddParamTPCRatInt(const AliCheb3D* param);
  void       AddParamDip(const AliCheb3D* param);
  void       BuildTable(Int_t npar,TObjArray *parArr, Int_t &nZSeg, Int_t &nYSeg, Int_t &nXSeg,
			Float_t &minZ,Float_t &maxZ,Float_t **segZ,Float_t **segY,Float_t **segX,
			Int_t **begSegY,Int_t **nSegY,Int_t **begSegX,Int_t **nSegX,Int_t **segID);
  void       BuildTableSol();
  void       BuildTableDip();
  void       BuildTableTPCInt();
  void       BuildTableTPCRatInt();
  void       ResetTPCInt();
  void       ResetTPCRatInt();
  void       ResetSol();
  void       ResetDip();
  //
  //
#endif
  //
  static double useATan2(double y, double x);
  
 protected:
  void     FieldCylSol(const Double_t *rphiz, Double_t *b)    const;
  Double_t FieldCylSolBz(const Double_t *rphiz)               const;
  static double fastATan2(float y, float x);
  static double fastATan2px(float y, float x);
  static double fastATan(float x);
  //
 protected:
  //
  Int_t      fNParamsSol;            // Total number of parameterization pieces for solenoid 
  Int_t      fNZSegSol;              // number of distinct Z segments in Solenoid
  Int_t      fNPSegSol;              // number of distinct P segments in Solenoid
  Int_t      fNRSegSol;              // number of distinct R segments in Solenoid
  Float_t*   fSegZSol;               //[fNZSegSol] coordinates of distinct Z segments in Solenoid
  Float_t*   fSegPSol;               //[fNPSegSol] coordinated of P segments for each Zsegment in Solenoid
  Float_t*   fSegRSol;               //[fNRSegSol] coordinated of R segments for each Psegment in Solenoid
  Int_t*     fBegSegPSol;            //[fNPSegSol] beginning of P segments array for each Z segment
  Int_t*     fNSegPSol;              //[fNZSegSol] number of P segments for each Z segment
  Int_t*     fBegSegRSol;            //[fNPSegSol] beginning of R segments array for each P segment
  Int_t*     fNSegRSol;              //[fNPSegSol] number of R segments for each P segment
  Int_t*     fSegIDSol;              //[fNRSegSol] ID of the solenoid parameterization for given RPZ segment
  Float_t    fMinZSol;               // Min Z of Solenoid parameterization
  Float_t    fMaxZSol;               // Max Z of Solenoid parameterization
  TObjArray* fParamsSol;             // Parameterization pieces for Solenoid field
  Float_t    fMaxRSol;               // max raduis for Solenoid field
  //
  Int_t      fNParamsTPC;            // Total number of parameterization pieces for TPCint 
  Int_t      fNZSegTPC;              // number of distinct Z segments in TPCint
  Int_t      fNPSegTPC;              // number of distinct P segments in TPCint
  Int_t      fNRSegTPC;              // number of distinct R segments in TPCint
  Float_t*   fSegZTPC;               //[fNZSegTPC] coordinates of distinct Z segments in TPCint
  Float_t*   fSegPTPC;               //[fNPSegTPC] coordinated of P segments for each Zsegment in TPCint
  Float_t*   fSegRTPC;               //[fNRSegTPC] coordinated of R segments for each Psegment in TPCint
  Int_t*     fBegSegPTPC;            //[fNPSegTPC] beginning of P segments array for each Z segment
  Int_t*     fNSegPTPC;              //[fNZSegTPC] number of P segments for each Z segment
  Int_t*     fBegSegRTPC;            //[fNPSegTPC] beginning of R segments array for each P segment
  Int_t*     fNSegRTPC;              //[fNPSegTPC] number of R segments for each P segment
  Int_t*     fSegIDTPC;              //[fNRSegTPC] ID of the TPCint parameterization for given RPZ segment
  Float_t    fMinZTPC;               // Min Z of TPCint parameterization
  Float_t    fMaxZTPC;               // Max Z of TPCint parameterization
  TObjArray* fParamsTPC;             // Parameterization pieces for TPCint field
  Float_t    fMaxRTPC;               // max raduis for Solenoid field integral in TPC
  //
  Int_t      fNParamsTPCRat;         // Total number of parameterization pieces for tr.field to Bz integrals in TPC region 
  Int_t      fNZSegTPCRat;           // number of distinct Z segments in TpcRatInt
  Int_t      fNPSegTPCRat;           // number of distinct P segments in TpcRatInt
  Int_t      fNRSegTPCRat;           // number of distinct R segments in TpcRatInt
  Float_t*   fSegZTPCRat;            //[fNZSegTPCRat] coordinates of distinct Z segments in TpcRatInt
  Float_t*   fSegPTPCRat;            //[fNPSegTPCRat] coordinated of P segments for each Zsegment in TpcRatInt
  Float_t*   fSegRTPCRat;            //[fNRSegTPCRat] coordinated of R segments for each Psegment in TpcRatInt
  Int_t*     fBegSegPTPCRat;         //[fNPSegTPCRat] beginning of P segments array for each Z segment
  Int_t*     fNSegPTPCRat;           //[fNZSegTPCRat] number of P segments for each Z segment
  Int_t*     fBegSegRTPCRat;         //[fNPSegTPCRat] beginning of R segments array for each P segment
  Int_t*     fNSegRTPCRat;           //[fNPSegTPCRat] number of R segments for each P segment
  Int_t*     fSegIDTPCRat;           //[fNRSegTPCRat] ID of the TpcRatInt parameterization for given RPZ segment
  Float_t    fMinZTPCRat;            // Min Z of TpcRatInt parameterization
  Float_t    fMaxZTPCRat;            // Max Z of TpcRatInt parameterization
  TObjArray* fParamsTPCRat;          // Parameterization pieces for TpcRatInt field
  Float_t    fMaxRTPCRat;            // max raduis for Solenoid field ratios integral in TPC 
  //
  Int_t      fNParamsDip;            // Total number of parameterization pieces for dipole 
  Int_t      fNZSegDip;              // number of distinct Z segments in Dipole
  Int_t      fNYSegDip;              // number of distinct Y segments in Dipole
  Int_t      fNXSegDip;              // number of distinct X segments in Dipole
  Float_t*   fSegZDip;               //[fNZSegDip] coordinates of distinct Z segments in Dipole
  Float_t*   fSegYDip;               //[fNYSegDip] coordinated of Y segments for each Zsegment in Dipole
  Float_t*   fSegXDip;               //[fNXSegDip] coordinated of X segments for each Ysegment in Dipole
  Int_t*     fBegSegYDip;            //[fNZSegDip] beginning of Y segments array for each Z segment
  Int_t*     fNSegYDip;              //[fNZSegDip] number of Y segments for each Z segment
  Int_t*     fBegSegXDip;            //[fNYSegDip] beginning of X segments array for each Y segment
  Int_t*     fNSegXDip;              //[fNYSegDip] number of X segments for each Y segment
  Int_t*     fSegIDDip;              //[fNXSegDip] ID of the dipole parameterization for given XYZ segment
  Float_t    fMinZDip;               // Min Z of Dipole parameterization
  Float_t    fMaxZDip;               // Max Z of Dipole parameterization
  TObjArray* fParamsDip;             // Parameterization pieces for Dipole field
  //
#ifdef _MAGCHEB_CACHE_
  mutable AliCheb3D* fCacheSol;              //! last used solenoid patch
  mutable AliCheb3D* fCacheDip;              //! last used dipole patch
  mutable AliCheb3D* fCacheTPCInt;           //! last used patch for TPC integral
  mutable AliCheb3D* fCacheTPCRat;           //! last used patch for TPC normalized integral
#endif 
  //
  ClassDef(AliMagWrapCheb,8)         // Wrapper class for the set of Chebishev parameterizations of Alice mag.field
  //
 };


//__________________________________________________________________________________________
inline void AliMagWrapCheb::FieldCyl(const Double_t *rphiz, Double_t *b) const
{
  // compute field in Cylindircal coordinates
  //  if (rphiz[2]<GetMinZSol() || rphiz[2]>GetMaxZSol() || rphiz[0]>GetMaxRSol()) {for (int i=3;i--;) b[i]=0; return;}
  b[0] = b[1] = b[2] = 0;
  FieldCylSol(rphiz,b);
}

//__________________________________________________________________________________________________
inline void AliMagWrapCheb::CylToCartCylB(const Double_t *rphiz, const Double_t *brphiz,Double_t *bxyz)
{
  // convert field in cylindrical coordinates to cartesian system, point is in cyl.system
  Double_t btr = TMath::Sqrt(brphiz[0]*brphiz[0]+brphiz[1]*brphiz[1]);
  Double_t psiPLUSphi = useATan2(brphiz[1],brphiz[0]) + rphiz[1];
  bxyz[0] = btr*TMath::Cos(psiPLUSphi);
  bxyz[1] = btr*TMath::Sin(psiPLUSphi);
  bxyz[2] = brphiz[2];
  //
}

//__________________________________________________________________________________________________
inline void AliMagWrapCheb::CylToCartCartB(const Double_t* xyz, const Double_t *brphiz, Double_t *bxyz)
{
  // convert field in cylindrical coordinates to cartesian system, point is in cart.system
  Double_t btr = TMath::Sqrt(brphiz[0]*brphiz[0]+brphiz[1]*brphiz[1]);
  Double_t phiPLUSpsi = useATan2(xyz[1],xyz[0]) +  useATan2(brphiz[1],brphiz[0]);
  bxyz[0] = btr*TMath::Cos(phiPLUSpsi);
  bxyz[1] = btr*TMath::Sin(phiPLUSpsi);
  bxyz[2] = brphiz[2];
  //
}

//__________________________________________________________________________________________________
inline void AliMagWrapCheb::CartToCylCartB(const Double_t *xyz, const Double_t *bxyz, Double_t *brphiz)
{
  // convert field in cylindrical coordinates to cartesian system, poin is in cart.system
  Double_t btr = TMath::Sqrt(bxyz[0]*bxyz[0]+bxyz[1]*bxyz[1]);
  Double_t psiMINphi = useATan2(bxyz[1],bxyz[0]) - useATan2(xyz[1],xyz[0]);
  //
  brphiz[0] = btr*TMath::Cos(psiMINphi);
  brphiz[1] = btr*TMath::Sin(psiMINphi);
  brphiz[2] = bxyz[2];
  //
}

//__________________________________________________________________________________________________
inline void AliMagWrapCheb::CartToCylCylB(const Double_t *rphiz, const Double_t *bxyz, Double_t *brphiz)
{
  // convert field in cylindrical coordinates to cartesian system, point is in cyl.system
  Double_t btr = TMath::Sqrt(bxyz[0]*bxyz[0]+bxyz[1]*bxyz[1]);
  Double_t psiMINphi =  useATan2(bxyz[1],bxyz[0]) - rphiz[1];
  brphiz[0] = btr*TMath::Cos(psiMINphi);
  brphiz[1] = btr*TMath::Sin(psiMINphi);
  brphiz[2] = bxyz[2];
  //
}

//__________________________________________________________________________________________________
inline void AliMagWrapCheb::CartToCyl(const Double_t *xyz, Double_t *rphiz)
{
  rphiz[0] = TMath::Sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]);
  rphiz[1] = useATan2(xyz[1],xyz[0]);
  rphiz[2] = xyz[2];
}

//__________________________________________________________________________________________________
inline void AliMagWrapCheb::CylToCart(const Double_t *rphiz, Double_t *xyz)
{
  xyz[0] = rphiz[0]*TMath::Cos(rphiz[1]);
  xyz[1] = rphiz[0]*TMath::Sin(rphiz[1]);
  xyz[2] = rphiz[2];
}

//__________________________________________________________________________________________________
inline double AliMagWrapCheb::useATan2(double y, double x)
{
#ifdef _USE_FAST_ATAN2_
  return fastATan2(y,x);
#else
  return TMath::ATan2(y,x);
#endif
}

//__________________________________________________________________________________________________
inline double AliMagWrapCheb::fastATan2(float y, float x)
{
  if (x>0.) {
    return fastATan2px(y,x);
  }
  else if (x<0.) {
    return ((y<0.) ? -TMath::Pi() : TMath::Pi()) - fastATan2px(y,-x);
  }
  else {
    return y<0.f ? -0.5*TMath::Pi() : 0.5*TMath::Pi();
  }
}

//__________________________________________________________________________________________________
inline double AliMagWrapCheb::fastATan2px(float y, float x)
{
  // x guaranteed > 0
  // use atan(t)+atan(1/t)=-pi/2 for t<0 and pi/2 for t>0
  if (y<0.) {
    return (x>-y) ? fastATan(y/x) : fastATan(-x/y)-0.5*TMath::Pi();
  }
  else { 
    return (x>y) ? fastATan(y/x) : 0.5*TMath::Pi()-fastATan(x/y);
  }
}

//__________________________________________________________________________________________________
inline double AliMagWrapCheb::fastATan(float x)
{
  // x >= 0
  const float kThresh = 0.315f; // parametrization switch here
  return x>kThresh ? x*(7.85398163397448279e-01+0.273f*(1.-x)) : x/(1.+x*x*0.2815f);
}


#endif
