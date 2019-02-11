#ifndef ALIEXTERNALTRACKPARAM_H
#define ALIEXTERNALTRACKPARAM_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

/*****************************************************************************
 *              "External" track parametrisation class                       *
 *                                                                           *
 *      external param0:   local Y-coordinate of a track (cm)                *
 *      external param1:   local Z-coordinate of a track (cm)                *
 *      external param2:   local sine of the track momentum azimuthal angle  *
 *      external param3:   tangent of the track momentum dip angle           *
 *      external param4:   1/pt (1/(GeV/c))                                  *
 *                                                                           *
 * The parameters are estimated at an exact position x in a local coord.     *
 * system rotated by angle alpha with respect to the global coord.system.    *
 *        Origin: I.Belikov, CERN, Jouri.Belikov@cern.ch                     *
 *****************************************************************************/
#include "TMath.h"

#include "AliVTrack.h"
#include "AliVMisc.h"
const Double_t kVeryBig=1./kAlmost0;
const Double_t kMostProbablePt=0.35;

class AliVVertex;
class TPolyMarker3D; 

const Double_t kC0max=100*100, // SigmaY<=100cm
               kC2max=100*100, // SigmaZ<=100cm
               kC5max=1*1,     // SigmaSin<=1
               kC9max=1*1,     // SigmaTan<=1
               kC14max=100*100; // Sigma1/Pt<=100 1/GeV

class AliExternalTrackParam: public AliVTrack {
 public:
  AliExternalTrackParam();
  AliExternalTrackParam(const AliExternalTrackParam &);
  AliExternalTrackParam& operator=(const AliExternalTrackParam & trkPar);
  AliExternalTrackParam(Double_t x, Double_t alpha, 
			const Double_t param[5], const Double_t covar[15]);
  AliExternalTrackParam(Double_t xyz[3],Double_t pxpypz[3],
			Double_t cv[21],Short_t sign);
  // constructor for reinitialisation of vtable
  AliExternalTrackParam( AliVConstructorReinitialisationFlag f) :AliVTrack(f), fX(), fAlpha(){}
  void Reinitialize() { new (this) AliExternalTrackParam( AliVReinitialize ); }

  virtual ~AliExternalTrackParam(){}
  void CopyFromVTrack(const AliVTrack *vTrack);
  
  template <typename T>
  void Set(T x, T alpha, const T param[5], const T covar[15]) {
    //  Sets the parameters
    if      (alpha < -TMath::Pi()) alpha += 2*TMath::Pi();
    else if (alpha >= TMath::Pi()) alpha -= 2*TMath::Pi();
    fX=x; fAlpha=alpha;
    for (Int_t i = 0; i < 5; i++)  fP[i] = param[i];
    for (Int_t i = 0; i < 15; i++) fC[i] = covar[i];

    CheckCovariance();

  }

  void SetParamOnly(double x, double alpha, const double param[5]) {
    //  Sets the parameters, neglect cov matrix
    if      (alpha < -TMath::Pi()) alpha += 2*TMath::Pi();
    else if (alpha >= TMath::Pi()) alpha -= 2*TMath::Pi();
    fX=x; fAlpha=alpha;
    for (Int_t i = 0; i < 5; i++)  fP[i] = param[i];
  }

  void Set(Double_t xyz[3],Double_t pxpypz[3],Double_t cv[21],Short_t sign);

  static void SetMostProbablePt(Double_t pt) { fgMostProbablePt=pt; }
  static Double_t GetMostProbablePt() { return fgMostProbablePt; }

  void Reset();
  void ResetCovariance(Double_t s2);
  void AddCovariance(const Double_t cov[15]);

  const Double_t *GetParameter() const {return fP;}
  const Double_t *GetCovariance() const {return fC;}
  virtual  Bool_t IsStartedTimeIntegral() const {return kFALSE;}
  virtual  void   AddTimeStep(Double_t ) {} // dummy method, real stuff is done in AliKalmanTrack
  Double_t GetAlpha() const {return fAlpha;}
  Double_t GetX() const {return fX;}
  Double_t GetY()    const {return fP[0];}
  Double_t GetZ()    const {return fP[1];}
  Double_t GetSnp()  const {return fP[2];}
  virtual Double_t GetTgl()  const {return fP[3];}
  using AliVTrack::GetImpactParameters;
  virtual void GetImpactParameters(Float_t& ,Float_t&) const {}
  Double_t GetSigned1Pt()  const {return fP[4];}

  Double_t GetSigmaY2() const {return fC[0];}
  Double_t GetSigmaZY() const {return fC[1];}
  Double_t GetSigmaZ2() const {return fC[2];}
  Double_t GetSigmaSnpY() const {return fC[3];}
  Double_t GetSigmaSnpZ() const {return fC[4];}
  Double_t GetSigmaSnp2() const {return fC[5];}
  Double_t GetSigmaTglY() const {return fC[6];}
  Double_t GetSigmaTglZ() const {return fC[7];}
  Double_t GetSigmaTglSnp() const {return fC[8];}
  Double_t GetSigmaTgl2() const {return fC[9];}
  Double_t GetSigma1PtY() const {return fC[10];}
  Double_t GetSigma1PtZ() const {return fC[11];}
  Double_t GetSigma1PtSnp() const {return fC[12];}
  Double_t GetSigma1PtTgl() const {return fC[13];}
  Double_t GetSigma1Pt2() const {return fC[14];}

  // additional functions for AliVParticle
  Double_t Px() const;
  Double_t Py() const;
  Double_t Pz() const { return Pt()*GetTgl(); }
  Double_t Pt() const { return TMath::Abs(GetSignedPt()); }
  Double_t P() const { return GetP(); }
  Bool_t   PxPyPz(Double_t p[3]) const { return GetPxPyPz(p); }
  
  Double_t Xv() const;
  Double_t Yv() const;
  Double_t Zv() const {return GetZ();}
  Bool_t   XvYvZv(Double_t x[3]) const { return GetXYZ(x); }

  Double_t OneOverPt() const { return 1./Pt(); }
  Double_t Phi() const;
  Double_t PhiPos() const;
  Double_t Theta() const;
  virtual Double_t E() const;
  virtual Double_t M() const;
  Double_t Eta() const;
  virtual Double_t Y() const;
  virtual Short_t  Charge() const { return (Short_t)GetSign(); }
  virtual const Double_t *PID() const { return 0x0; }

  // additional functions from AliVTrack
  virtual Int_t    GetID() const { return -999; }
  virtual UChar_t  GetITSClusterMap() const {return 0; }
  virtual ULong64_t GetStatus() const { return 0; }

  Double_t GetSign() const {return (fP[4]>0) ? 1 : -1;}
  Double_t GetP() const;
  Double_t GetSignedPt() const {
    return (TMath::Abs(fP[4])>kAlmost0) ? 1./fP[4]:TMath::Sign(kVeryBig,fP[4]);
  }
  Double_t Get1P() const;
  virtual Double_t GetC(Double_t b) const {return fP[4]*b*kB2C;}
  void GetDZ(Double_t x,Double_t y,Double_t z,Double_t b,Float_t dz[2]) const; 
  Double_t GetD(Double_t xv, Double_t yv, Double_t b) const; 
  Double_t GetLinearD(Double_t xv, Double_t yv) const; 

  Bool_t CorrectForMeanMaterial(Double_t xOverX0, Double_t xTimesRho, 
        Double_t mass,  Bool_t anglecorr=kFALSE,
	Double_t (*f)(Double_t)=AliExternalTrackParam::BetheBlochSolid);

  Bool_t CorrectForMeanMaterialdEdx(Double_t xOverX0, Double_t xTimesRho, 
	Double_t mass, Double_t dEdx, Bool_t anglecorr=kFALSE);

  Bool_t CorrectForMeanMaterialZA(Double_t xOverX0, Double_t xTimesRho, 
                                  Double_t mass,
                                  Double_t zOverA=0.49848,
                                  Double_t density=2.33,
                                  Double_t exEnergy=173e-9,
                                  Double_t jp1=0.20,
                                  Double_t jp2=3.00,
                                  Bool_t anglecorr=kFALSE
  );

  //
  // Bethe-Bloch formula parameterizations
  //
  static Double_t BetheBlochAleph(Double_t bg,
                                  Double_t kp1=0.76176e-1,
                                  Double_t kp2=10.632,
                                  Double_t kp3=0.13279e-4,
                                  Double_t kp4=1.8631,
                                  Double_t kp5=1.9479
				  );
  static Double_t BetheBlochGeant(Double_t bg,
                                  Double_t kp0=2.33,
                                  Double_t kp1=0.20,
                                  Double_t kp2=3.00,
                                  Double_t kp3=173e-9,
                                  Double_t kp4=0.49848
				  );
    
  static Double_t BetheBlochSolid(Double_t bg);
  static Double_t BetheBlochGas(Double_t bg);

  Double_t GetPredictedChi2(const Double_t p[2],const Double_t cov[3]) const;

  Double_t 
    GetPredictedChi2(const Double_t p[3],const Double_t covyz[3],const Double_t covxyz[3]) const;

  Double_t GetPredictedChi2(const AliExternalTrackParam *t) const;

  Bool_t 
    PropagateTo(Double_t p[3],Double_t covyz[3],Double_t covxyz[3],Double_t b);

  Double_t *GetResiduals(Double_t *p,Double_t *cov,Bool_t updated=kTRUE) const;
  Bool_t Update(const Double_t p[2],const Double_t cov[3]);
  Bool_t Rotate(Double_t alpha);
  Bool_t RotateParamOnly(Double_t alpha);
  Bool_t Invert();
  Bool_t PropagateTo(Double_t x, Double_t b);
  Bool_t PropagateParamOnlyTo(Double_t xk, Double_t b);
  Bool_t Propagate(Double_t alpha, Double_t x, Double_t b);
  Bool_t PropagateBxByBz(Double_t alpha, Double_t x, Double_t b[3]);
  Bool_t PropagateParamOnlyBxByBzTo(Double_t xk, const Double_t b[3]);
  void   Propagate(Double_t len,Double_t x[3],Double_t p[3],Double_t bz) const;
  Bool_t Intersect(Double_t pnt[3], Double_t norm[3], Double_t bz) const;

  static void g3helx3(Double_t qfield, Double_t step, Double_t vect[7]); 
  Bool_t PropagateToBxByBz(Double_t x, const Double_t b[3]);
  Bool_t RelateToVVertexBxByBzDCA(const AliVVertex *vtx, Double_t b[3], Double_t maxd,
    AliExternalTrackParam *cParam=NULL, Double_t dz[2]=NULL, Double_t dzcov[3]=NULL);

  void GetHelixParameters(Double_t h[6], Double_t b) const;
  Double_t GetDCA(const AliExternalTrackParam *p, Double_t b,
    Double_t &xthis,Double_t &xp) const;
  Double_t PropagateToDCA(AliExternalTrackParam *p, Double_t b);
  Bool_t PropagateToDCA(const AliVVertex *vtx, Double_t b, Double_t maxd,
                        Double_t dz[2]=0, Double_t cov[3]=0);
  Bool_t PropagateToDCABxByBz(const AliVVertex *vtx, Double_t b[3], 
         Double_t maxd, Double_t dz[2]=0, Double_t cov[3]=0);
  Bool_t ConstrainToVertex(const AliVVertex* vtx, Double_t b[3]);
  
  void GetDirection(Double_t d[3]) const;
  Bool_t GetPxPyPz(Double_t *p) const;  
  Bool_t GetXYZ(Double_t *p) const;
  Bool_t GetCovarianceXYZPxPyPz(Double_t cv[21]) const;
  Bool_t GetPxPyPzAt(Double_t x, Double_t b, Double_t p[3]) const;
  Bool_t GetXYZAt(Double_t x, Double_t b, Double_t r[3]) const;
  Double_t GetParameterAtRadius(Double_t r, Double_t bz, Int_t parType) const;

  Bool_t GetYAt(Double_t x,  Double_t b,  Double_t &y) const;
  Bool_t GetZAt(Double_t x,  Double_t b,  Double_t &z) const;
  Double_t GetYAtFast(Double_t x, Double_t b) const {double y=0; return GetYAt(x,b,y) ? y : -99999;}
  Double_t GetZAtFast(Double_t x, Double_t b) const {double z=0; return GetZAt(x,b,z) ? z : -99999;}
  Bool_t GetYZAt(Double_t x, Double_t b, Double_t *yz) const;
  void Print(Option_t* option = "") const;
  Double_t GetSnpAt(Double_t x,Double_t b) const;
  Bool_t GetXatLabR(Double_t r,Double_t &x, Double_t bz, Int_t dir=0) const;
  Bool_t GetXYZatR(Double_t xr,Double_t bz, Double_t *xyz=0, Double_t* alpSect=0) const;

  //Deprecated
  Bool_t CorrectForMaterial(Double_t d, Double_t x0, Double_t mass,
	 Double_t (*f)(Double_t)=AliExternalTrackParam::BetheBlochSolid);

  Bool_t GetDistance(AliExternalTrackParam *param2, Double_t x, Double_t dist[3], Double_t b);
  static Int_t GetIndex(Int_t i, Int_t j);
  Int_t GetLabel() const {return -1;} 
  Int_t PdgCode()  const {return 0;}

  //
  // visualization (M. Ivanov)
  //
  virtual void FillPolymarker(TPolyMarker3D *pol, Float_t magf, Float_t minR, Float_t maxR, Float_t stepR);
  virtual void DrawTrack(Float_t magF, Float_t minR, Float_t maxR, Float_t stepR);

  virtual Bool_t Translate(Double_t *vTrasl,Double_t *covV);

  void CheckCovariance();

  static Bool_t  GetUseLogTermMS()                {return fgUseLogTermMS;} 
  static void    SetUseLogTermMS(Bool_t v=kTRUE)  {fgUseLogTermMS = v;} 

  //---------------------------------------------------------------------------
  //--the calibration interface--
  //--to be used in online calibration/QA
  //--should also be implemented in ESD so it works offline as well
  //-----------
  virtual Int_t GetTrackParam         ( AliExternalTrackParam & ) const {return 0;}
  virtual Int_t GetTrackParamRefitted ( AliExternalTrackParam & ) const {return 0;}
  virtual Int_t GetTrackParamIp       ( AliExternalTrackParam & ) const {return 0;}
  virtual Int_t GetTrackParamTPCInner ( AliExternalTrackParam & ) const {return 0;}
  virtual Int_t GetTrackParamOp       ( AliExternalTrackParam & ) const {return 0;}
  virtual Int_t GetTrackParamCp       ( AliExternalTrackParam & ) const {return 0;}
  virtual Int_t GetTrackParamITSOut   ( AliExternalTrackParam & ) const {return 0;}

 protected:
  AliExternalTrackParam(const AliVTrack *vTrack);

/*  protected: */
 private:
  Double_t &Par(Int_t i) {return fP[i];}
  Double_t &Cov(Int_t i) {return fC[i];}
 protected:
  Double32_t           fX;     // X coordinate for the point of parametrisation
  Double32_t           fAlpha; // Local <-->global coor.system rotation angle
  Double32_t           fP[5];  // The track parameters
  Double32_t           fC[15]; // The track parameter covariance matrix

  static Double32_t    fgMostProbablePt; // "Most probable" pt
                                         // (to be used if Bz=0)
  static Bool_t        fgUseLogTermMS;   // use log term in Mult.Stattering evaluation
  ClassDef(AliExternalTrackParam, 8)
};

inline void AliExternalTrackParam::ResetCovariance(Double_t s2) {
  //
  // Reset the covarince matrix to "something big"
  //

  s2 = TMath::Abs(s2);
  Double_t fC0=fC[0]*s2,
           fC2=fC[2]*s2,
           fC5=fC[5]*s2,
           fC9=fC[9]*s2,
           fC14=fC[14]*s2;
 
  if (fC0>kC0max)  fC0 = kC0max;
  if (fC2>kC2max)  fC2 = kC2max;
  if (fC5>kC5max)  fC5 = kC5max;
  if (fC9>kC9max)  fC9 = kC9max;
  if (fC14>kC14max)  fC14 = kC14max;


    fC[0] = fC0;
    fC[1] = 0.;  fC[2] = fC2;
    fC[3] = 0.;  fC[4] = 0.;  fC[5] = fC5;
    fC[6] = 0.;  fC[7] = 0.;  fC[8] = 0.;  fC[9] = fC9;
    fC[10]= 0.;  fC[11]= 0.;  fC[12]= 0.;  fC[13]= 0.;  fC[14] = fC14;
}




#endif
