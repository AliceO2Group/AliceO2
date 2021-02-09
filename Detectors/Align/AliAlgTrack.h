#ifndef ALIALGTRACK_H
#define ALIALGTRACK_H

#include "AliExternalTrackParam.h"
#include "AliAlgPoint.h"
#include <TObjArray.h>
#include <TArrayD.h>
#include <TArrayI.h>

/*--------------------------------------------------------
  Track model for the alignment: AliExternalTrackParam for kinematics
  proper with number of multiple scattering kinks.
  Full support for derivatives and residuals calculation
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch

//#define DEBUG 4

class AliAlgTrack: public AliExternalTrackParam
{
 public:

  enum {kCosmicBit=BIT(14),kFieldONBit=BIT(15),kResidDoneBit=BIT(16),
	kDerivDoneBit=BIT(17),kKalmanDoneBit=BIT(18)};
  enum {kNKinParBOFF=4                       // N params for ExternalTrackParam part w/o field
	,kNKinParBON=5                       // N params for ExternalTrackParam part with field
	,kParY = 0                           // track parameters
	,kParZ
	,kParSnp
	,kParTgl
	,kParQ2Pt
  };
  AliAlgTrack();
  virtual ~AliAlgTrack();
  void         DefineDOFs();
  Double_t     GetMass()                         const {return fMass;}
  Double_t     GetMinX2X0Pt2Account()            const {return fMinX2X0Pt2Account;}
  Int_t        GetNPoints()                      const {return fPoints.GetEntriesFast();}
  AliAlgPoint* GetPoint(int i)                   const {return (AliAlgPoint*)fPoints[i];}
  void         AddPoint(AliAlgPoint* p)                {fPoints.AddLast(p);}
  void         SetMass(double m)                       {fMass = m;}
  void         SetMinX2X0Pt2Account(double v)          {fMinX2X0Pt2Account = v;}
  Int_t        GetNLocPar()                      const {return fNLocPar;}
  Int_t        GetNLocExtPar()                   const {return fNLocExtPar;}
  Int_t        GetInnerPointID()                 const {return fInnerPointID;}
  AliAlgPoint* GetInnerPoint()                   const {return GetPoint(fInnerPointID);}
  //
  virtual void Clear(Option_t *opt="");
  virtual void Print(Option_t *opt="")           const;
  virtual void DumpCoordinates()                 const;
  //
  Bool_t PropagateToPoint(AliExternalTrackParam& tr, const AliAlgPoint* pnt, 
			  int minNSteps,double maxStep,Bool_t matCor, double* matPar=0);
  Bool_t PropagateParamToPoint(AliExternalTrackParam& tr, const AliAlgPoint* pnt, double maxStep=3); // param only
  Bool_t PropagateParamToPoint(AliExternalTrackParam* trSet, int nTr, const AliAlgPoint* pnt, double maxStep=3); // params only
  //
  Bool_t CalcResiduals(const double *params=0);
  Bool_t CalcResidDeriv(double *params=0);
  Bool_t CalcResidDerivGlo(AliAlgPoint* pnt);
  //
  Bool_t IsCosmic()                              const {return TestBit(kCosmicBit);}
  void   SetCosmic(Bool_t v=kTRUE)                     {SetBit(kCosmicBit,v);}
  Bool_t GetFieldON()                            const {return TestBit(kFieldONBit);}
  void   SetFieldON(Bool_t v=kTRUE)                    {SetBit(kFieldONBit,v);}
  Bool_t GetResidDone()                          const {return TestBit(kResidDoneBit);}
  void   SetResidDone(Bool_t v=kTRUE)                  {SetBit(kResidDoneBit,v);}
  Bool_t GetDerivDone()                          const {return TestBit(kDerivDoneBit);}
  void   SetDerivDone(Bool_t v=kTRUE)                  {SetBit(kDerivDoneBit,v);}
  Bool_t GetKalmanDone()                         const {return TestBit(kKalmanDoneBit);}
  void   SetKalmanDone(Bool_t v=kTRUE)                 {SetBit(kKalmanDoneBit,v);}
  //
  void   SortPoints();
  Bool_t IniFit();
  Bool_t ResidKalman();
  Bool_t ProcessMaterials();
  Bool_t CombineTracks(AliExternalTrackParam& trcL, const AliExternalTrackParam& trcU);
  //
  void     SetChi2(double c)                           {fChi2 = c;};
  Double_t GetChi2()                             const {return fChi2;}
  void     SetChi2Ini(double c)                        {fChi2Ini = c;};
  Double_t GetChi2Ini()                          const {return fChi2Ini;}
  Double_t GetChi2CosmUp()                       const {return fChi2CosmUp;}
  Double_t GetChi2CosmDn()                       const {return fChi2CosmDn;}
  //
  void   ImposePtBOff(double pt)                       {fP[kParQ2Pt] = 1./pt;}
  // propagation methods
  void   CopyFrom(const AliExternalTrackParam* etp);
  Bool_t ApplyMatCorr(AliExternalTrackParam& trPar, const Double_t *corrDiag, const AliAlgPoint* pnt);
  Bool_t ApplyMatCorr(AliExternalTrackParam* trSet, int ntr, const Double_t *corrDiaf,const AliAlgPoint* pnt);
  Bool_t ApplyMatCorr(AliExternalTrackParam& trPar, const Double_t *corrpar);
  //
  Double_t  GetResidual(int dim, int pntID)       const {return fResidA[dim][pntID];}
  Double_t *GetDResDLoc(int dim, int pntID)       const {return &fDResDLocA[dim][pntID*fNLocPar];}
  Double_t *GetDResDGlo(int dim, int id)          const {return &fDResDGloA[dim][id];}
  Int_t    *GetGloParID()                         const {return fGloParIDA;}
  //
  void SetParams(AliExternalTrackParam& tr, double x, double alp, const double* par,Bool_t add);
  void SetParams(AliExternalTrackParam* trSet, int ntr, double x, double alp, const double* par,Bool_t add);
  void SetParam(AliExternalTrackParam& tr, int par, double val);
  void SetParam(AliExternalTrackParam* trSet, int ntr, int par, double val);
  void ModParam(AliExternalTrackParam& tr, int par, double delta);
  void ModParam(AliExternalTrackParam* trSet, int ntr, int par, double delta);
  //
  void RichardsonDeriv(const AliExternalTrackParam* trSet, const double *delta, 
		       const AliAlgPoint* pnt, double& derY, double& derZ);
  //
  const Double_t* GetLocPars()                    const {return fLocParA;}
  void            SetLocPars(const double* pars);
  //
 protected: 
  //
  Bool_t CalcResidDeriv(double *params,Bool_t invert,int pFrom,int pTo);
  Bool_t CalcResiduals(const double *params,Bool_t invert,int pFrom,int pTo);
  Bool_t FitLeg(AliExternalTrackParam& trc, int pFrom,int pTo, Bool_t &inv);
  Bool_t ProcessMaterials(AliExternalTrackParam& trc, int pFrom,int pTo);
  //
  void   CheckExpandDerGloBuffer(int minSize);
  //
  static Double_t RichardsonExtrap(double *val, int ord=1);
  static Double_t RichardsonExtrap(const double *val, int ord=1);
  //
  // ---------- dummies ----------
  AliAlgTrack(const AliAlgTrack&);
  AliAlgTrack& operator=(const AliAlgTrack&);
  //
 protected:

  //
  Int_t     fNLocPar;                    // number of local params
  Int_t     fNLocExtPar;                 // number of local params for the external track param
  Int_t     fNGloPar;                    // number of free global parameters the track depends on
  Int_t     fNDF;                        // number of degrees of freedom
  Int_t     fInnerPointID;               // ID of inner point in sorted track. For 2-leg cosmics - innermost point of lower leg
  Bool_t    fNeedInv[2];                 // set if one of cosmic legs need inversion
  Double_t  fMinX2X0Pt2Account;          // minimum X2X0/pT accumulated between 2 points worth to account
  Double_t  fMass;                       // assumed mass
  Double_t  fChi2;                       // chi2 with current residuals
  Double_t  fChi2CosmUp;                 // chi2 for cosmic upper leg
  Double_t  fChi2CosmDn;                 // chi2 for cosmic down leg
  Double_t  fChi2Ini;                    // chi2 with current residuals
  TObjArray fPoints;                     // alignment points
  TArrayD   fResid[2];                   // residuals array
  TArrayD   fDResDLoc[2];                // array for derivatives over local params
  TArrayD   fDResDGlo[2];                // array for derivatives over global params
  TArrayD   fLocPar;                     // local parameters array
  TArrayI   fGloParID;                   // IDs of relevant global params
  Double_t  *fResidA[2];                 //! fast access to residuals
  Double_t  *fDResDLocA[2];              //! fast access to local derivatives
  Double_t  *fDResDGloA[2];              //! fast access to global derivatives
  Int_t     *fGloParIDA;                 //! fast access to relevant global param IDs
  Double_t  *fLocParA;                   //! fast access to local params
  //
  ClassDef(AliAlgTrack,2)
};

//____________________________________________________________________________________________
inline void AliAlgTrack::SetParams(AliExternalTrackParam& tr, double x, double alp, const double* par,Bool_t add)
{
  // set track params
  const float kDefQ2PtCosm = 1., kDefG2PtColl = 1./0.6;
  tr.SetParamOnly(x,alp,par);
  double *parTr = (double*) tr.GetParameter();
  if (add) { // par is correction to reference params
    for (int i=kNKinParBON;i--;) parTr[i] += GetParameter()[i];
  }
  if (!GetFieldON()) parTr[4] = IsCosmic() ? kDefQ2PtCosm : kDefG2PtColl; // only 4 params are valid
  //
}

//____________________________________________________________________________________________
inline void AliAlgTrack::SetParams(AliExternalTrackParam* trSet, int ntr, double x, double alp, const double* par,Bool_t add)
{
  // set parames for multiple tracks (VECTORIZE THIS)
  if (!add) { // full parameter supplied
    for (int itr=ntr;itr--;) SetParams(trSet[itr],x,alp,par,kFALSE);
    return;
  }
  double partr[kNKinParBON]={0}; // par is a correction to reference parameter
  for (int i=fNLocExtPar;i--;) partr[i] = GetParameter()[i] + par[i];
  for (int itr=ntr;itr--;) SetParams(trSet[itr],x,alp,partr,kFALSE);
}

//____________________________________________________________________________________________
inline void AliAlgTrack::SetParam(AliExternalTrackParam& tr, int par, double val)
{
  // set track parameter
  ((double*)tr.GetParameter())[par] = val;
}

//____________________________________________________________________________________________
inline void AliAlgTrack::SetParam(AliExternalTrackParam* trSet, int ntr, int par, double val)
{
  // set parames for multiple tracks (VECTORIZE THIS)
  for (int itr=ntr;itr--;) ((double*)trSet[itr].GetParameter())[par] = val;
}

//____________________________________________________________________________________________
inline void AliAlgTrack::ModParam(AliExternalTrackParam & tr, int par, double delta)
{
  // modify track parameter
  ((double*)tr.GetParameter())[par] += delta;
}

//____________________________________________________________________________________________
inline void AliAlgTrack::ModParam(AliExternalTrackParam* trSet, int ntr, int par, double delta)
{
  // modify track parameter (VECTORIZE THOS)
  for (int itr=ntr;itr--;) ModParam(trSet[itr],par,delta);
}

//______________________________________________
inline void AliAlgTrack::CopyFrom(const AliExternalTrackParam* etp)
{
  // assign kinematics
  Set(etp->GetX(),etp->GetAlpha(),etp->GetParameter(),etp->GetCovariance());
}


#endif
