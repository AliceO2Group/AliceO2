#ifndef ALIALGPOINT_H
#define ALIALGPOINT_H

#include <TObject.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include "AliAlgSens.h"

class AliExternalTrackParam;

/*--------------------------------------------------------
  Meausered point in the sensor. 
  The measurement is in the tracking frame.
  Apart from measurement may contain also material information.
  Cashes residuals and track positions at its reference X
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgPoint : public TObject
{
 public:
  enum {kMaterialBit=BIT(14)     // point contains material
	,kMeasurementBit=BIT(15) // point contains measurement
	,kUpdateFromTrackBit=BIT(16)  // point needs to recalculate itself using track info
	,kVaryELossBit=BIT(17)   // ELoss variation allowed
	,kUseBzOnly=BIT(18)      // use only Bz component (ITS)
	,kInvDir=BIT(19)         // propagation via this point is in decreasing X direction (upper cosmic leg)
	,kStatOK=BIT(20)         // point is accounted in global statistics
  };
  enum {kParY = 0                           // track parameters
	,kParZ
	,kParSnp
	,kParTgl
	,kParQ2Pt,
	kNMSPar=4,kNELossPar=1,kNMatDOFs=kNMSPar+kNELossPar};
  enum {kX,kY,kZ};
  //
  AliAlgPoint();
  virtual   ~AliAlgPoint() {}
  //
  void       Init();
  void       UpdatePointByTrackInfo(const AliExternalTrackParam* t);
  //
  Double_t   GetAlphaSens()           const {return fAlphaSens;}
  Double_t   GetXSens()               const {return fXSens;}
  Double_t   GetXPoint()              const {return fXSens + GetXTracking();}
  Double_t   GetXTracking()           const {return fXYZTracking[0];}
  Double_t   GetYTracking()           const {return fXYZTracking[1];}
  Double_t   GetZTracking()           const {return fXYZTracking[2];}
  const Double_t* GetYZTracking()     const {return &fXYZTracking[1];}
  const Double_t* GetXYZTracking()    const {return fXYZTracking;}
  const Double_t* GetYZErrTracking()  const {return fErrYZTracking;}
  //
  const AliAlgSens* GetSensor()       const {return fSensor;}
  UInt_t     GetVolID()               const {return fSensor->GetVolID();}
  void       SetSensor(AliAlgSens* s)       {fSensor = s;}
  Int_t      GetDetID()               const {return fDetID;}
  Int_t      GetSID()                 const {return fSID;}
  Int_t      GetMinLocVarID()         const {return fMinLocVarID;}
  Int_t      GetMaxLocVarID()         const {return fMaxLocVarID;}
  Int_t      GetNMatPar()             const;
  Bool_t     ContainsMaterial()       const {return TestBit(kMaterialBit);}
  Bool_t     ContainsMeasurement()    const {return TestBit(kMeasurementBit);}
  Bool_t     GetNeedUpdateFromTrack() const {return TestBit(kUpdateFromTrackBit);}
  Bool_t     GetELossVaried()         const {return TestBit(kVaryELossBit);}
  Bool_t     GetUseBzOnly()           const {return TestBit(kUseBzOnly);}
  Bool_t     IsInvDir()               const {return TestBit(kInvDir);}
  Bool_t     IsStatOK()               const {return TestBit(kStatOK);}
  //
  Double_t   GetXTimesRho()          const {return fXTimesRho;}
  Double_t   GetX2X0()               const {return fX2X0;}
  void       SetXTimesRho(double v)        {fXTimesRho = v;}
  void       SetX2X0(double v)             {fX2X0 = v;}
  //
  void       SetDetID(Int_t id)                       {fDetID = (Char_t)id;}
  void       SetSID(Int_t id)                         {fSID = (Short_t)id;}
  //
  void       SetMinLocVarID(Int_t id)                 {fMinLocVarID = id;}
  void       SetMaxLocVarID(Int_t id)                 {fMaxLocVarID = id;}
  void       SetELossVaried(Bool_t v=kTRUE)           {SetBit(kVaryELossBit,v);}
  void       SetContainsMaterial(Bool_t v=kTRUE)      {SetBit(kMaterialBit,v);}
  void       SetContainsMeasurement(Bool_t v=kTRUE)   {SetBit(kMeasurementBit,v);}
  void       SetNeedUpdateFromTrack(Bool_t v=kTRUE )  {SetBit(kUpdateFromTrackBit,v);}
  void       SetUseBzOnly(Bool_t v=kTRUE)             {SetBit(kUseBzOnly,v);}
  void       SetInvDir(Bool_t v=kTRUE)                {SetBit(kInvDir,v);}
  void       SetStatOK(Bool_t v=kTRUE)                {SetBit(kStatOK,v);}
  //
  void       GetResidualsDiag(const double* pos, double &resU, double &resV) const;
  void       DiagonalizeResiduals(double rY, double rZ, double &resU, double &resV) const;
  //
  void       SetAlphaSens(double a)                   {fAlphaSens = a;}
  void       SetXSens(double x)                       {fXSens = x;}
  void       SetXYZTracking(const double r[3])   {for (int i=3;i--;) fXYZTracking[i]=r[i];}
  void       SetXYZTracking(double x,double y,double z);
  void       SetYZErrTracking(double sy2, double syz, double sz2);
  void       SetYZErrTracking(const double *err) {for (int i=3;i--;) fErrYZTracking[i]=err[i];}
  Double_t   GetErrDiag(int i)             const {return fErrDiag[i];}
  //
  Double_t*  GetTrParamWSA()               const {return (Double_t*)fTrParamWSA;}
  Double_t*  GetTrParamWSB()               const {return (Double_t*)fTrParamWSB;}
  Double_t   GetTrParamWSA(int ip)         const {return fTrParamWSA[ip];}
  Double_t   GetTrParamWSB(int ip)         const {return fTrParamWSB[ip];}
  void       GetTrWSA(AliExternalTrackParam& etp) const;
  void       GetTrWSB(AliExternalTrackParam& etp) const;
  void       SetTrParamWSA(const double* param)   {for (int i=5;i--;) fTrParamWSA[i] = param[i];}
  void       SetTrParamWSB(const double* param)   {for (int i=5;i--;) fTrParamWSB[i] = param[i];}
  Double_t   GetResidY()                   const {return GetTrParamWSA(kParY) - GetYTracking();}
  Double_t   GetResidZ()                   const {return GetTrParamWSA(kParZ) - GetZTracking();}
  //
  void       SetMatCovDiagonalizationMatrix(const TMatrixD& d);
  void       SetMatCovDiag(const TVectorD& v);
  void       SetMatCovDiagElem(int i, double err2) {fMatCorrCov[i] = err2;}
  void       UnDiagMatCorr(const double* diag, double* nodiag) const;
  void       DiagMatCorr(const double* nodiag, double* diag) const;
  void       UnDiagMatCorr(const float* diag, float* nodiag) const;
  void       DiagMatCorr(const float* nodiag, float* diag) const;
  //
  void       SetMatCorrExp(Double_t *p)           {for (int i=5;i--;) fMatCorrExp[i] = p[i];}
  Float_t*   GetMatCorrExp()                const {return (float*)fMatCorrExp;}
  Float_t*   GetMatCorrCov()                const {return (float*)fMatCorrCov;}
  //
  void       GetXYZGlo(Double_t r[3])       const;
  Double_t   GetPhiGlo()                    const;
  Int_t      GetAliceSector()               const;
  //
  Int_t      GetNGloDOFs()                  const {return fNGloDOFs;}
  Int_t      GetDGloOffs()                  const {return fDGloOffs;}
  void       SetNGloDOFs(int n)                   {fNGloDOFs = n;}
  void       SetDGloOffs(int n)                   {fDGloOffs = n;}
  //
  void       IncrementStat();
  //
  virtual void DumpCoordinates()            const;
  virtual void Print(Option_t* option = "") const;
  virtual void Clear(Option_t* option = "");
  //
 protected:
  virtual Bool_t  IsSortable()                         const {return kTRUE;}
  virtual Int_t   Compare(const TObject* a)            const;
  //
  // ---------- dummies ----------
  AliAlgPoint(const AliAlgPoint&);
  AliAlgPoint& operator=(const AliAlgPoint&);
  //
 protected:
  //
  Int_t      fMinLocVarID;                             // The residuals/derivatives depend on fNLocExtPar params 
                                                       // and point params>=fMinLocVarID.
  Int_t      fMaxLocVarID;                             // The residuals/derivatives depend on fNLocExtPar params 
                                                       // and point params<fMaxLocVarID.
                                                       // If the point contains materials, fMaxLocVarID also marks
                                                       // the parameters associated with this point
  Char_t     fDetID;                                   // DetectorID
  Short_t    fSID;                                     // sensor ID in the detector
  Float_t    fAlphaSens;                               // Alpha of tracking frame
  Float_t    fXSens;                                   // X of tracking frame
  Float_t    fCosDiagErr;                              // Cos of Phi of rotation in YZ plane which diagonalize errors
  Float_t    fSinDiagErr;                              // Sin of Phi of rotation in YZ plane which diagonalize errors
  Float_t    fErrDiag[2];                              // diagonalized errors
  Double_t   fXYZTracking[3];                          // X,Y,Z in tracking frame
  Double_t   fErrYZTracking[3];                        // errors in tracking frame
  //
  Float_t    fX2X0;                                    // X2X0 seen by the track (including inclination)
  Float_t    fXTimesRho;                               // signed Density*Length seen by the track (including inclination)
  //
  Int_t      fNGloDOFs;                                // number of global DOFs this point depends on
  Int_t      fDGloOffs;                                // 1st entry slot of d/dGloPar in the AlgTrack fDResDGlo arrays
  Float_t    fMatCorrExp[kNMatDOFs];                   // material correction due to ELoss expectation (non-diagonalized)
  Float_t    fMatCorrCov[kNMatDOFs];                   // material correction delta covariance (diagonalized)
  Float_t    fMatDiag[kNMatDOFs][kNMatDOFs];           // matrix for  diagonalization of material effects errors
  //
  Double_t   fTrParamWSA[kNMatDOFs];                   // workspace for tracks params at this point AFTER material correction
  Double_t   fTrParamWSB[kNMatDOFs];                   // workspace for tracks params at this point BEFORE material correction
  //
  AliAlgSens* fSensor;                                 // sensor of this point
  //
  ClassDef(AliAlgPoint,1)
};

//____________________________________________________
inline void AliAlgPoint::SetXYZTracking(double x,double y,double z) 
{
  // assign tracking coordinates
  fXYZTracking[0]=x; fXYZTracking[1]=y; fXYZTracking[2]=z;
}

//____________________________________________________
inline void AliAlgPoint::SetYZErrTracking(double sy2,double syz,double sz2) 
{
  // assign tracking coordinates
  fErrYZTracking[0]=sy2; fErrYZTracking[1]=syz; fErrYZTracking[2]=sz2;
}

inline Int_t AliAlgPoint::GetNMatPar() const 
{
  // get number of free params for material descriptoin
  return ContainsMaterial() ? (GetELossVaried() ? kNMSPar+kNELossPar:kNMSPar) : 0;
}

//_____________________________________
inline void AliAlgPoint::DiagonalizeResiduals(double rY,double rZ, double &resU, double &resV) const
{
  // rotate residuals to frame where their error matrix is diagonal
  resU = fCosDiagErr*rY - fSinDiagErr*rZ;
  resV = fSinDiagErr*rY + fCosDiagErr*rZ;
  //
}

//_____________________________________
inline void AliAlgPoint::GetResidualsDiag(const double* pos, double &resU, double &resV) const
{
  // calculate residuals in the frame where the errors are diagonal, given the position
  // of the track in the standard tracking frame
  DiagonalizeResiduals(pos[0]-fXYZTracking[1],pos[1]-fXYZTracking[2],resU,resV);
  //
}

//__________________________________________________________________
inline void AliAlgPoint::IncrementStat() 
{
  // increment statistics for detectors this point depends on
  fSensor->IncrementStat();
  SetStatOK();
}


#endif
