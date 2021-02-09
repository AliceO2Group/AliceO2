#ifndef ALIALGVOL_H
#define ALIALGVOL_H

#include <TNamed.h>
#include <TObjArray.h>
#include <TGeoMatrix.h>
#include <stdio.h>

class TObjArray;
class TClonesArray;
class AliAlgDOFStat;
class TH1;


/*--------------------------------------------------------
  Base class of alignable volume. Has at least geometric 
  degrees of freedom + user defined calibration DOFs.
  The name provided to constructor must be the SYMNAME which
  AliGeomManager can trace to geometry.
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgVol : public TNamed
{
 public:
  enum DOFGeom_t {kDOFTX,kDOFTY,kDOFTZ,kDOFPS,kDOFTH,kDOFPH,kNDOFGeom,kAllGeomDOF=0x3F};
  enum {kDOFBitTX=BIT(kDOFTX),kDOFBitTY=BIT(kDOFTY),kDOFBitTZ=BIT(kDOFTZ),
	kDOFBitPS=BIT(kDOFPS),kDOFBitTH=BIT(kDOFTH),kDOFBitPH=BIT(kDOFPH)};
  enum {kNDOFMax=32};
  enum Frame_t {kLOC,kTRA,kNVarFrames};  // variation frames defined
  enum {kInitDOFsDoneBit=BIT(14),kSkipBit=BIT(15),kExclFromParentConstraintBit=BIT(16)};
  enum {kDefChildConstr=0xff};
  //
  AliAlgVol(const char* symname=0, int iid=0);
  virtual ~AliAlgVol();
  //
  const char* GetSymName()                       const {return GetName();}
  //
  Int_t      GetVolID()                          const {return (Int_t)GetUniqueID();}
  void       SetVolID(Int_t v)                         {SetUniqueID(v);}
  Int_t      GetInternalID()                     const {return fIntID;}
  void       SetInternalID(Int_t v)                    {fIntID = v;}
  //
  //
  void       AssignDOFs(Int_t &cntDOFs, Float_t *pars, Float_t *errs, Int_t *labs);
  void       InitDOFs();
  //
  Frame_t    GetVarFrame()                       const {return fVarFrame;}
  void       SetVarFrame(Frame_t f)                    {fVarFrame = f;}
  Bool_t     IsFrameTRA()                        const {return fVarFrame == kTRA;}
  Bool_t     IsFrameLOC()                        const {return fVarFrame == kLOC;}
  //
  void       SetFreeDOF(Int_t dof)                     {fDOF |= 0x1<<dof; CalcFree();}
  void       FixDOF(Int_t dof)                         {fDOF &=~(0x1<<dof); CalcFree();}
  void       SetFreeDOFPattern(UInt_t pat)             {fDOF = pat; CalcFree();}
  Bool_t     IsFreeDOF(Int_t dof)                const {return (fDOF&(0x1<<dof))!=0;}
  Bool_t     IsCondDOF(Int_t dof)                const;
  UInt_t     GetFreeDOFPattern()                 const {return fDOF;}
  UInt_t     GetFreeDOFGeomPattern()             const {return fDOF&kAllGeomDOF;}
  //
  void       AddAutoConstraints(TObjArray* constrArr);
  Bool_t     IsChildrenDOFConstrained(Int_t dof) const {return fConstrChild&0x1<<dof;}
  UChar_t    GetChildrenConstraintPattern()      const {return fConstrChild;}
  void       ConstrainChildrenDOF(Int_t dof)            {fConstrChild |= 0x1<<dof;}
  void       UConstrainChildrenDOF(Int_t dof)           {fConstrChild &=~(0x1<<dof);}
  void       SetChildrenConstrainPattern(UInt_t pat)   {fConstrChild = pat;}
  Bool_t     HasChildrenConstraint()             const {return  fConstrChild;}
  //
  AliAlgVol* GetParent()                         const {return fParent;}
  void       SetParent(AliAlgVol* par)                 {fParent = par; if (par) par->AddChild(this);}
  Int_t      CountParents()                      const;
  //
  Int_t      GetNChildren()                      const {return fChildren ? fChildren->GetEntriesFast():0;}
  AliAlgVol* GetChild(int i)                     const {return fChildren ? (AliAlgVol*)fChildren->UncheckedAt(i):0;}
  virtual void AddChild(AliAlgVol* ch);
  //
  Double_t   GetXTracking()                      const {return fX;}
  Double_t   GetAlpTracking()                    const {return fAlp;}
  //
  Int_t      GetNProcessedPoints()               const {return fNProcPoints;}
  virtual Int_t FinalizeStat(AliAlgDOFStat* h=0);
  void       FillDOFStat(AliAlgDOFStat* h)       const;
  //
  Float_t*   GetParVals()                        const {return fParVals;}
  Double_t   GetParVal(int par)                  const {return fParVals[par];}
  Double_t   GetParErr(int par)                  const {return fParErrs[par];}
  Int_t      GetParLab(int par)                  const {return fParLabs[par];}
  void       GetParValGeom(double* delta)        const {for (int i=kNDOFGeom;i--;) delta[i]=fParVals[i];}
  //
  void       SetParVals(Int_t npar,Double_t *vl,Double_t *er);
  void       SetParVal(Int_t par,Double_t v=0)          {fParVals[par] = v;}
  void       SetParErr(Int_t par,Double_t e=0)          {fParErrs[par] = e;}
  //
  Int_t      GetNDOFs()                          const  {return fNDOFs;}
  Int_t      GetNDOFFree()                       const  {return fNDOFFree;}
  Int_t      GetNDOFGeomFree()                   const  {return fNDOFGeomFree;}
  Int_t      GetFirstParGloID()                  const  {return fFirstParGloID;}
  Int_t      GetParGloID(Int_t par)              const  {return fFirstParGloID+par;}
  void       SetFirstParGloID(Int_t id)                 {fFirstParGloID=id;}
  //
  virtual void   PrepareMatrixT2L();
  virtual void   SetTrackingFrame();
  //
  const TGeoHMatrix&  GetMatrixL2G()             const {return fMatL2G;}
  const TGeoHMatrix&  GetMatrixL2GIdeal()        const {return fMatL2GIdeal;}
  const TGeoHMatrix&  GetMatrixL2GReco()         const {return fMatL2GReco;}
  const TGeoHMatrix&  GetGlobalDeltaRef()        const {return fMatDeltaRefGlo;}
  void  SetMatrixL2G(const TGeoHMatrix& m)             {fMatL2G = m;}
  void  SetMatrixL2GIdeal(const TGeoHMatrix& m)        {fMatL2GIdeal = m;}
  void  SetMatrixL2GReco(const TGeoHMatrix& m)         {fMatL2GReco = m;}
  void  SetGlobalDeltaRef(TGeoHMatrix& mat)            {fMatDeltaRefGlo = mat;}
  //
  virtual void   PrepareMatrixL2G(Bool_t reco=kFALSE);
  virtual void   PrepareMatrixL2GIdeal();
  virtual void   UpdateL2GRecoMatrices(const TClonesArray* algArr,const TGeoHMatrix* cumulDelta);
  //
  void  GetMatrixT2G(TGeoHMatrix& m)             const;
  //
  const TGeoHMatrix&  GetMatrixT2L()             const {return fMatT2L;}
  void  SetMatrixT2L(const TGeoHMatrix& m);
  //
  void  Delta2Matrix(TGeoHMatrix& deltaM, const Double_t *delta)         const;
  //
  // preparation of variation matrices
  void GetDeltaT2LmodLOC(TGeoHMatrix& matMod, const Double_t *delta) const;
  void GetDeltaT2LmodTRA(TGeoHMatrix& matMod, const Double_t *delta) const;
  void GetDeltaT2LmodLOC(TGeoHMatrix& matMod, const Double_t *delta, const TGeoHMatrix& relMat) const;
  void GetDeltaT2LmodTRA(TGeoHMatrix& matMod, const Double_t *delta, const TGeoHMatrix& relMat) const;
  //
  // creation of global matrices for storage
  void CreateGloDeltaMatrix(TGeoHMatrix& deltaM) const;
  void CreateLocDeltaMatrix(TGeoHMatrix& deltaM) const;
  void CreatePreGloDeltaMatrix(TGeoHMatrix &deltaM) const;
  void CreatePreLocDeltaMatrix(TGeoHMatrix &deltaM) const;
  void CreateAlignmenMatrix(TGeoHMatrix& alg) const;
  void CreateAlignmentObjects(TClonesArray* arr) const;
  //
  void    SetSkip(Bool_t v=kTRUE)                   {SetBit(kSkipBit,v);}
  Bool_t  GetSkip()                           const {return TestBit(kSkipBit);}
  //
  void    ExcludeFromParentConstraint(Bool_t v=kTRUE) {SetBit(kExclFromParentConstraintBit,v);}
  Bool_t  GetExcludeFromParentConstraint()    const {return TestBit(kExclFromParentConstraintBit);}
 //
  void    SetInitDOFsDone()                         {SetBit(kInitDOFsDoneBit);}
  Bool_t  GetInitDOFsDone()                   const {return TestBit(kInitDOFsDoneBit);}
  //
  Bool_t      OwnsDOFID(Int_t id)             const;
  AliAlgVol*  GetVolOfDOFID(Int_t id)         const;
  //
  virtual Bool_t IsSensor()                   const {return kFALSE;}
  //
  virtual const char* GetDOFName(int i)       const;
  virtual void   Print(const Option_t *opt="")  const;
  virtual void   WritePedeInfo(FILE* parOut, const Option_t *opt="") const;
  //
  static const char*   GetGeomDOFName(int i)     {return i<kNDOFGeom ? fgkDOFName[i] : 0;}
  static void    SetDefGeomFree(UChar_t patt)    {fgDefGeomFree = patt;}
  static UChar_t GetDefGeomFree()                {return fgDefGeomFree;}
  //
 protected:
  void       SetNDOFs(Int_t n=kNDOFGeom);
  void       CalcFree(Bool_t condFree=kFALSE);
  //
  // ------- dummies -------
  AliAlgVol(const AliAlgVol&);
  AliAlgVol& operator=(const AliAlgVol&);
  //
 protected:
  //
  Frame_t    fVarFrame;               // Variation frame for this volume
  Int_t      fIntID;                  // internal id within the detector
  Double_t   fX;                      // tracking frame X offset
  Double_t   fAlp;                    // tracking frame alpa
  //
  Char_t     fNDOFs;                  // number of degrees of freedom, including fixed ones
  UInt_t     fDOF;                    // bitpattern degrees of freedom
  Char_t     fNDOFGeomFree;           // number of free geom degrees of freedom
  Char_t     fNDOFFree;               // number of all free degrees of freedom
  UChar_t    fConstrChild;            // bitpattern for constraints on children corrections
  //
  AliAlgVol* fParent;                 // parent volume
  TObjArray* fChildren;               // array of childrens
  //
  Int_t      fNProcPoints;            // n of processed points
  Int_t      fFirstParGloID;          // ID of the 1st parameter in the global results array
  Float_t*   fParVals;                //! values of the fitted params
  Float_t*   fParErrs;                //! errors of the fitted params
  Int_t*     fParLabs;                //! labels for parameters
  //
  TGeoHMatrix fMatL2GReco;            // local to global matrix used for reco of data being processed
  TGeoHMatrix fMatL2G;                // local to global matrix, including current alignment
  TGeoHMatrix fMatL2GIdeal;           // local to global matrix, ideal
  TGeoHMatrix fMatT2L;                // tracking to local matrix (ideal)
  TGeoHMatrix fMatDeltaRefGlo;        // global reference delta from Align/Data
  //
  static const char* fgkDOFName[kNDOFGeom];
  static const char* fgkFrameName[kNVarFrames];
  static UInt_t      fgDefGeomFree;
  //
  ClassDef(AliAlgVol,2)
};

//___________________________________________________________
inline void AliAlgVol::GetMatrixT2G(TGeoHMatrix& m) const
{
  // compute tracking to global matrix, i.e. glo = T2G*tra = L2G*loc = L2G*T2L*tra
  m = GetMatrixL2GIdeal();
  m *= GetMatrixT2L();
}


#endif
