// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgVol.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Base class of alignable volume

/**
 * Base class of alignable volume. Has at least geometric
 * degrees of freedom + user defined calibration DOFs.
 * The name provided to constructor must be the SYMNAME which
 * AliGeomManager can trace to geometry.
 */

#ifndef ALIALGVOL_H
#define ALIALGVOL_H

#include <TNamed.h>
#include <TObjArray.h>
#include <TGeoMatrix.h>
#include <cstdio>
#include "Align/AliAlgDOFStat.h"

class TObjArray;
class TClonesArray;
class TH1;

namespace o2
{
namespace align
{

class AliAlgVol : public TNamed
{
 public:
  enum DOFGeom_t { kDOFTX,
                   kDOFTY,
                   kDOFTZ,
                   kDOFPS,
                   kDOFTH,
                   kDOFPH,
                   kNDOFGeom,
                   kAllGeomDOF = 0x3F };
  enum { kDOFBitTX = BIT(kDOFTX),
         kDOFBitTY = BIT(kDOFTY),
         kDOFBitTZ = BIT(kDOFTZ),
         kDOFBitPS = BIT(kDOFPS),
         kDOFBitTH = BIT(kDOFTH),
         kDOFBitPH = BIT(kDOFPH) };
  enum { kNDOFMax = 32 };
  enum Frame_t { kLOC,
                 kTRA,
                 kNVarFrames }; // variation frames defined
  enum { kInitDOFsDoneBit = BIT(14),
         kSkipBit = BIT(15),
         kExclFromParentConstraintBit = BIT(16) };
  enum { kDefChildConstr = 0xff };
  //
  AliAlgVol(const char* symname = 0, int iid = 0);
  virtual ~AliAlgVol();
  //
  const char* GetSymName() const { return GetName(); }
  //
  int GetVolID() const { return (int)GetUniqueID(); }
  void SetVolID(int v) { SetUniqueID(v); }
  int GetInternalID() const { return fIntID; }
  void SetInternalID(int v) { fIntID = v; }
  //
  //
  void AssignDOFs(int& cntDOFs, float* pars, float* errs, int* labs);
  void InitDOFs();
  //
  Frame_t GetVarFrame() const { return fVarFrame; }
  void SetVarFrame(Frame_t f) { fVarFrame = f; }
  bool IsFrameTRA() const { return fVarFrame == kTRA; }
  bool IsFrameLOC() const { return fVarFrame == kLOC; }
  //
  void SetFreeDOF(int dof)
  {
    fDOF |= 0x1 << dof;
    CalcFree();
  }
  void FixDOF(int dof)
  {
    fDOF &= ~(0x1 << dof);
    CalcFree();
  }
  void SetFreeDOFPattern(uint32_t pat)
  {
    fDOF = pat;
    CalcFree();
  }
  bool IsFreeDOF(int dof) const { return (fDOF & (0x1 << dof)) != 0; }
  bool IsCondDOF(int dof) const;
  uint32_t GetFreeDOFPattern() const { return fDOF; }
  uint32_t GetFreeDOFGeomPattern() const { return fDOF & kAllGeomDOF; }
  //
  void AddAutoConstraints(TObjArray* constrArr);
  bool IsChildrenDOFConstrained(int dof) const { return fConstrChild & 0x1 << dof; }
  uint8_t GetChildrenConstraintPattern() const { return fConstrChild; }
  void ConstrainChildrenDOF(int dof) { fConstrChild |= 0x1 << dof; }
  void UConstrainChildrenDOF(int dof) { fConstrChild &= ~(0x1 << dof); }
  void SetChildrenConstrainPattern(uint32_t pat) { fConstrChild = pat; }
  bool HasChildrenConstraint() const { return fConstrChild; }
  //
  AliAlgVol* GetParent() const { return fParent; }
  void SetParent(AliAlgVol* par)
  {
    fParent = par;
    if (par)
      par->AddChild(this);
  }
  int CountParents() const;
  //
  int GetNChildren() const { return fChildren ? fChildren->GetEntriesFast() : 0; }
  AliAlgVol* GetChild(int i) const { return fChildren ? (AliAlgVol*)fChildren->UncheckedAt(i) : 0; }
  virtual void AddChild(AliAlgVol* ch);
  //
  double GetXTracking() const { return fX; }
  double GetAlpTracking() const { return fAlp; }
  //
  int GetNProcessedPoints() const { return fNProcPoints; }
  virtual int FinalizeStat(AliAlgDOFStat* h = 0);
  void FillDOFStat(AliAlgDOFStat* h) const;
  //
  float* GetParVals() const { return fParVals; }
  double GetParVal(int par) const { return fParVals[par]; }
  double GetParErr(int par) const { return fParErrs[par]; }
  int GetParLab(int par) const { return fParLabs[par]; }
  void GetParValGeom(double* delta) const
  {
    for (int i = kNDOFGeom; i--;)
      delta[i] = fParVals[i];
  }
  //
  void SetParVals(int npar, double* vl, double* er);
  void SetParVal(int par, double v = 0) { fParVals[par] = v; }
  void SetParErr(int par, double e = 0) { fParErrs[par] = e; }
  //
  int GetNDOFs() const { return fNDOFs; }
  int GetNDOFFree() const { return fNDOFFree; }
  int GetNDOFGeomFree() const { return fNDOFGeomFree; }
  int GetFirstParGloID() const { return fFirstParGloID; }
  int GetParGloID(int par) const { return fFirstParGloID + par; }
  void SetFirstParGloID(int id) { fFirstParGloID = id; }
  //
  virtual void PrepareMatrixT2L();
  virtual void SetTrackingFrame();
  //
  const TGeoHMatrix& GetMatrixL2G() const { return fMatL2G; }
  const TGeoHMatrix& GetMatrixL2GIdeal() const { return fMatL2GIdeal; }
  const TGeoHMatrix& GetMatrixL2GReco() const { return fMatL2GReco; }
  const TGeoHMatrix& GetGlobalDeltaRef() const { return fMatDeltaRefGlo; }
  void SetMatrixL2G(const TGeoHMatrix& m) { fMatL2G = m; }
  void SetMatrixL2GIdeal(const TGeoHMatrix& m) { fMatL2GIdeal = m; }
  void SetMatrixL2GReco(const TGeoHMatrix& m) { fMatL2GReco = m; }
  void SetGlobalDeltaRef(TGeoHMatrix& mat) { fMatDeltaRefGlo = mat; }
  //
  virtual void PrepareMatrixL2G(bool reco = false);
  virtual void PrepareMatrixL2GIdeal();
  virtual void UpdateL2GRecoMatrices(const TClonesArray* algArr, const TGeoHMatrix* cumulDelta);
  //
  void GetMatrixT2G(TGeoHMatrix& m) const;
  //
  const TGeoHMatrix& GetMatrixT2L() const { return fMatT2L; }
  void SetMatrixT2L(const TGeoHMatrix& m);
  //
  void Delta2Matrix(TGeoHMatrix& deltaM, const double* delta) const;
  //
  // preparation of variation matrices
  void GetDeltaT2LmodLOC(TGeoHMatrix& matMod, const double* delta) const;
  void GetDeltaT2LmodTRA(TGeoHMatrix& matMod, const double* delta) const;
  void GetDeltaT2LmodLOC(TGeoHMatrix& matMod, const double* delta, const TGeoHMatrix& relMat) const;
  void GetDeltaT2LmodTRA(TGeoHMatrix& matMod, const double* delta, const TGeoHMatrix& relMat) const;
  //
  // creation of global matrices for storage
  void CreateGloDeltaMatrix(TGeoHMatrix& deltaM) const;
  void CreateLocDeltaMatrix(TGeoHMatrix& deltaM) const;
  void CreatePreGloDeltaMatrix(TGeoHMatrix& deltaM) const;
  void CreatePreLocDeltaMatrix(TGeoHMatrix& deltaM) const;
  void CreateAlignmenMatrix(TGeoHMatrix& alg) const;
  void CreateAlignmentObjects(TClonesArray* arr) const;
  //
  void SetSkip(bool v = true) { SetBit(kSkipBit, v); }
  bool GetSkip() const { return TestBit(kSkipBit); }
  //
  void ExcludeFromParentConstraint(bool v = true) { SetBit(kExclFromParentConstraintBit, v); }
  bool GetExcludeFromParentConstraint() const { return TestBit(kExclFromParentConstraintBit); }
  //
  void SetInitDOFsDone() { SetBit(kInitDOFsDoneBit); }
  bool GetInitDOFsDone() const { return TestBit(kInitDOFsDoneBit); }
  //
  bool OwnsDOFID(int id) const;
  AliAlgVol* GetVolOfDOFID(int id) const;
  //
  virtual bool IsSensor() const { return false; }
  //
  virtual const char* GetDOFName(int i) const;
  virtual void Print(const Option_t* opt = "") const;
  virtual void WritePedeInfo(FILE* parOut, const Option_t* opt = "") const;
  //
  static const char* GetGeomDOFName(int i) { return i < kNDOFGeom ? fgkDOFName[i] : 0; }
  static void SetDefGeomFree(uint8_t patt) { fgDefGeomFree = patt; }
  static uint8_t GetDefGeomFree() { return fgDefGeomFree; }
  //
 protected:
  void SetNDOFs(int n = kNDOFGeom);
  void CalcFree(bool condFree = false);
  //
  // ------- dummies -------
  AliAlgVol(const AliAlgVol&);
  AliAlgVol& operator=(const AliAlgVol&);
  //
 protected:
  //
  Frame_t fVarFrame; // Variation frame for this volume
  int fIntID;        // internal id within the detector
  double fX;         // tracking frame X offset
  double fAlp;       // tracking frame alpa
  //
  char fNDOFs;          // number of degrees of freedom, including fixed ones
  uint32_t fDOF;        // bitpattern degrees of freedom
  char fNDOFGeomFree;   // number of free geom degrees of freedom
  char fNDOFFree;       // number of all free degrees of freedom
  uint8_t fConstrChild; // bitpattern for constraints on children corrections
  //
  AliAlgVol* fParent;   // parent volume
  TObjArray* fChildren; // array of childrens
  //
  int fNProcPoints;   // n of processed points
  int fFirstParGloID; // ID of the 1st parameter in the global results array
  float* fParVals;    //! values of the fitted params
  float* fParErrs;    //! errors of the fitted params
  int* fParLabs;      //! labels for parameters
  //
  TGeoHMatrix fMatL2GReco;     // local to global matrix used for reco of data being processed
  TGeoHMatrix fMatL2G;         // local to global matrix, including current alignment
  TGeoHMatrix fMatL2GIdeal;    // local to global matrix, ideal
  TGeoHMatrix fMatT2L;         // tracking to local matrix (ideal)
  TGeoHMatrix fMatDeltaRefGlo; // global reference delta from Align/Data
  //
  static const char* fgkDOFName[kNDOFGeom];
  static const char* fgkFrameName[kNVarFrames];
  static uint32_t fgDefGeomFree;
  //
  ClassDef(AliAlgVol, 2)
};

//___________________________________________________________
inline void AliAlgVol::GetMatrixT2G(TGeoHMatrix& m) const
{
  // compute tracking to global matrix, i.e. glo = T2G*tra = L2G*loc = L2G*T2L*tra
  m = GetMatrixL2GIdeal();
  m *= GetMatrixT2L();
}
} // namespace align
} // namespace o2
#endif
