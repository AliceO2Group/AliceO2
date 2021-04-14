// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableVolume.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Base class of alignable volume

/**
 * Base class of alignable volume. Has at least geometric
 * degrees of freedom + user defined calibration DOFs.
 * The name provided to constructor must be the SYMNAME which
 * AliGeomManager can trace to geometry.
 */

#ifndef ALIGNABLEVOLUME_H
#define ALIGNABLEVOLUME_H

#include <TNamed.h>
#include <TObjArray.h>
#include <TGeoMatrix.h>
#include <cstdio>
#include "Align/DOFStatistics.h"

class TObjArray;
class TClonesArray;
class TH1;

namespace o2
{
namespace align
{

class AlignableVolume : public TNamed
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
  AlignableVolume(const char* symname = 0, int iid = 0);
  virtual ~AlignableVolume();
  //
  const char* getSymName() const { return GetName(); }
  //
  int getVolID() const { return (int)GetUniqueID(); }
  void setVolID(int v) { SetUniqueID(v); }
  int getInternalID() const { return mIntID; }
  void setInternalID(int v) { mIntID = v; }
  //
  //
  void assignDOFs(int& cntDOFs, float* pars, float* errs, int* labs);
  void initDOFs();
  //
  Frame_t getVarFrame() const { return mVarFrame; }
  void setVarFrame(Frame_t f) { mVarFrame = f; }
  bool isFrameTRA() const { return mVarFrame == kTRA; }
  bool isFrameLOC() const { return mVarFrame == kLOC; }
  //
  void setFreeDOF(int dof)
  {
    mDOF |= 0x1 << dof;
    calcFree();
  }
  void fixDOF(int dof)
  {
    mDOF &= ~(0x1 << dof);
    calcFree();
  }
  void setFreeDOFPattern(uint32_t pat)
  {
    mDOF = pat;
    calcFree();
  }
  bool isFreeDOF(int dof) const { return (mDOF & (0x1 << dof)) != 0; }
  bool isCondDOF(int dof) const;
  uint32_t getFreeDOFPattern() const { return mDOF; }
  uint32_t getFreeDOFGeomPattern() const { return mDOF & kAllGeomDOF; }
  //
  void addAutoConstraints(TObjArray* constrArr);
  bool isChildrenDOFConstrained(int dof) const { return mConstrChild & 0x1 << dof; }
  uint8_t getChildrenConstraintPattern() const { return mConstrChild; }
  void constrainChildrenDOF(int dof) { mConstrChild |= 0x1 << dof; }
  void uConstrainChildrenDOF(int dof) { mConstrChild &= ~(0x1 << dof); }
  void setChildrenConstrainPattern(uint32_t pat) { mConstrChild = pat; }
  bool hasChildrenConstraint() const { return mConstrChild; }
  //
  AlignableVolume* getParent() const { return mParent; }
  void setParent(AlignableVolume* par)
  {
    mParent = par;
    if (par)
      par->addChild(this);
  }
  int countParents() const;
  //
  int getNChildren() const { return mChildren ? mChildren->GetEntriesFast() : 0; }
  AlignableVolume* getChild(int i) const { return mChildren ? (AlignableVolume*)mChildren->UncheckedAt(i) : 0; }
  virtual void addChild(AlignableVolume* ch);
  //
  double getXTracking() const { return mX; }
  double getAlpTracking() const { return mAlp; }
  //
  int getNProcessedPoints() const { return mNProcPoints; }
  virtual int finalizeStat(DOFStatistics* h = 0);
  void fillDOFStat(DOFStatistics* h) const;
  //
  float* getParVals() const { return mParVals; }
  double getParVal(int par) const { return mParVals[par]; }
  double getParErr(int par) const { return mParErrs[par]; }
  int getParLab(int par) const { return mParLabs[par]; }
  void getParValGeom(double* delta) const
  {
    for (int i = kNDOFGeom; i--;)
      delta[i] = mParVals[i];
  }
  //
  void setParVals(int npar, double* vl, double* er);
  void setParVal(int par, double v = 0) { mParVals[par] = v; }
  void setParErr(int par, double e = 0) { mParErrs[par] = e; }
  //
  int getNDOFs() const { return mNDOFs; }
  int getNDOFFree() const { return mNDOFFree; }
  int getNDOFGeomFree() const { return mNDOFGeomFree; }
  int getFirstParGloID() const { return mFirstParGloID; }
  int getParGloID(int par) const { return mFirstParGloID + par; }
  void setFirstParGloID(int id) { mFirstParGloID = id; }
  //
  virtual void prepareMatrixT2L();
  virtual void setTrackingFrame();
  //
  const TGeoHMatrix& getMatrixL2G() const { return mMatL2G; }
  const TGeoHMatrix& getMatrixL2GIdeal() const { return mMatL2GIdeal; }
  const TGeoHMatrix& getMatrixL2GReco() const { return mMatL2GReco; }
  const TGeoHMatrix& getGlobalDeltaRef() const { return mMatDeltaRefGlo; }
  void setMatrixL2G(const TGeoHMatrix& m) { mMatL2G = m; }
  void setMatrixL2GIdeal(const TGeoHMatrix& m) { mMatL2GIdeal = m; }
  void setMatrixL2GReco(const TGeoHMatrix& m) { mMatL2GReco = m; }
  void setGlobalDeltaRef(TGeoHMatrix& mat) { mMatDeltaRefGlo = mat; }
  //
  virtual void prepareMatrixL2G(bool reco = false);
  virtual void prepareMatrixL2GIdeal();
  virtual void updateL2GRecoMatrices(const TClonesArray* algArr, const TGeoHMatrix* cumulDelta);
  //
  void getMatrixT2G(TGeoHMatrix& m) const;
  //
  const TGeoHMatrix& getMatrixT2L() const { return mMatT2L; }
  void setMatrixT2L(const TGeoHMatrix& m);
  //
  void delta2Matrix(TGeoHMatrix& deltaM, const double* delta) const;
  //
  // preparation of variation matrices
  void getDeltaT2LmodLOC(TGeoHMatrix& matMod, const double* delta) const;
  void getDeltaT2LmodTRA(TGeoHMatrix& matMod, const double* delta) const;
  void getDeltaT2LmodLOC(TGeoHMatrix& matMod, const double* delta, const TGeoHMatrix& relMat) const;
  void getDeltaT2LmodTRA(TGeoHMatrix& matMod, const double* delta, const TGeoHMatrix& relMat) const;
  //
  // creation of global matrices for storage
  void createGloDeltaMatrix(TGeoHMatrix& deltaM) const;
  void createLocDeltaMatrix(TGeoHMatrix& deltaM) const;
  void createPreGloDeltaMatrix(TGeoHMatrix& deltaM) const;
  void createPreLocDeltaMatrix(TGeoHMatrix& deltaM) const;
  void createAlignmenMatrix(TGeoHMatrix& alg) const;
  void createAlignmentObjects(TClonesArray* arr) const;
  //
  void setSkip(bool v = true) { SetBit(kSkipBit, v); }
  bool getSkip() const { return TestBit(kSkipBit); }
  //
  void excludeFromParentConstraint(bool v = true) { SetBit(kExclFromParentConstraintBit, v); }
  bool getExcludeFromParentConstraint() const { return TestBit(kExclFromParentConstraintBit); }
  //
  void setInitDOFsDone() { SetBit(kInitDOFsDoneBit); }
  bool getInitDOFsDone() const { return TestBit(kInitDOFsDoneBit); }
  //
  bool ownsDOFID(int id) const;
  AlignableVolume* getVolOfDOFID(int id) const;
  //
  virtual bool isSensor() const { return false; }
  //
  virtual const char* getDOFName(int i) const;
  virtual void Print(const Option_t* opt = "") const;
  virtual void writePedeInfo(FILE* parOut, const Option_t* opt = "") const;
  //
  static const char* getGeomDOFName(int i) { return i < kNDOFGeom ? sDOFName[i] : 0; }
  static void setDefGeomFree(uint8_t patt) { sDefGeomFree = patt; }
  static uint8_t getDefGeomFree() { return sDefGeomFree; }
  //
 protected:
  void setNDOFs(int n = kNDOFGeom);
  void calcFree(bool condFree = false);
  //
  // ------- dummies -------
  AlignableVolume(const AlignableVolume&);
  AlignableVolume& operator=(const AlignableVolume&);
  //
 protected:
  //
  Frame_t mVarFrame; // Variation frame for this volume
  int mIntID;        // internal id within the detector
  double mX;         // tracking frame X offset
  double mAlp;       // tracking frame alpa
  //
  char mNDOFs;          // number of degrees of freedom, including fixed ones
  uint32_t mDOF;        // bitpattern degrees of freedom
  char mNDOFGeomFree;   // number of free geom degrees of freedom
  char mNDOFFree;       // number of all free degrees of freedom
  uint8_t mConstrChild; // bitpattern for constraints on children corrections
  //
  AlignableVolume* mParent; // parent volume
  TObjArray* mChildren;     // array of childrens
  //
  int mNProcPoints;   // n of processed points
  int mFirstParGloID; // ID of the 1st parameter in the global results array
  float* mParVals;    //! values of the fitted params
  float* mParErrs;    //! errors of the fitted params
  int* mParLabs;      //! labels for parameters
  //
  TGeoHMatrix mMatL2GReco;     // local to global matrix used for reco of data being processed
  TGeoHMatrix mMatL2G;         // local to global matrix, including current alignment
  TGeoHMatrix mMatL2GIdeal;    // local to global matrix, ideal
  TGeoHMatrix mMatT2L;         // tracking to local matrix (ideal)
  TGeoHMatrix mMatDeltaRefGlo; // global reference delta from Align/Data
  //
  static const char* sDOFName[kNDOFGeom];
  static const char* sFrameName[kNVarFrames];
  static uint32_t sDefGeomFree;
  //
  ClassDef(AlignableVolume, 2)
};

//___________________________________________________________
inline void AlignableVolume::getMatrixT2G(TGeoHMatrix& m) const
{
  // compute tracking to global matrix, i.e. glo = T2G*tra = L2G*loc = L2G*T2L*tra
  m = getMatrixL2GIdeal();
  m *= getMatrixT2L();
}
} // namespace align
} // namespace o2
#endif
