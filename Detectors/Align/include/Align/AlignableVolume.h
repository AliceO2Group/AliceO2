// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "Align/DOFSet.h"
#include <vector>

class TObjArray;
class TH1;

namespace o2
{
namespace align
{

class Controller;

class AlignableVolume : public DOFSet
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
  AlignableVolume() = default;
  AlignableVolume(const char* symname, int iid, Controller* ctr);
  ~AlignableVolume() override;
  //
  const char* getSymName() const { return GetName(); }
  //
  int getVolID() const { return (int)GetUniqueID(); }
  void setVolID(int v) { SetUniqueID(v); }
  int getInternalID() const { return mIntID; }
  void setInternalID(int v) { mIntID = v; }
  //
  //
  void assignDOFs();
  void initDOFs();
  //
  void getParValGeom(double* delta) const;

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

  void setMeasuredDOFPattern(uint32_t pat)
  {
    mDOFAsMeas = pat;
  }
  bool isNameMatching(const std::string& regexStr) const;
  bool isFreeDOF(int dof) const { return (mDOF & (0x1 << dof)) != 0; }
  bool isMeasuredDOF(int dof) const { return isFreeDOF(dof) && ((mDOFAsMeas & (0x1 << dof)) != 0); }
  bool isCondDOF(int dof) const;
  uint32_t getFreeDOFPattern() const { return mDOF; }
  uint32_t getFreeDOFGeomPattern() const { return mDOF & kAllGeomDOF; }
  //
  void addAutoConstraints();
  bool isChildrenDOFConstrained(int dof) const { return mConstrChild & 0x1 << dof; }
  uint8_t getChildrenConstraintPattern() const { return mConstrChild; }
  void constrainChildrenDOF(int dof) { mConstrChild |= 0x1 << dof; }
  void unConstrainChildrenDOF(int dof) { mConstrChild &= ~(0x1 << dof); }
  void setChildrenConstrainPattern(uint32_t pat) { mConstrChild = pat; }
  bool hasChildrenConstraint() const { return mConstrChild; }
  //
  AlignableVolume* getParent() const { return mParent; }
  void setParent(AlignableVolume* par)
  {
    mParent = par;
    if (par) {
      par->addChild(this);
    }
  }
  int countParents() const;
  //
  int getNChildren() const { return mChildren ? mChildren->GetEntriesFast() : 0; }
  AlignableVolume* getChild(int i) const { return mChildren ? (AlignableVolume*)mChildren->UncheckedAt(i) : nullptr; }
  virtual void addChild(AlignableVolume* ch);
  //
  double getXTracking() const { return mX; }
  double getAlpTracking() const { return mAlp; }
  //
  int getNProcessedPoints() const { return mNProcPoints; }
  virtual int finalizeStat();
  //
  int getNDOFGeomFree() const { return mNDOFGeomFree; }
  //
  virtual void prepareMatrixT2L();
  //
  const TGeoHMatrix& getMatrixL2G() const { return mMatL2G; }
  const TGeoHMatrix& getMatrixL2GIdeal() const { return mMatL2GIdeal; }
  const TGeoHMatrix& getMatrixL2GReco() const { return mMatL2GReco; }
  const TGeoHMatrix& getGlobalDeltaRef() const { return mMatDeltaRefGlo; }
  const TGeoHMatrix& getMatrixT2L() const { return mMatT2L; }

  void setMatrixL2G(const TGeoHMatrix& m) { mMatL2G = m; }
  void setMatrixL2GIdeal(const TGeoHMatrix& m) { mMatL2GIdeal = m; }
  void setMatrixL2GReco(const TGeoHMatrix& m) { mMatL2GReco = m; }
  void setGlobalDeltaRef(const TGeoHMatrix& mat) { mMatDeltaRefGlo = mat; }
  void setMatrixT2L(const TGeoHMatrix& m) { mMatT2L = m; }

  //
  virtual void prepareMatrixL2G(bool reco = false);
  virtual void prepareMatrixL2GIdeal();
  virtual void updateL2GRecoMatrices(const std::vector<o2::detectors::AlignParam>& algArr, const TGeoHMatrix* cumulDelta);
  //
  void getMatrixT2G(TGeoHMatrix& m) const;
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
  bool createGloDeltaMatrix(TGeoHMatrix& deltaM) const;
  bool createLocDeltaMatrix(TGeoHMatrix& deltaM) const; // return true if the matrix is not unity
  void createPreGloDeltaMatrix(TGeoHMatrix& deltaM) const;
  void createPreLocDeltaMatrix(TGeoHMatrix& deltaM) const;
  void createAlignmenMatrix(TGeoHMatrix& alg) const;
  void createAlignmentObjects(std::vector<o2::detectors::AlignParam>& arr) const;
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
  bool isDummy() const { return mIsDummy; }
  void setDummy(bool v) { mIsDummy = v; }
  //
  virtual bool isSensor() const { return false; }
  //
  virtual const char* getDOFName(int i) const;
  void Print(const Option_t* opt = "") const override;
  virtual void writePedeInfo(FILE* parOut, const Option_t* opt = "") const;
  virtual void writeLabeledPedeResults(FILE* parOut) const;
  //
  static const char* getGeomDOFName(int i) { return i < kNDOFGeom ? sDOFName[i] : nullptr; }
  static void setDefGeomFree(uint8_t patt) { sDefGeomFree = patt; }
  static uint8_t getDefGeomFree() { return sDefGeomFree; }
  //
 protected:
  void calcFree(bool condFree = true);
  //
  // ------- dummies -------
  AlignableVolume(const AlignableVolume&);
  AlignableVolume& operator=(const AlignableVolume&);
  //
 protected:
  //
  Frame_t mVarFrame = kTRA; // Variation frame for this volume
  int mIntID = -1;          // internal id within the detector
  double mX = 0.;           // tracking frame X offset
  double mAlp = 0.;         // tracking frame alpa
  //
  uint32_t mDOF = 0;        // pattern of DOFs
  uint32_t mDOFAsMeas = 0;  // consider DOF as measured with presigma error
  bool mIsDummy = false;    // placeholder (e.g. inactive), used to have the numbering corresponding to position in the container
  char mNDOFGeomFree = 0;   // number of free geom degrees of freedom
  uint8_t mConstrChild = 0; // bitpattern for constraints on children corrections
  //
  AlignableVolume* mParent = nullptr; // parent volume
  TObjArray* mChildren = nullptr;     // array of childrens
  //
  int mNProcPoints = 0; // n of processed points

  TGeoHMatrix mMatL2GReco{};     // local to global matrix used for reco of data being processed
  TGeoHMatrix mMatL2G{};         // local to global matrix, including current alignment
  TGeoHMatrix mMatL2GIdeal{};    // local to global matrix, ideal
  TGeoHMatrix mMatT2L{};         // tracking to local matrix (ideal)
  TGeoHMatrix mMatDeltaRefGlo{}; // global reference delta from Align/Data
  //
  static const char* sDOFName[kNDOFGeom];
  static const char* sFrameName[kNVarFrames];
  static uint32_t sDefGeomFree;
  //
  ClassDefOverride(AlignableVolume, 2);
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
