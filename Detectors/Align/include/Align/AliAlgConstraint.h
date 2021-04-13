// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgConstraint.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Descriptor of geometrical constraint

/**
 *  Descriptor of geometrical constraint: the cumulative
 *  corrections of children for requested DOFs in the frame of
 *  parent (of LAB if parent is not defined) forced to be 0.
 *  The parent - child relationship need not to be real
 *
 *  Constraint wil be quazi-exact (Lagrange multiplier) if
 *  corresponding sigma = 0, or of gaussian type is sigma>0
 */

#ifndef ALIALGCONSTRAINT_H
#define ALIALGCONSTRAINT_H

#include <stdio.h>
#include <TNamed.h>
#include <TObjArray.h>
#include "Align/AliAlgVol.h"

namespace o2
{
namespace align
{

class AliAlgConstraint : public TNamed
{
 public:
  enum { kNDOFGeom = AliAlgVol::kNDOFGeom };
  enum { kNoJacobianBit = BIT(14) };
  //
  AliAlgConstraint(const char* name = 0, const char* title = 0);
  virtual ~AliAlgConstraint();
  //
  void setParent(const AliAlgVol* par);
  const AliAlgVol* getParent() const { return mParent; }
  //
  int getNChildren() const { return mChildren.GetEntriesFast(); }
  AliAlgVol* getChild(int i) const { return (AliAlgVol*)mChildren[i]; }
  void addChild(const AliAlgVol* v)
  {
    if (v)
      mChildren.AddLast((AliAlgVol*)v);
  }
  //
  bool isDOFConstrained(int dof) const { return mConstraint & 0x1 << dof; }
  uint8_t getConstraintPattern() const { return mConstraint; }
  void constrainDOF(int dof) { mConstraint |= 0x1 << dof; }
  void unConstrainDOF(int dof) { mConstraint &= ~(0x1 << dof); }
  void setConstrainPattern(uint32_t pat) { mConstraint = pat; }
  bool hasConstraint() const { return mConstraint; }
  double getSigma(int i) const { return mSigma[i]; }
  void setSigma(int i, double s = 0) { mSigma[i] = s; }
  //
  void setNoJacobian(bool v = true) { SetBit(kNoJacobianBit, v); }
  bool getNoJacobian() const { return TestBit(kNoJacobianBit); }
  //
  void constrCoefGeom(const TGeoHMatrix& matRD, float* jac /*[kNDOFGeom][kNDOFGeom]*/) const;
  //
  virtual void Print(const Option_t* opt = "") const;
  virtual void writeChildrenConstraints(FILE* conOut) const;
  virtual void checkConstraint() const;
  virtual const char* getDOFName(int i) const { return AliAlgVol::getGeomDOFName(i); }
  //
 protected:
  // ------- dummies -------
  AliAlgConstraint(const AliAlgConstraint&);
  AliAlgConstraint& operator=(const AliAlgConstraint&);
  //
 protected:
  uint32_t mConstraint;     // bit pattern of constraint
  double mSigma[kNDOFGeom]; // optional sigma if constraint is gaussian
  const AliAlgVol* mParent; // parent volume for contraint, lab if 0
  TObjArray mChildren;      // volumes subjected to constraints
  //
  ClassDef(AliAlgConstraint, 2);
};

} // namespace align
} // namespace o2
#endif
