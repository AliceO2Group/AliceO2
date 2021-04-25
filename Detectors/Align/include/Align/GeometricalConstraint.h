// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   GeometricalConstraint.h
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

#ifndef GEOMETRICALCONSTRAINT_H
#define GEOMETRICALCONSTRAINT_H

#include <cstdio>
#include <TNamed.h>
#include <TObjArray.h>
#include "Align/AlignableVolume.h"

namespace o2
{
namespace align
{

class GeometricalConstraint : public TNamed
{
 public:
  enum { kNDOFGeom = AlignableVolume::kNDOFGeom };
  enum { kNoJacobianBit = BIT(14) };
  //
  GeometricalConstraint(const char* name = nullptr, const char* title = nullptr);
  ~GeometricalConstraint() override;
  //
  void setParent(const AlignableVolume* par);
  const AlignableVolume* getParent() const { return mParent; }
  //
  int getNChildren() const { return mChildren.GetEntriesFast(); }
  AlignableVolume* getChild(int i) const { return (AlignableVolume*)mChildren[i]; }
  void addChild(const AlignableVolume* v)
  {
    if (v) {
      mChildren.AddLast((AlignableVolume*)v);
    }
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
  void Print(const Option_t* opt = "") const final;
  virtual void writeChildrenConstraints(FILE* conOut) const;
  virtual void checkConstraint() const;
  virtual const char* getDOFName(int i) const { return AlignableVolume::getGeomDOFName(i); }
  //
 protected:
  // ------- dummies -------
  GeometricalConstraint(const GeometricalConstraint&);
  GeometricalConstraint& operator=(const GeometricalConstraint&);
  //
 protected:
  uint32_t mConstraint;           // bit pattern of constraint
  double mSigma[kNDOFGeom];       // optional sigma if constraint is gaussian
  const AlignableVolume* mParent; // parent volume for contraint, lab if 0
  TObjArray mChildren;            // volumes subjected to constraints
  //
  ClassDef(GeometricalConstraint, 2);
};

} // namespace align
} // namespace o2
#endif
