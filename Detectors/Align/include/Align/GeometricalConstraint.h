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
#include <Rtypes.h>
#include "Align/AlignableVolume.h"

namespace o2
{
namespace align
{

class GeometricalConstraint
{
 public:
  enum { kNDOFGeom = AlignableVolume::kNDOFGeom };
  //
  const std::string getName() const { return mName; }
  void setName(const std::string& n) { mName = n; }
  void setParent(const AlignableVolume* par)
  {
    mParent = par;
    if (mName.empty()) {
      mName = par->getSymName();
    }
  }
  const AlignableVolume* getParent() const { return mParent; }
  //
  int getNChildren() const { return mChildren.size(); }
  const AlignableVolume* getChild(int i) const { return mChildren[i]; }
  void addChild(const AlignableVolume* v)
  {
    if (v) {
      mChildren.push_back(v);
    }
  }
  //
  bool isDOFConstrained(int dof) const { return mConstraint & 0x1 << dof; }
  uint8_t getConstraintPattern() const { return mConstraint; }
  void constrainDOF(int dof) { mConstraint |= 0x1 << dof; }
  void unConstrainDOF(int dof) { mConstraint &= ~(0x1 << dof); }
  void setConstrainPattern(uint8_t pat) { mConstraint = pat; }
  bool hasConstraint() const { return mConstraint; }
  double getSigma(int i) const { return mSigma[i]; }
  void setSigma(int i, double s = 0) { mSigma[i] = s; }
  //
  void setNoJacobian(bool v = true) { mNoJacobian = v; }
  bool getNoJacobian() const { return mNoJacobian; }
  //
  void constrCoefGeom(const TGeoHMatrix& matRD, double* jac /*[kNDOFGeom][kNDOFGeom]*/) const;
  //
  void print() const;
  void writeChildrenConstraints(FILE* conOut) const;
  void checkConstraint() const;
  const char* getDOFName(int i) const { return AlignableVolume::getGeomDOFName(i); }

 protected:
  bool mNoJacobian = false;                      // flag that Jacobian is not needed
  uint8_t mConstraint = 0;                       // bit pattern of constraint
  double mSigma[kNDOFGeom] = {};                 // optional sigma if constraint is gaussian
  const AlignableVolume* mParent = nullptr;      // parent volume for contraint, lab if 0
  std::vector<const AlignableVolume*> mChildren; // volumes subjected to constraints
  std::string mName{};
  //
  ClassDefNV(GeometricalConstraint, 2);
};

} // namespace align
} // namespace o2
#endif
