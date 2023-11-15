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

/// @file   DOFSet.h
/// @author ruben.shahoyan@cern.ch
/// @brief  Interface to contiguous set of DOFs in the controller class

#ifndef ALG_DOFSET_H
#define ALG_DOFSET_H

#include <TNamed.h>
#include "Framework/Logger.h"

namespace o2
{
namespace align
{
class Controller;

class DOFSet : public TNamed
{
 public:
  DOFSet() = default;
  DOFSet(const char* symname, Controller* ctr);
  ~DOFSet() override = default;

  const float* getParVals() const;
  const float* getParErrs() const;
  const int* getParLabs() const;

  float getParVal(int par) const { return getParVals()[par]; }
  float getParErr(int par) const { return getParErrs()[par]; }
  int getParLab(int par) const { return getParLabs()[par]; }
  void getParValGeom(double* delta) const;
  //
  int getNDOFs() const { return mNDOFs; }
  int getNDOFsFree() const { return mNDOFsFree; }
  int getNCalibDOFs() const { return mNCalibDOFs; }
  int getNCalibDOFsFree() const { return mNCalibDOFsFree; }
  int getFirstParGloID() const { return mFirstParGloID; }
  int getParGloID(int par) const { return mFirstParGloID + par; }

  void setNDOFs(int n) { mNDOFs = n; }
  void setNDOFsFree(int n) { mNDOFsFree = n; }
  void setNCalibDOFs(int n) { mNCalibDOFs = n; }
  void setNCalibDOFsFree(int n) { mNCalibDOFsFree = n; }
  void setFirstParGloID(int id) { mFirstParGloID = id; }
  //
  void setParVals(int npar, double* vl, double* er);
  void setParVal(int par, double v = 0) { getParVals()[par] = v; }
  void setParErr(int par, double e = 0) { getParErrs()[par] = e; }
  void setParLab(int par, int lab)
  {
    getParLabs()[par] = lab;
    LOGP(debug, "Assign label {} to DOF{}/{} of {}", lab, par, mNDOFs, GetName());
  }

 protected:
  auto getController() { return mController; }
  float* getParVals();
  float* getParErrs();
  int* getParLabs();
  bool varsSet() const { return mFirstParGloID != -1; }

  Controller* mController = nullptr;
  int mNDOFs = 0;          // number of DOFs
  int mNDOFsFree = 0;      // numer of DOFs free
  int mNCalibDOFs = 0;     // number of calibDOFs
  int mNCalibDOFsFree = 0; // number of calibDOFs free
  int mFirstParGloID = -1; // ID of the 1st parameter in the global results array

  ClassDefOverride(DOFSet, 1);
};

} // namespace align
} // namespace o2

#endif
