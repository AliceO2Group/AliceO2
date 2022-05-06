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

/*
 * O2MCApplicationEvalMat.h
 *
 *  Created on: March 17, 2022
 *      Author: amorsch
 */

#ifndef STEER_INCLUDE_STEER_O2MCAPPLICATIONEVALMAT_H_
#define STEER_INCLUDE_STEER_O2MCAPPLICATIONEVALMAT_H_

#include <Steer/O2MCApplicationBase.h>
#include <Steer/MaterialBudgetMap.h>
#include "Rtypes.h" // for Int_t, Bool_t, Double_t, etc
#include <TVirtualMC.h>
#include "SimConfig/SimParams.h"

namespace o2
{
namespace steer
{

// O2 specific changes/overrides to FairMCApplication
// Here in particular for custom adjustments to stepping logic
// and tracking limits
class O2MCApplicationEvalMat : public O2MCApplicationBase
{
 public:
  O2MCApplicationEvalMat() : O2MCApplicationBase() {}
  O2MCApplicationEvalMat(const char* name, const char* title, TObjArray* ModList, const char* MatName) : O2MCApplicationBase(name, title, ModList, MatName), mMaterialBudgetMap(nullptr), mPhi(0), mMode(-1) {}

  ~O2MCApplicationEvalMat() override = default;
  void FinishPrimary() override;
  void Stepping() override;
  void BeginEvent() override;
  void FinishEvent() override;

 protected:
  MaterialBudgetMap* mMaterialBudgetMap;
  Int_t mMode;    // 0: theta-phi, 1: eta-phi, 2: z-phi
  Float_t mC1[3]; // current coordinate 1
  Float_t mPhi;   // current phi

  ClassDefOverride(O2MCApplicationEvalMat, 1);
};

} // end namespace steer
} // end namespace o2

#endif /* STEER_INCLUDE_STEER_O2MCAPPLICATIONEVALMAT_H_ */
