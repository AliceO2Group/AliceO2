// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    Cave  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------

#ifndef ALICEO2_PASSIVE_Cave_H
#define ALICEO2_PASSIVE_Cave_H

#include "FairDetector.h" // for FairModule
#include "Rtypes.h"       // for ClassDef, etc
#include <functional>     // for std::function
#include <vector>
namespace o2
{
namespace passive
{

// This class represents the mother container
// holding all the detector (passive and active) modules

// The Cave is a FairDetector rather than a FairModule
// in order to be able to receive notifications about
// BeginPrimary/FinishPrimary/etc from the FairMCApplication and
// eventually dispatch to further O2 specific observers. This special role
// is justifiable since the Cave instance necessarily always exists.
class Cave : public FairDetector
{
 public:
  Cave(const char* name, const char* Title = "Exp Cave");
  Cave();
  ~Cave() override;
  void ConstructGeometry() override;
  void createMaterials();

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override;

  // the following methods are required for FairDetector but are not actually
  // implemented
  Bool_t ProcessHits(FairVolume* v = nullptr) override; // should never be actually called
  void Register() override {}
  TClonesArray* GetCollection(Int_t iColl) const override { return nullptr; }
  void Reset() override {}

  void FinishPrimary() override;
  void addFinishPrimaryHook(std::function<void()>&& hook) { mFinishPrimaryHooks.emplace_back(hook); }

  void includeZDC(bool hasZDC) { mHasZDC = hasZDC; }

  void BeginPrimary() override;

 private:
  Cave(const Cave& orig);
  Cave& operator=(const Cave&);

  std::vector<std::function<void()>> mFinishPrimaryHooks; //!

  bool mHasZDC = true; //! flag indicating if ZDC will be included

  ClassDefOverride(o2::passive::Cave, 1);
};
} // namespace passive
} // namespace o2
#endif //Cave_H
