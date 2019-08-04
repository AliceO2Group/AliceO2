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
// -----                  M. Al-Turany   June 2014                     -----
// -------------------------------------------------------------------------

#ifndef PNDP8GENERATOR_H
#define PNDP8GENERATOR_H 1

#include "FairGenerator.h" // for FairGenerator
#include "Rtypes.h"        // for Double_t, Bool_t, Int_t, etc
#include <memory>

class FairPrimaryGenerator;
namespace Pythia8
{
class Pythia;
class RndmEngine;
} // namespace Pythia8
namespace o2
{
namespace eventgen
{

class Pythia8Generator : public FairGenerator
{
 public:
  /** default constructor **/
  Pythia8Generator();

  /** destructor **/
  ~Pythia8Generator() override;

  /** public method ReadEvent **/
  Bool_t ReadEvent(FairPrimaryGenerator*) override;
  void SetParameters(const char*);
  void Print(); //!

  Bool_t Init() override; //!

  void SetMom(Double_t mom) { mMom = mom; };
  void SetId(Double_t id) { mId = id; };
  void SetHNLId(Int_t id) { mHNL = id; };
  void UseRandom1()
  {
    mUseRandom1 = kTRUE;
    mUseRandom3 = kFALSE;
  };
  void UseRandom3()
  {
    mUseRandom1 = kFALSE;
    mUseRandom3 = kTRUE;
  };
  void GetPythiaInstance(int);

 private:
  std::unique_ptr<Pythia8::Pythia> mPythia;           //!
  std::unique_ptr<Pythia8::RndmEngine> mRandomEngine; //!

 protected:
  Double_t mMom;      // proton momentum
  Int_t mHNL;         // HNL ID
  Int_t mId;          // target type
  Bool_t mUseRandom1; // flag to use TRandom1
  Bool_t mUseRandom3; // flag to use TRandom3 (default)

  ClassDefOverride(Pythia8Generator, 1);
};

} // namespace eventgen
} // namespace o2
#endif /* !PNDP8GENERATOR_H */
