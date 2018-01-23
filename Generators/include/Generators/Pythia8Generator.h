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

// Avoid the inclusion of dlfcn.h by Pyhtia.h that CINT is not able to process (avoid compile error on GCC > 5)
#ifdef __CLING__
#define _DLFCN_H_
#define _DLFCN_H
#endif

#include "Pythia8/Basics.h"          // for RndmEngine
#include "FairGenerator.h"   // for FairGenerator
#include "Pythia8/Pythia.h"  // for Pythia
#include "Rtypes.h"          // for Double_t, Bool_t, Int_t, etc
#include "TRandom.h"         // for TRandom
#include "TRandom1.h"        // for TRandom1
#include "TRandom3.h"        // for TRandom3, gRandom
class FairPrimaryGenerator;  // lines 22-22

class FairPrimaryGenerator;
using namespace Pythia8;
namespace o2
{
namespace eventgen
{

class PyTr1Rng : public RndmEngine
{
 public:
  PyTr1Rng() {  rng = new TRandom1(gRandom->GetSeed()); };
  ~PyTr1Rng() override = default;

  Double_t flat() override { return rng->Rndm(); };

 private:
  TRandom1 *rng; //!
};

class PyTr3Rng : public RndmEngine
{
 public:
  PyTr3Rng() {  rng = new TRandom3(gRandom->GetSeed()); };
  ~PyTr3Rng() override = default;

  Double_t flat() override { return rng->Rndm(); };

 private:
  TRandom3 *rng; //!
};




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
  void SetId(Double_t id) { mId  = id; };
  void SetHNLId(Int_t id) { mHNL = id; };
  void UseRandom1() { mUseRandom1 = kTRUE; mUseRandom3 = kFALSE; };
  void UseRandom3() { mUseRandom1 = kFALSE; mUseRandom3 = kTRUE; };
  void GetPythiaInstance(int);

 private:

  Pythia mPythia;             //!
  RndmEngine* mRandomEngine;  //!

 protected:

  Double_t mMom;       // proton momentum
  Int_t    mHNL;       // HNL ID
  Int_t    mId;       // target type
  Bool_t mUseRandom1;  // flag to use TRandom1
  Bool_t mUseRandom3;  // flag to use TRandom3 (default)

  ClassDefOverride(Pythia8Generator,1);
};

} // namespace eventgen
} // namespace o2
#endif /* !PNDP8GENERATOR_H */
