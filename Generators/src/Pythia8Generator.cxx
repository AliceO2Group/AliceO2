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

#include <cmath>
#include "TROOT.h"
#include "Pythia8/Basics.h" // for RndmEngine
#include "Pythia8/Pythia.h"
#include "FairPrimaryGenerator.h"
#include "FairGenerator.h"
#include "TRandom.h"  // for TRandom
#include "TRandom1.h" // for TRandom1
#include "TRandom3.h" // for TRandom3, gRandom

#include "Generators/Pythia8Generator.h"

using namespace Pythia8;

namespace o2
{
namespace eventgen
{

class PyTr1Rng : public RndmEngine
{
 public:
  PyTr1Rng() { mRng = std::make_unique<TRandom1>(gRandom->GetSeed()); };
  ~PyTr1Rng() override = default;

  Double_t flat() override { return mRng->TRandom1::Rndm(); };

 private:
  std::unique_ptr<TRandom1> mRng; //!
};

class PyTr3Rng : public RndmEngine
{
 public:
  PyTr3Rng() { mRng = std::make_unique<TRandom3>(gRandom->GetSeed()); };
  ~PyTr3Rng() override = default;

  Double_t flat() override { return mRng->TRandom3::Rndm(); };

 private:
  std::unique_ptr<TRandom3> mRng; //!
};

// -----   Default constructor   -------------------------------------------
Pythia8Generator::Pythia8Generator()
{
  mUseRandom1 = kFALSE;
  mUseRandom3 = kTRUE;
  mId = 2212; // proton
  mMom = 400; // proton
  mHNL = 0;   // HNL  if set to !=0, for example 9900014, only track
  mPythia = std::make_unique<Pythia>();
}
// -------------------------------------------------------------------------

// -----   Default constructor   -------------------------------------------
Bool_t Pythia8Generator::Init()
{
  if (mUseRandom1) {
    mRandomEngine = std::make_unique<PyTr1Rng>();
  }
  if (mUseRandom3) {
    mRandomEngine = std::make_unique<PyTr3Rng>();
  }
  mPythia->setRndmEnginePtr(mRandomEngine.get());

  /** commenting these lines  
      as they would override external settings **/
  /**
  cout<<"Beam Momentum "<<fMom<<endl;
  // Set arguments in Settings database.
  mPythia.settings.mode("Beams:idA",  fId);
  mPythia.settings.mode("Beams:idB",  2212);
  mPythia.settings.mode("Beams:frameType",  3);
  mPythia.settings.parm("Beams:pxA",    0.);
  mPythia.settings.parm("Beams:pyA",    0.);
  mPythia.settings.parm("Beams:pzA",    fMom);
  mPythia.settings.parm("Beams:pxB",    0.);
  mPythia.settings.parm("Beams:pyB",    0.);
  mPythia.settings.parm("Beams:pzB",    0.);
  **/
  mPythia->init();
  return kTRUE;
}
// -------------------------------------------------------------------------

// -----   Destructor   ----------------------------------------------------
Pythia8Generator::~Pythia8Generator() = default;
// -------------------------------------------------------------------------

// -----   Passing the event   ---------------------------------------------
Bool_t Pythia8Generator::ReadEvent(FairPrimaryGenerator* cpg)
{
  const double mm2cm = 0.1;
  const double clight = 2.997924580e10; //cm/c
  Int_t npart = 0;
  while (npart == 0) {
    mPythia->next();
    for (int i = 0; i < mPythia->event.size(); i++) {
      if (mPythia->event[i].isFinal()) {
        // only send HNL decay products to G4
        if (mHNL != 0) {
          Int_t im = mPythia->event[i].mother1();
          if (mPythia->event[im].id() == mHNL) {
            // for the moment, hardcode 110m is maximum decay length
            Double_t z = mPythia->event[i].zProd();
            Double_t x = abs(mPythia->event[i].xProd());
            Double_t y = abs(mPythia->event[i].yProd());
            // cout<<"debug HNL decay pos "<<x<<" "<< y<<" "<< z <<endl;
            if (z < 11000. && z > 7000. && x < 250. && y < 250.) {
              npart++;
            }
          }
        } else {
          npart++;
        }
      };
    };
    // happens if a charm particle being produced which does decay without producing a HNL. Try another event.
    //       if (npart == 0){ mPythia->event.list();}
  };
  // cout<<"debug p8 event 0 " << mPythia->event[0].id()<< " "<< mPythia->event[1].id()<< " "
  // << mPythia->event[2].id()<< " "<< npart <<endl;
  for (Int_t ii = 0; ii < mPythia->event.size(); ii++) {
    if (mPythia->event[ii].isFinal()) {
      Bool_t wanttracking = true;
      if (mHNL != 0) {
        Int_t im = mPythia->event[ii].mother1();
        if (mPythia->event[im].id() != mHNL) {
          wanttracking = false;
        }
      }
      if (wanttracking) {
        Double_t z = mPythia->event[ii].zProd();
        Double_t x = mPythia->event[ii].xProd();
        Double_t y = mPythia->event[ii].yProd();
        Double_t pz = mPythia->event[ii].pz();
        Double_t px = mPythia->event[ii].px();
        Double_t py = mPythia->event[ii].py();
        Double_t t = mPythia->event[ii].tProd();
        x *= mm2cm;
        y *= mm2cm;
        z *= mm2cm;
        t *= mm2cm / clight;
        cpg->AddTrack((Int_t)mPythia->event[ii].id(), px, py, pz, x, y, z,
                      (Int_t)mPythia->event[ii].mother1(), wanttracking, -9e9, t);
        // cout<<"debug p8->geant4 "<< wanttracking << " "<< ii <<  " "
        // << mPythia->event[ii].id()<< " "<< mPythia->event[ii].mother1()<<" "<<x<<" "<< y<<" "<< z <<endl;
      }
    };
    if (mHNL != 0 && mPythia->event[ii].id() == mHNL) {
      Int_t im = (Int_t)mPythia->event[ii].mother1();
      Double_t z = mPythia->event[ii].zProd();
      Double_t x = mPythia->event[ii].xProd();
      Double_t y = mPythia->event[ii].yProd();
      Double_t pz = mPythia->event[ii].pz();
      Double_t px = mPythia->event[ii].px();
      Double_t py = mPythia->event[ii].py();
      Double_t t = mPythia->event[ii].tProd();
      x *= mm2cm;
      y *= mm2cm;
      z *= mm2cm;
      t *= mm2cm / clight;
      cpg->AddTrack((Int_t)mPythia->event[im].id(), px, py, pz, x, y, z, 0, false, -9e9, t);
      cpg->AddTrack((Int_t)mPythia->event[ii].id(), px, py, pz, x, y, z, im, false, -9e9, t);
      //cout<<"debug p8->geant4 "<< 0 << " "<< ii <<  " " << fake<< " "<< mPythia->event[ii].mother1()<<endl;
    };
  }

  // make separate container ??
  //    FairRootManager *ioman =FairRootManager::Instance();

  return kTRUE;
}
// -------------------------------------------------------------------------
void Pythia8Generator::SetParameters(const char* par)
{
  // Set Parameters
  mPythia->readString(par);
  cout << R"(mPythia->readString(")" << par << R"("))" << endl;
}

// -------------------------------------------------------------------------
void Pythia8Generator::Print()
{
  mPythia->settings.listAll();
}
// -------------------------------------------------------------------------
void Pythia8Generator::GetPythiaInstance(int arg)
{
  mPythia->particleData.list(arg);
  cout << "canDecay " << mPythia->particleData.canDecay(arg) << " " << mPythia->particleData.mayDecay(arg) << endl;
}
// -------------------------------------------------------------------------

} // namespace eventgen
} // namespace o2

ClassImp(o2::eventgen::Pythia8Generator);
