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
#include "Pythia8/Pythia.h"
#include "FairPrimaryGenerator.h"
//#include "FairGenerator.h"

#include "Generators/Pythia8Generator.h"

using namespace Pythia8;

// -----   Default constructor   -------------------------------------------
Pythia8Generator::Pythia8Generator()
{
  fUseRandom1 = kFALSE;
  fUseRandom3 = kTRUE;
  fId         = 2212; // proton
  fMom        = 400;  // proton
  fHNL        = 0;    // HNL  if set to !=0, for example 9900014, only track
}
// -------------------------------------------------------------------------

// -----   Default constructor   -------------------------------------------
Bool_t Pythia8Generator::Init()
{
  if (fUseRandom1) fRandomEngine = new PyTr1Rng();
  if (fUseRandom3) fRandomEngine = new PyTr3Rng();

  fPythia.setRndmEnginePtr(fRandomEngine);

  cout<<"Beam Momentum "<<fMom<<endl;
  // Set arguments in Settings database.
  fPythia.settings.mode("Beams:idA",  fId);
  fPythia.settings.mode("Beams:idB",  2212);
  fPythia.settings.mode("Beams:frameType",  3);
  fPythia.settings.parm("Beams:pxA",    0.);
  fPythia.settings.parm("Beams:pyA",    0.);
  fPythia.settings.parm("Beams:pzA",    fMom);
  fPythia.settings.parm("Beams:pxB",    0.);
  fPythia.settings.parm("Beams:pyB",    0.);
  fPythia.settings.parm("Beams:pzB",    0.);
  fPythia.init();
  return kTRUE;
}
// -------------------------------------------------------------------------


// -----   Destructor   ----------------------------------------------------
Pythia8Generator::~Pythia8Generator()
= default;
// -------------------------------------------------------------------------

// -----   Passing the event   ---------------------------------------------
Bool_t Pythia8Generator::ReadEvent(FairPrimaryGenerator* cpg)
{
  Int_t npart = 0;
  while(npart == 0)
    {
      fPythia.next();
      for(int i=0; i<fPythia.event.size(); i++)
	{
	  if(fPythia.event[i].isFinal())
	    {
// only send HNL decay products to G4
              if (fHNL != 0){
                Int_t im = fPythia.event[i].mother1();
                if (fPythia.event[im].id()==fHNL ){
// for the moment, hardcode 110m is maximum decay length
                 Double_t z = fPythia.event[i].zProd();
                 Double_t x = abs(fPythia.event[i].xProd());
                 Double_t y = abs(fPythia.event[i].yProd());
                 // cout<<"debug HNL decay pos "<<x<<" "<< y<<" "<< z <<endl;
                 if ( z < 11000. && z > 7000. && x<250. && y<250.) {
                   npart++;
                 }
               }
              }
	      else {npart++;}
	    };
	};
// happens if a charm particle being produced which does decay without producing a HNL. Try another event.
//       if (npart == 0){ fPythia.event.list();}
    };
// cout<<"debug p8 event 0 " << fPythia.event[0].id()<< " "<< fPythia.event[1].id()<< " "<< fPythia.event[2].id()<< " "<< npart <<endl;
  for(Int_t ii=0; ii<fPythia.event.size(); ii++){
    if(fPythia.event[ii].isFinal())
      {
        Bool_t wanttracking=true;
        if (fHNL != 0){
           Int_t im = fPythia.event[ii].mother1();
           if (fPythia.event[im].id() != fHNL) {wanttracking=false;}
        }
        if (  wanttracking ) {
          Double_t z  = fPythia.event[ii].zProd();
          Double_t x  = fPythia.event[ii].xProd();
          Double_t y  = fPythia.event[ii].yProd();
          Double_t pz = fPythia.event[ii].pz();
          Double_t px = fPythia.event[ii].px();
          Double_t py = fPythia.event[ii].py();
	  cpg->AddTrack((Int_t)fPythia.event[ii].id(),px,py,pz,x,y,z,
		      (Int_t)fPythia.event[ii].mother1(),wanttracking);
          // cout<<"debug p8->geant4 "<< wanttracking << " "<< ii <<  " " << fPythia.event[ii].id()<< " "<< fPythia.event[ii].mother1()<<" "<<x<<" "<< y<<" "<< z <<endl;
        }
//    virtual void AddTrack(Int_t pdgid, Double_t px, Double_t py, Double_t pz,
//                          Double_t vx, Double_t vy, Double_t vz, Int_t parent=-1,Bool_t wanttracking=true,Double_t e=-9e9);
    };
    if (fHNL != 0 && fPythia.event[ii].id() == fHNL){
         Int_t im = (Int_t)fPythia.event[ii].mother1();
         Double_t z  = fPythia.event[ii].zProd();
         Double_t x  = fPythia.event[ii].xProd();
         Double_t y  = fPythia.event[ii].yProd();
         Double_t pz = fPythia.event[ii].pz();
         Double_t px = fPythia.event[ii].px();
         Double_t py = fPythia.event[ii].py();
	 cpg->AddTrack((Int_t)fPythia.event[im].id(),px,py,pz,x,y,z,0,false);
	 cpg->AddTrack((Int_t)fPythia.event[ii].id(),px,py,pz,x,y,z, im,false);
         //cout<<"debug p8->geant4 "<< 0 << " "<< ii <<  " " << fake<< " "<< fPythia.event[ii].mother1()<<endl;
      };
  }

// make separate container ??
  //    FairRootManager *ioman =FairRootManager::Instance();


  return kTRUE;
}
// -------------------------------------------------------------------------
void Pythia8Generator::SetParameters(char* par)
{
  // Set Parameters
    fPythia.readString(par);
    cout<<R"(fPythia.readString(")"<<par<<R"("))"<<endl;
}

// -------------------------------------------------------------------------
void Pythia8Generator::Print(){
  fPythia.settings.listAll();
}
// -------------------------------------------------------------------------
void Pythia8Generator::GetPythiaInstance(int arg){
  fPythia.particleData.list(arg) ;
  cout<<"canDecay "<<fPythia.particleData.canDecay(arg)<<" "<<fPythia.particleData.mayDecay(arg)<<endl ;
}
// -------------------------------------------------------------------------

ClassImp(Pythia8Generator)
