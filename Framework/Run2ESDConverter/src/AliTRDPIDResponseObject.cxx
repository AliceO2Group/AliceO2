/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/
//
// Container class for the reference distributions for TRD PID
// The class contains the reference distributions and the momentum steps
// the references are taken at. Mapping is done inside. To derive references,
// the functions GetUpperReference and GetLowerReference return the next
// reference distribution object and the momentum step above respectively below
// the tracklet momentum.
//
// Authors:
//    Markus Fasel <M.Fasel@gsi.de>
//    Daniel Lohner <Daniel.Lohner@cern.ch>

#include "AliLog.h"

#include "AliTRDPIDResponseObject.h"

#ifndef AliTRDPIDREFERENCE_H
#include "AliTRDPIDReference.h"
#endif

#ifndef AliTRDPIDPARAMS_H
#include "AliTRDPIDParams.h"
#endif


ClassImp(AliTRDPIDResponseObject)

//____________________________________________________________
AliTRDPIDResponseObject::AliTRDPIDResponseObject():
    TNamed(),
    fNSlicesQ0(4)
{
    //
    // Dummy constructor
    //
    SetBit(kIsOwner, kTRUE);

    for(Int_t method=0;method<AliTRDPIDResponse::kNMethod;method++){
	fPIDParams[method]=NULL;
	fPIDReference[method]=NULL;
    }
}

//____________________________________________________________
AliTRDPIDResponseObject::AliTRDPIDResponseObject(const Char_t *name):
TNamed(name, "TRD PID Response Object"),
fNSlicesQ0(4)
{
	//
	// Default constructor
	//
	SetBit(kIsOwner, kTRUE);

	for(Int_t method=0;method<AliTRDPIDResponse::kNMethod;method++){
	    fPIDParams[method]=NULL;
	    fPIDReference[method]=NULL;
	}
}

//____________________________________________________________
AliTRDPIDResponseObject::AliTRDPIDResponseObject(const AliTRDPIDResponseObject &ref):
TNamed(ref),
fNSlicesQ0(ref.fNSlicesQ0)
{
    //
    // Copy constructor
    // Only copies pointers, object is not the owner of the references
    //
    SetBit(kIsOwner, kFALSE);

    for(Int_t method=0;method<AliTRDPIDResponse::kNMethod;method++){
	fPIDParams[method]=ref.fPIDParams[method];       // new Object is not owner, copy only pointer
	fPIDReference[method]=ref.fPIDReference[method];    // new Object is not owner, copy only pointer
    }
}
//____________________________________________________________
AliTRDPIDResponseObject &AliTRDPIDResponseObject::operator=(const AliTRDPIDResponseObject &ref){
	//
	// Assginment operator
	// Only copies poiters, object is not the owner of the references
	//
	if(this != &ref){
	    TNamed::operator=(ref);
	    fNSlicesQ0=ref.fNSlicesQ0;
	    for(Int_t method=0;method<AliTRDPIDResponse::kNMethod;method++){
	      if(TestBit(kIsOwner) && fPIDParams[method]){
		delete fPIDParams[method];
		fPIDParams[method]= 0;
	      }
	      if(TestBit(kIsOwner) && fPIDReference[method]){
		delete fPIDReference[method];
		fPIDReference[method] = 0;
	      }
	      printf("Assignment");
	      fPIDParams[method]=ref.fPIDParams[method];       // new Object is not owner, copy only pointer
	      fPIDReference[method]=ref.fPIDReference[method];    // new Object is not owner, copy only pointer
	    }
	    SetBit(kIsOwner, kFALSE);
	}
	return *this;
}

//____________________________________________________________
AliTRDPIDResponseObject::~AliTRDPIDResponseObject(){
	//
	// Destructor
	// references are deleted if the object is the owner
	//
    for(Int_t method=0;method<AliTRDPIDResponse::kNMethod;method++){
	if(fPIDParams[method] && TestBit(kIsOwner)){
	delete fPIDParams[method];fPIDParams[method] = 0;
      }
      if(fPIDReference[method] && TestBit(kIsOwner)){
	delete fPIDReference[method];
	fPIDReference[method] = 0;
      }
    }
}

//____________________________________________________________
void AliTRDPIDResponseObject::SetPIDParams(AliTRDPIDParams *params,AliTRDPIDResponse::ETRDPIDMethod method){

    printf("in trd pid response %i \n",method);

    if(Int_t(method)>=Int_t(AliTRDPIDResponse::kNMethod)||Int_t(method)<0){
	AliError("Method does not exist");
	return;
    }
    if(fPIDParams[method]){
	delete fPIDParams[method];
        fPIDParams[method]=NULL;
    }

    fPIDParams[method]=new AliTRDPIDParams(*params);
}

//____________________________________________________________
void AliTRDPIDResponseObject::SetPIDReference(AliTRDPIDReference *reference,AliTRDPIDResponse::ETRDPIDMethod method, Int_t NofCharges){

    if(Int_t(method)>=Int_t(AliTRDPIDResponse::kNMethod)||Int_t(method)<0){
        AliError("Method does not exist");
	return;
    }
    if(fPIDReference[method]){
	delete fPIDReference[method];
	fPIDReference[method]=NULL;
    }
    fPIDReference[method]=new AliTRDPIDReference(*reference, NofCharges);
}

//____________________________________________________________
TObject *AliTRDPIDResponseObject::GetUpperReference(AliPID::EParticleType spec, Float_t p, Float_t &pUpper,AliTRDPIDResponse::ETRDPIDMethod method, Int_t Charge) const{

    if(Int_t(method)>=Int_t(AliTRDPIDResponse::kNMethod)||Int_t(method)<0){
	AliError("Method does not exist");
	return NULL;
    }
   
    if(fPIDReference[method]){
    return fPIDReference[method]->GetUpperReference(spec,p,pUpper,Charge);
    }
    return NULL;
}


//____________________________________________________________
TObject *AliTRDPIDResponseObject::GetLowerReference(AliPID::EParticleType spec, Float_t p, Float_t &pLower,AliTRDPIDResponse::ETRDPIDMethod method, Int_t Charge) const{

    if(Int_t(method)>=Int_t(AliTRDPIDResponse::kNMethod)||Int_t(method)<0){
	AliError("Method does not exist");
	return NULL;
    }

    if(fPIDReference[method]){
     return fPIDReference[method]->GetLowerReference(spec,p,pLower,Charge);
     }
    return NULL;
}

//____________________________________________________________
Bool_t AliTRDPIDResponseObject::GetThresholdParameters(Int_t ntracklets, Double_t efficiency, Double_t *params,Double_t centrality,AliTRDPIDResponse::ETRDPIDMethod method, Int_t charge) const{

    if(Int_t(method)>=Int_t(AliTRDPIDResponse::kNMethod)||Int_t(method)<0){
	AliError("Method does not exist");
	return kFALSE;
    }

    if(fPIDParams[method]){
    return fPIDParams[method]->GetThresholdParameters(ntracklets,efficiency,params,centrality, charge);
    }
    AliError("TRD Threshold Container does not exist");
    return kFALSE;
}

//____________________________________________________________
Int_t AliTRDPIDResponseObject::GetNumberOfMomentumBins(AliTRDPIDResponse::ETRDPIDMethod method) const{

    if(Int_t(method)>=Int_t(AliTRDPIDResponse::kNMethod)||Int_t(method)<0){
	AliError("Method does not exist");
	return 0;
    }

    if(fPIDReference[method]){
	return fPIDReference[method]->GetNumberOfMomentumBins();
    }
    return 0;
}

//____________________________________________________________
void AliTRDPIDResponseObject::Print(const Option_t* opt) const{
	//
	// Print content of the PID object
	//
    printf("Content of AliTRDPIDResponseObject \n\n");
   
    for(Int_t method=0;method<AliTRDPIDResponse::kNMethod;method++){
	if(fPIDReference[method])fPIDReference[method]->Print(opt);
	if(fPIDParams[method])fPIDParams[method]->Print(opt);
    }
}
