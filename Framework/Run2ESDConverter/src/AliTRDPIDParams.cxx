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
// Container for TRD Threshold parameters stored in the OADB
//
// Author: Markus Fasel <M.Fasel@gsi.de>
//
#include <TList.h>
#include <TMath.h>
#include <TSortedList.h>

#include "AliLog.h"

#include "AliTRDPIDParams.h"


ClassImp(AliTRDPIDParams)
//ClassImp(AliTRDPIDParams::AliTRDPIDThresholds)
//ClassImp(AliTRDPIDParams::AliTRDPIDCentrality)

const Double_t AliTRDPIDParams::kVerySmall = 1e-5;

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDParams():
    TNamed(),
    fEntries(NULL)
{
    //
    // Dummy constructor
    //
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDParams(const char *name) :
    TNamed(name, ""),
    fEntries(NULL)
{
    //
    // Default constructor
    //
    fEntries = new TList;
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDParams(const AliTRDPIDParams &ref):
    TNamed(ref),
    fEntries(NULL)
{
    //
    // Copy constructor
    //

    fEntries=(TList*)ref.fEntries->Clone();
}

//____________________________________________________________
AliTRDPIDParams::~AliTRDPIDParams(){
    //
    // Destructor
    //
    delete fEntries;
}

//____________________________________________________________
void AliTRDPIDParams::AddCentralityClass(Double_t minCentrality, Double_t maxCentrality){
    //
    // Add new centrality class
    //

    // check whether centrality class already exists
    AliTRDPIDCentrality *checklow = FindCentrality(minCentrality + 0.01),
            *checkhigh = FindCentrality(maxCentrality - 0.01);

    if(!checklow && ! checkhigh)
        fEntries->Add(new AliTRDPIDCentrality(minCentrality, maxCentrality));
}

//____________________________________________________________ 
AliTRDPIDParams::AliTRDPIDCentrality *AliTRDPIDParams::FindCentrality(Double_t val) const {
    //
    // Find centrality bin
    //
    TIter centralities(fEntries);
    AliTRDPIDCentrality *obj(NULL), *tmp(NULL);
    while((obj = dynamic_cast<AliTRDPIDCentrality *>(centralities()))){
        if(val >= obj->GetMinCentrality() && val <= obj->GetMaxCentrality()){
            tmp = obj;
            break;
        }
    }
    return tmp;
}

//____________________________________________________________
Bool_t AliTRDPIDParams::GetThresholdParameters(Int_t ntracklets, Double_t efficiency, Double_t *params, Double_t centrality, Int_t charge) const{
    //
    // Retrieve params
    // Use IsEqual definition
    //
    AliTRDPIDCentrality *cent = FindCentrality(centrality);
    if(!cent)cent = FindCentrality(-1);// try default class
    if(!cent){
        AliDebug(1, "Centrality class not available");
        return kFALSE;
    }

    return cent->GetThresholdParameters(ntracklets, efficiency, params, charge);
}

//____________________________________________________________
void AliTRDPIDParams::SetThresholdParameters(Int_t ntracklets, Double_t effMin, Double_t effMax, Double_t *params, Double_t centrality, Int_t charge){
    //
    // Set new threshold parameters
    //
    AliTRDPIDCentrality *cent = FindCentrality(centrality);
    if(cent) cent->SetThresholdParameters(ntracklets, effMin, effMax, params, charge);
    else AliDebug(1, "Centrality class not available");
}

//____________________________________________________________
void AliTRDPIDParams::Print(Option_t *) const {
    TIter centIter(fEntries);
    AliTRDPIDCentrality *cent;
    while((cent = dynamic_cast<AliTRDPIDCentrality *>(centIter()))) cent->Print(NULL);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDThresholds::AliTRDPIDThresholds():
    TObject(),
    fNTracklets(0),
    fCharge(0)
{
    //
    // Default constructor
    //
    memset(fParams, 0, sizeof(Double_t) * 4);
    memset(fEfficiency, 0, sizeof(Double_t) * 2);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDThresholds::AliTRDPIDThresholds(Int_t nTracklets, Double_t effMin, Double_t effMax, Double_t *params, Int_t charge) :
    TObject(),
    fNTracklets(nTracklets),
    fCharge(charge)
{
    //
    // Contructor to store params
    //
    fEfficiency[0] = effMin;
    fEfficiency[1] = effMax;
    if(params) memcpy(fParams, params, sizeof(Double_t) * 4);
    else memset(fParams, 0, sizeof(Double_t) * 4);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDThresholds::AliTRDPIDThresholds(Int_t nTracklets, Double_t eff, Double_t *params, Int_t charge) :
    TObject(),
    fNTracklets(nTracklets),
    fCharge(charge)
{
    //
    // Constructor used to find object in sorted list
    //
    fEfficiency[0] = fEfficiency[1] = eff;
    if(params) memcpy(fParams, params, sizeof(Double_t) * 4);
    else memset(fParams, 0, sizeof(Double_t) * 4);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDThresholds::AliTRDPIDThresholds(Int_t nTracklets, Double_t eff, Int_t charge) :
    TObject(),
    fNTracklets(nTracklets),
    fCharge(charge)
{
    //
    // Constructor used to find object in sorted list
    //
    fEfficiency[0] = fEfficiency[1] = eff;
    memset(fParams, 0, sizeof(Double_t) * 4);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDThresholds::AliTRDPIDThresholds(Int_t nTracklets, Double_t effMin, Double_t effMax, Double_t *params) :
    TObject(),
    fNTracklets(nTracklets),
    fCharge(0)
{
    //
    // Contructor to store params
    //
    fEfficiency[0] = effMin;
    fEfficiency[1] = effMax;
    if(params) memcpy(fParams, params, sizeof(Double_t) * 4);
    else memset(fParams, 0, sizeof(Double_t) * 4);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDThresholds::AliTRDPIDThresholds(Int_t nTracklets, Double_t eff, Double_t *params) :
  TObject(),
  fNTracklets(nTracklets),
  fCharge(0)
{
  //
  // Constructor used to find object in sorted list
  //
  fEfficiency[0] = fEfficiency[1] = eff;
  if(params) memcpy(fParams, params, sizeof(Double_t) * 4);
  else memset(fParams, 0, sizeof(Double_t) * 4);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDThresholds::AliTRDPIDThresholds(const AliTRDPIDThresholds &ref) :
    TObject(ref),
    fNTracklets(ref.fNTracklets),
    fCharge(ref.fCharge)
{
    //
    // Copy constructor
    //
    memcpy(fParams, ref.fParams, sizeof(Double_t) * 4);
    memcpy(fEfficiency, ref.fEfficiency, sizeof(Double_t) * 2);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDThresholds &AliTRDPIDParams::AliTRDPIDThresholds::operator=(const AliTRDPIDThresholds &ref){
    //
    // Assignment operator
    //
    if(&ref == this) return *this;

    TObject::operator=(ref);

    fNTracklets = ref.fNTracklets;
    fCharge=ref.fCharge;
    memcpy(fEfficiency, ref.fEfficiency, sizeof(Double_t) * 2);
    memcpy(fParams, ref.fParams, sizeof(Double_t) * 4);
    return *this;
}

//____________________________________________________________
Int_t AliTRDPIDParams::AliTRDPIDThresholds::Compare(const TObject *ref) const{
    //
    // Compares two objects
    // Order:
    //   First compare number of tracklets, if they are equal compare electron efficiency
    //
    const AliTRDPIDThresholds *refObj = static_cast<const AliTRDPIDThresholds *>(ref);
    if(fNTracklets < refObj->GetNTracklets()) return -1;
    else if(fNTracklets > refObj->GetNTracklets()) return 1;
    else{
        if(fEfficiency[1] < refObj->GetElectronEfficiency(0)) return -1;
        else if(fEfficiency[0] > refObj->GetElectronEfficiency(1)) return 1;
        else {
            if (fCharge<refObj->GetCharge()) return -1;
            else if (fCharge>refObj->GetCharge()) return 1;
            else return 0;
        }
    }
}

//____________________________________________________________
Bool_t AliTRDPIDParams::AliTRDPIDThresholds::IsEqual(const TObject *ref) const {
    //
    // Check for equality
    // Tracklets and Efficiency are used
    //
  
    const AliTRDPIDThresholds *refObj = dynamic_cast<const AliTRDPIDThresholds *>(ref);
    if(!refObj) return kFALSE;
    Bool_t eqNTracklets = fNTracklets == refObj->GetNTracklets();
    Bool_t eqCharge = fCharge==refObj->GetCharge();
    Bool_t eqEff = kFALSE;
    Bool_t hasRange = TMath::Abs(fEfficiency[1] - fEfficiency[0]) > kVerySmall;
    Bool_t hasRangeRef = TMath::Abs(refObj->GetElectronEfficiency(1) - refObj->GetElectronEfficiency(0)) > kVerySmall;
    
    if(hasRange && hasRangeRef){
        // Both have ranges, check if they match
        eqEff = TMath::Abs(fEfficiency[0] - refObj->GetElectronEfficiency(0)) < kVerySmall && TMath::Abs(fEfficiency[1] - refObj->GetElectronEfficiency(1)) < kVerySmall;
    } else if(hasRange){
        // this object has ranges, check if the efficiency of ref is inside the range
        eqEff = refObj->GetElectronEfficiency(0) >= fEfficiency[0] && refObj->GetElectronEfficiency(0) < fEfficiency[1];
    } else {
        // ref has ranges, check if this is in range
        eqEff = fEfficiency[0] >= refObj->GetElectronEfficiency(0) && fEfficiency[0] < refObj->GetElectronEfficiency(1);
    }

    return  eqNTracklets && eqEff && eqCharge;
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDCentrality::AliTRDPIDCentrality():
    fEntries(NULL),
    fMinCentrality(-1.),
    fMaxCentrality(-1.)
{
    //
    // Dummy constructor
    //
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDCentrality::AliTRDPIDCentrality(Double_t minCentrality, Double_t maxCentrality):
    fEntries(NULL),
    fMinCentrality(minCentrality),
    fMaxCentrality(maxCentrality)
{
    //
    // Default constructor
    //
    fEntries = new TSortedList;
    fEntries->SetOwner();
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDCentrality::AliTRDPIDCentrality(const AliTRDPIDParams::AliTRDPIDCentrality &ref):
    TObject(),
    fEntries(NULL),
    fMinCentrality(ref.fMinCentrality),
    fMaxCentrality(ref.fMaxCentrality)
{
    //
    // Copy constructor
    //
    fEntries = new TSortedList;
    // Coply entries to the new list
    TIter entries(ref.fEntries);
    TObject *o;
    while((o = entries())) fEntries->Add(o);
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDCentrality &AliTRDPIDParams::AliTRDPIDCentrality::operator=(const AliTRDPIDCentrality &ref){
    //
    // Assignment operator
    //
    if(&ref != this){
        if(fEntries) delete fEntries;
        fEntries = new TSortedList;
        TIter entries(ref.fEntries);
        TObject *o;
        while((o = entries())) fEntries->Add(o);
        fMinCentrality = ref.fMinCentrality;
        fMaxCentrality = ref.fMaxCentrality;
    }
    return *this;
}

//____________________________________________________________
AliTRDPIDParams::AliTRDPIDCentrality::~AliTRDPIDCentrality(){
    //
    // Destructor
    //
    if(fEntries) delete fEntries;
}

//____________________________________________________________
Bool_t AliTRDPIDParams::AliTRDPIDCentrality::GetThresholdParameters(Int_t ntracklets, Double_t efficiency,Double_t *params, Int_t charge) const{
    //
    // Get the threshold parameters
    //
    AliTRDPIDThresholds test(ntracklets, efficiency, charge);
    
    TObject *result = fEntries->FindObject(&test);
    if(!result){
        AliDebug(1, Form("No threshold params found for %d tracklets, an electron efficiency of %f and a charge %d", ntracklets, efficiency, charge));
        return kFALSE;
    }
    AliTRDPIDThresholds *parResult = static_cast<AliTRDPIDThresholds *>(result);
    AliDebug(1, Form("Threshold params found: NTracklets %d, Electron Efficiency %f, Charge %d", parResult->GetNTracklets(), parResult->GetElectronEfficiency(), parResult->GetCharge()));
    
    memcpy(params, parResult->GetThresholdParams(), sizeof(Double_t) * 4);


    Double_t epsilon=0.0001;
    if(((params[0]+999)<epsilon)&&((params[1]+999)<epsilon)&&((params[2]+999)<epsilon)&&((params[3]+999)<epsilon)){
      AliError("Threshold Parameters set to -999: Parameters not available for this configuration.");
      return kFALSE;
    }

    return kTRUE;
}

//____________________________________________________________
void AliTRDPIDParams::AliTRDPIDCentrality::SetThresholdParameters(Int_t ntracklets, Double_t effMin, Double_t effMax, Double_t *params, Int_t charge){
    //
    // Store new Params in the Object
    //
    if(effMin > effMax){
        AliError("Min. efficiency has to be >= max. efficiency");
        return;
    }
    if ((charge<0)||(charge>3)){
        AliError("Charge has to be between 0 and 3");
    }
    AliDebug(1, Form("Save Parameters for %d tracklets at and electron efficiency of [%f|%f]", ntracklets, effMin, effMax));
    fEntries->Add(new AliTRDPIDThresholds(ntracklets, effMin, effMax, params, charge));
}

//____________________________________________________________
void AliTRDPIDParams::AliTRDPIDCentrality::Print(Option_t *) const {
    printf("Min. Centrality: %f, Max. Centrality: %f\n", fMinCentrality, fMaxCentrality);
    printf("Available thresholds:\n");
    printf("_________________________________________\n");
    TIter objects(fEntries);
    AliTRDPIDThresholds *par;
    Double_t params[4];
    while((par = dynamic_cast<AliTRDPIDThresholds *>(objects()))){
        printf("Number of tracklets %d, Electron efficiency %f, Charge %d\n", par->GetNTracklets(), 0.5*(par->GetElectronEfficiency(0)+par->GetElectronEfficiency(1)),par->GetCharge());

	memcpy(params, par->GetThresholdParams(),sizeof(Double_t) *4);
	printf("threshold parameters: %f, %f, %f, %f\n",params[0],params[1],params[2],params[3]);
    }
}

