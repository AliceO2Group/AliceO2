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


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ALICE Reconstruction parameterization:                                    //
//                                                                           //
//                                                                           //
// Base Class for Detector reconstruction parameters                         //
// Revision: cvetan.cheshkov@cern.ch 12/06/2008                              //
// Its structure has been revised and it is interfaced to AliEventInfo.      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TClass.h"
#include "TObjArray.h"
#include "TMath.h"
#include "THashTable.h"
#include "TString.h"
#include "TRegexp.h"
#include "AliDetectorRecoParam.h"

#include "AliLog.h"
#include "AliRecoParam.h"
#include "AliRunInfo.h"
#include "AliEventInfo.h"
#include "AliLog.h"

ClassImp(AliRecoParam)

TString AliRecoParam::fkgEventSpecieName[] = {"Default", "LowMultiplicity", "HighMultiplicity", "Cosmic", "Calib", "Unknown"} ; 

AliRecoParam::AliRecoParam(): 
  TObject(),
  fEventSpecie(kDefault)
{
  // Default constructor
  // ...
  for(Int_t iDet = 0; iDet < kNDetectors; iDet++)
    fDetRecoParams[iDet] = NULL;
  for(Int_t iSpecie = 0; iSpecie < kNSpecies; iSpecie++) {
    for(Int_t iDet = 0; iDet < kNDetectors; iDet++) {
      fDetRecoParamsIndex[iSpecie][iDet] = -1;
    }
  }
}

AliRecoParam::AliRecoParam(const AliRecoParam& par) :
  TObject(),
  fEventSpecie(par.fEventSpecie)
{
  // copy constructor
  for(Int_t iDet = 0; iDet < kNDetectors; iDet++) {
    if (par.fDetRecoParams[iDet])
      fDetRecoParams[iDet] = (TObjArray*)(par.fDetRecoParams[iDet]->Clone());
    else
      fDetRecoParams[iDet] = NULL;
  }
  for(Int_t iSpecie = 0; iSpecie < kNSpecies; iSpecie++) {
    for(Int_t iDet = 0; iDet < kNDetectors; iDet++) {
      fDetRecoParamsIndex[iSpecie][iDet] = par.fDetRecoParamsIndex[iSpecie][iDet];
    }
  }
}

//_____________________________________________________________________________
AliRecoParam& AliRecoParam::operator = (const AliRecoParam& par)
{
  // assignment operator

  if(&par == this) return *this;

  this->~AliRecoParam();
  new(this) AliRecoParam(par);
  return *this;
}

AliRecoParam::~AliRecoParam(){
  // Destructor
  // ...
  // Delete the array with the reco-param objects
  for(Int_t iDet = 0; iDet < kNDetectors; iDet++) {
    if (fDetRecoParams[iDet]){
      fDetRecoParams[iDet]->Delete();
      delete fDetRecoParams[iDet];
    }
  }
}

Int_t AliRecoParam::AConvert(EventSpecie_t es)
{
  //Converts EventSpecie_t  into int
  Int_t rv = -1 ; 
  switch (es) {
    case kDefault:
      rv = 0 ; 
      break;
    case kLowMult:
      rv = 1 ; 
      break;
    case kHighMult:
      rv = 2 ; 
      break;
    case kCosmic:
      rv = 3 ; 
      break;
    case kCalib:
      rv = 4 ; 
      break;
    default:
      break;
  }

  if (rv < 0) 
    AliFatalClass(Form("Wrong event specie conversion %d", es)) ; 

  return rv ;
}

AliRecoParam::EventSpecie_t AliRecoParam::Convert(Int_t ies)
{
  //Converts int into EventSpecie_t
  AliRecoParam::EventSpecie_t es = kDefault ; 
  if ( ies >> 1) 
    es = kLowMult ; 
  if ( ies >> 2) 
    es = kHighMult ;   
  if ( ies >> 3) 
    es = kCosmic ;   
  if ( ies >> 4) 
    es = kCalib ;

  return es ;   
}

AliRecoParam::EventSpecie_t AliRecoParam::ConvertIndex(Int_t index)
{
  //Converts index of lists into eventspecie
  EventSpecie_t es = kDefault ; 
  switch (index) {
    case 0:
      es = kDefault ; 
      break;
    case 1:
      es = kLowMult ; 
      break;
    case 2:
      es = kHighMult ;   
      break;
    case 3:
      es = kCosmic ;   
      break;
    case 4:
      es = kCalib ;
      break;
    default:
      break;
  }
  return es ;
}

void  AliRecoParam::Print(Option_t *option) const {
  //
  // Print reconstruction setup
  //
  for(Int_t iDet = 0; iDet < kNDetectors; iDet++) {
    if (fDetRecoParams[iDet]){
      printf("AliDetectorRecoParam objects for detector %d:\n",iDet); 
      Int_t nparam = fDetRecoParams[iDet]->GetEntriesFast();
      for (Int_t iparam=0; iparam<nparam; iparam++){
	AliDetectorRecoParam * param = (AliDetectorRecoParam *)fDetRecoParams[iDet]->At(iparam);
	if (!param) continue;
	param->Print(option);
      }
    }
    else {
      printf("No AliDetectorRecoParam objects specified for detector %d\n",iDet); 
    }
  }
}

AliRecoParam::EventSpecie_t AliRecoParam::SuggestRunEventSpecie(const char* runTypeGRP,
						  const char* beamTypeGRP, 
						  const char* lhcStateGRP)
{
  // suggest eventSpecie according to provided runType, beamType and LHC state
  TString runType(runTypeGRP);
  if (runType != "PHYSICS") return kCalib;
  //
  TString lhcState(lhcStateGRP);
  TString beamType(beamTypeGRP);
  TRegexp reStable("^STABLE[_ ]BEAMS$");
  TRegexp reNoBeam("^NO[_ ]BEAM$");
  TRegexp reASthg("^A-");
  TRegexp reSthgA(".*-A$");
  TRegexp repSthg("^[pP]-.*");
  TRegexp reSthgp(".*-[pP]$");
  //
  EventSpecie_t evSpecie = kDefault;
  //
  if(lhcState.Index(reStable)==0){
    if(beamType.Index(repSthg)==0 || beamType.Index(reSthgp)==0){
      // Proton run, the event specie is set to kLowMult
      evSpecie = kLowMult;
    }else if(beamType.Index(reASthg)==0 || beamType.Index(reSthgA)==0){
      // Heavy ion run (any beam that is not pp, the event specie is set to kHighMult
      evSpecie = kHighMult;
    }
  }
  if(beamType=="-" || lhcState.Index(reNoBeam)==0 ) {
    // No beams, we assume cosmic data
    evSpecie = kCosmic;
  }
  //
  return evSpecie;
}


void AliRecoParam::SetEventSpecie(const AliRunInfo *runInfo, const AliEventInfo &evInfo, const THashTable *cosmicTriggersList)
{
    // Implemented according to the discussions
    // and meetings with physics and trigger coordination

    fEventSpecie = kDefault;

    // Special DAQ events considered as calibration events
    if (evInfo.GetEventType() != 7) {
	// START_OF_*, END_OF_*, CALIBRATION etc events
	fEventSpecie = kCalib;
	return;
    }
    //
    fEventSpecie = SuggestRunEventSpecie(runInfo->GetRunType(), runInfo->GetBeamType(),runInfo->GetLHCState());
    if (fEventSpecie==kCalib) return;

    if (strcmp(runInfo->GetRunType(),"PHYSICS")) {
	// Not a physics run, the event specie is set to kCalib
	fEventSpecie = kCalib;
	return;
    }

    // Now we look into the trigger type in order to decide
    // on the remaining cases (cosmic event within LHC run,
    // calibration, for example TPC laser, triggers within physics run
    //
    Bool_t cosmicTrigger = evInfo.HasCosmicTrigger();
    Bool_t calibTrigger = evInfo.HasCalibLaserTrigger();
    Bool_t otherTrigger = evInfo.HasBeamTrigger();
    // 
    // -------------------------------------------------------------- >>
    // for BWD compatibility, preserve also old way of checking
    if (!cosmicTrigger && !calibTrigger) { // not set via alias
      TString triggerClasses = evInfo.GetTriggerClasses();
      TObjArray* trClassArray = triggerClasses.Tokenize(" ");
      Int_t nTrClasses = trClassArray->GetEntriesFast();
      for( Int_t i=0; i<nTrClasses; ++i ) {
	TString trClass = ((TObjString*)trClassArray->At(i))->String();
	if (trClass.BeginsWith("C0L")) { // Calibration triggers always start with C0L
	  calibTrigger = kTRUE;
	  continue;
	}
	//
	if (cosmicTriggersList) {
	  if (cosmicTriggersList->FindObject(trClass.Data())) {
	    // Cosmic trigger accorind to the table provided in OCDB
	    cosmicTrigger = kTRUE;
	    AliDebug(1,Form("Trigger %s identified as cosmic according to the list defined in OCDB.",trClass.Data()));
	    continue;
	  }
	}
	else {
	  AliDebug(1,"Cosmic trigger list is not provided, cosmic event specie is effectively disabled!");
	}
      //
	otherTrigger = kTRUE;
      }
      delete trClassArray;
      //
    }
    // -------------------------------------------------------------- <<
    //
    if (calibTrigger) {
	fEventSpecie = kCalib;
	return;
    }
    if (cosmicTrigger && !otherTrigger) {
	fEventSpecie = kCosmic;
	return;
    }

    // Here we have to add if we have other cases
    // and also HLT info if any...
}

const AliDetectorRecoParam *AliRecoParam::GetDetRecoParam(Int_t iDet) const
{
  // Return AliDetectorRecoParam object for a given detector
  // according to the event specie provided as an argument
  if ( iDet >= kNDetectors) return NULL;
  if (!fDetRecoParams[iDet]) return NULL;
  if (fDetRecoParams[iDet]->GetEntries() == 0) return NULL;

  for(Int_t iBit = 0; iBit < kNSpecies; iBit++) {
    if (fEventSpecie & (1 << iBit)) {
      if (fDetRecoParamsIndex[iBit][iDet] >= 0)
	return (AliDetectorRecoParam *)fDetRecoParams[iDet]->At(fDetRecoParamsIndex[iBit][iDet]);
      else if (fDetRecoParamsIndex[0][iDet] >= 0)
	return (AliDetectorRecoParam *)fDetRecoParams[iDet]->At(fDetRecoParamsIndex[0][iDet]);
      else {
	AliError(Form("no RecoParam set for detector %d", iDet));
	return NULL;
      }
    }
  }

  // Default one
  AliError(Form("Invalid event specie: %d!",fEventSpecie));
  if (fDetRecoParamsIndex[0][iDet] >= 0)
    return (AliDetectorRecoParam *)fDetRecoParams[iDet]->At(fDetRecoParamsIndex[0][iDet]);

  AliError(Form("no RecoParam set for detector %d", iDet));
  return NULL;
}

void  AliRecoParam::AddDetRecoParam(Int_t iDet, AliDetectorRecoParam* param)
{
  // Add an instance of reco params object into
  // the fDetRecoParams for detector iDet
  // Updates the fDetRecoParams index
  if (!fDetRecoParams[iDet]) fDetRecoParams[iDet] = new TObjArray;
  fDetRecoParams[iDet]->AddLast(param);
  Int_t index = fDetRecoParams[iDet]->GetLast();

  // Index
  Int_t specie = param->GetEventSpecie();
  for(Int_t iBit = 0; iBit < kNSpecies; iBit++) {
    if (specie & (1 << iBit)) {
      fDetRecoParamsIndex[iBit][iDet] = index;
    }
  }
}

Bool_t AliRecoParam::AddDetRecoParamArray(Int_t iDet, TObjArray* parArray)
{
  // Add an array of reconstruction parameter objects
  // for a given detector
  // Basic check on the consistency of the array
  Bool_t defaultFound = kFALSE;
  if (!parArray) return defaultFound;
  for(Int_t i = 0; i < parArray->GetEntriesFast(); i++) {
    AliDetectorRecoParam *par = (AliDetectorRecoParam*)parArray->At(i);
    if (!par) continue;
    if (par->IsDefault()) defaultFound = kTRUE;

    Int_t specie = par->GetEventSpecie();
    for(Int_t iBit = 0; iBit < kNSpecies; iBit++) {
      if (specie & (1 << iBit)) {
	fDetRecoParamsIndex[iBit][iDet] = i;
      }
    }
 }
   
  fDetRecoParams[iDet] = parArray;

  return defaultFound;
}

const char*  AliRecoParam::PrintEventSpecie() const
{
  // Print the current
  // event specie
  switch (fEventSpecie) {
  case kDefault:
    return fkgEventSpecieName[0].Data() ;
    break;
  case kLowMult:
    return fkgEventSpecieName[1].Data() ;
    break;
  case kHighMult:
    return fkgEventSpecieName[2].Data() ;
    break;
  case kCosmic:
    return fkgEventSpecieName[3].Data() ;
    break;
  case kCalib:
    return fkgEventSpecieName[4].Data() ;
    break;
  default:
    return fkgEventSpecieName[5].Data() ;
    break;
  }
}

const char * AliRecoParam::GetEventSpecieName(EventSpecie_t es)
{
  switch (es) {
    case kDefault:
      return fkgEventSpecieName[0].Data() ;
      break;
    case kLowMult:
      return fkgEventSpecieName[1].Data() ;
      break;
    case kHighMult:
      return fkgEventSpecieName[2].Data() ;
      break;
    case kCosmic:
      return fkgEventSpecieName[3].Data() ;
      break;
    case kCalib:
      return fkgEventSpecieName[4].Data() ;
      break;
    default:
      return fkgEventSpecieName[5].Data() ;
      break;
  }
}

const char * AliRecoParam::GetEventSpecieName(Int_t esIndex)
{
  if ( esIndex >= 0 && esIndex < kNSpecies) 
    return fkgEventSpecieName[esIndex].Data() ;
  else 
    return fkgEventSpecieName[kNSpecies].Data() ;
}
