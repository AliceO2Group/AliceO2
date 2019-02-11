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
#include <TNamed.h>
#include <TGeoMatrix.h>
#include <TGeoGlobalMagField.h>

#include "AliESDRun.h"
#include "AliESDVertex.h"
#include "AliLog.h"
#include "AliMagF.h"

//-------------------------------------------------------------------------
//                     Implementation Class AliESDRun
//   Run by run data
//   for the ESD   
//   Origin: Christian Klein-Boesing, CERN, Christian.Klein-Boesing@cern.ch 
//-------------------------------------------------------------------------

ClassImp(AliESDRun)  
 
//______________________________________________________________________________
AliESDRun::AliESDRun() :
  TObject(),
  fCurrentL3(0),
  fCurrentDip(0),
  fBeamEnergy(0),
  fMagneticField(0),
  fDiamondZ(0),
  fDiamondSig2Z(0),
  fPeriodNumber(0),
  fRunNumber(0),
  fRecoVersion(0),
  fBeamType(""),
  fTriggerClasses(kNTriggerClasses),
  fDetInDAQ(0),
  fDetInReco(0),
  fCTPStart()
{
  //
  // default ctor
  //
  for (Int_t i=0; i<2; i++) fDiamondXY[i]=0.;
  fBeamParticle[0] = fBeamParticle[1] = 0;
  fDiamondCovXY[0]=fDiamondCovXY[2]=3.*3.;
  fDiamondCovXY[1]=0.;
  fTriggerClasses.SetOwner(kTRUE);
  fMeanBeamInt[0][0]=fMeanBeamInt[0][1]=fMeanBeamInt[1][0]=fMeanBeamInt[1][1]=-1;
  for (Int_t m=0; m<kNPHOSMatrix; m++) fPHOSMatrix[m]=NULL;
  for (Int_t sm=0; sm<kNEMCALMatrix; sm++) fEMCALMatrix[sm]=NULL;
  for (Int_t i=0; i<kT0spreadSize;i++) fT0spread[i]=0.;
  for (Int_t it=0; it<15; it++) fCaloTriggerType[it]=0;
  for (Int_t it=0; it<19; it++) fCaloTriggerTypeNew[it]=0;
  for (Int_t j=0; j<64; ++j) fVZEROEqFactors[j]=-1;
}

//______________________________________________________________________________
AliESDRun::AliESDRun(const AliESDRun &esd) :
  TObject(esd),
  fCurrentL3(0),
  fCurrentDip(0),
  fBeamEnergy(0),
  fMagneticField(esd.fMagneticField),
  fDiamondZ(esd.fDiamondZ),
  fDiamondSig2Z(esd.fDiamondSig2Z),
  fPeriodNumber(esd.fPeriodNumber),
  fRunNumber(esd.fRunNumber),
  fRecoVersion(esd.fRecoVersion),
  fBeamType(""),
  fTriggerClasses(TObjArray(kNTriggerClasses)),
  fDetInDAQ(0),
  fDetInReco(0),
  fCTPStart(esd.fCTPStart)
{ 
  //
  // Copy constructor
  //
  for (Int_t i=0; i<2; i++) fDiamondXY[i]=esd.fDiamondXY[i];
  for (Int_t i=0; i<3; i++) fDiamondCovXY[i]=esd.fDiamondCovXY[i];
  for (Int_t i=0; i<2; i++) fBeamParticle[i] = esd.fBeamParticle[i];
  for(Int_t i = 0; i < kNTriggerClasses; i++) {
    TNamed *str = (TNamed *)((esd.fTriggerClasses).At(i));
    if (str) fTriggerClasses.AddAt(new TNamed(*str),i);
  }

  for(Int_t m=0; m<kNPHOSMatrix; m++){
    if(esd.fPHOSMatrix[m])
      fPHOSMatrix[m]=new TGeoHMatrix(*(esd.fPHOSMatrix[m])) ;
    else
      fPHOSMatrix[m]=NULL;
  }
  
  for (int ib=2;ib--;) for (int it=2;it--;) fMeanBeamInt[ib][it] = esd.fMeanBeamInt[ib][it];

  for(Int_t sm=0; sm<kNEMCALMatrix; sm++){
	if(esd.fEMCALMatrix[sm])
		fEMCALMatrix[sm]=new TGeoHMatrix(*(esd.fEMCALMatrix[sm])) ;
	else
		fEMCALMatrix[sm]=NULL;
  }
  for (Int_t i=0; i<kT0spreadSize;i++) fT0spread[i]=esd.fT0spread[i];
  for (Int_t it=0; it<15; it++) fCaloTriggerType[it]=esd.fCaloTriggerType[it];
  for (Int_t it=0; it<19; it++) fCaloTriggerTypeNew[it]=esd.fCaloTriggerTypeNew[it];
  for (Int_t j=0; j<64; ++j) fVZEROEqFactors[j]=esd.fVZEROEqFactors[j];

}

//______________________________________________________________________________
AliESDRun& AliESDRun::operator=(const AliESDRun &esd)
{ 
  // assigment operator
  if(this!=&esd) {
    TObject::operator=(esd);
    fRunNumber=esd.fRunNumber;
    fPeriodNumber=esd.fPeriodNumber;
    fRecoVersion=esd.fRecoVersion;
    fMagneticField=esd.fMagneticField;
    fDiamondZ=esd.fDiamondZ;
    fDiamondSig2Z=esd.fDiamondSig2Z;
    fBeamType = esd.fBeamType;
    fCurrentL3  = esd.fCurrentL3;
    fCurrentDip = esd.fCurrentDip;
    fBeamEnergy = esd.fBeamEnergy;
    for (Int_t i=0; i<2; i++) fDiamondXY[i]=esd.fDiamondXY[i];
    for (Int_t i=0; i<3; i++) fDiamondCovXY[i]=esd.fDiamondCovXY[i];
    for (Int_t i=0; i<2; i++) fBeamParticle[i] = esd.fBeamParticle[i];
    fTriggerClasses.Clear();
    for(Int_t i = 0; i < kNTriggerClasses; i++) {
      TNamed *str = (TNamed *)((esd.fTriggerClasses).At(i));
      if (str) fTriggerClasses.AddAt(new TNamed(*str),i);
    }

    fDetInDAQ   = esd.fDetInDAQ;
    fDetInReco  = esd.fDetInReco;

    for (int ib=2;ib--;) for (int it=2;it--;) fMeanBeamInt[ib][it] = esd.fMeanBeamInt[ib][it];

    for(Int_t m=0; m<kNPHOSMatrix; m++){
      delete fPHOSMatrix[m];
      if(esd.fPHOSMatrix[m])
	fPHOSMatrix[m]=new TGeoHMatrix(*(esd.fPHOSMatrix[m])) ;
      else
	fPHOSMatrix[m]=0;
    }
	  
    for(Int_t sm=0; sm<kNEMCALMatrix; sm++){
      delete fEMCALMatrix[sm];
      if(esd.fEMCALMatrix[sm])
        fEMCALMatrix[sm]=new TGeoHMatrix(*(esd.fEMCALMatrix[sm])) ;
      else
        fEMCALMatrix[sm]=0;
    }
  } 
  for (Int_t i=0; i<kT0spreadSize;i++) fT0spread[i]=esd.fT0spread[i];
  for (Int_t it=0; it<15; it++) fCaloTriggerType[it]=esd.fCaloTriggerType[it];
  for (Int_t it=0; it<19; it++) fCaloTriggerTypeNew[it]=esd.fCaloTriggerTypeNew[it];
  for (Int_t j=0; j<64; ++j) fVZEROEqFactors[j]=esd.fVZEROEqFactors[j];
  fCTPStart = esd.fCTPStart;
  return *this;
}

void AliESDRun::Copy(TObject &obj) const{

  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDRun *robj = dynamic_cast<AliESDRun*>(&obj);
  if(!robj)return; // not an aliesdrun
  *robj = *this;

}

//______________________________________________________________________________
AliESDRun::~AliESDRun() {
  // Destructor
  // Delete PHOS position matrices
  for(Int_t m=0; m<kNPHOSMatrix; m++) {
    if(fPHOSMatrix[m]) delete fPHOSMatrix[m] ;
    fPHOSMatrix[m] = NULL;
  }
  // Delete PHOS position matrices
  for(Int_t sm=0; sm<kNEMCALMatrix; sm++) {
	if(fEMCALMatrix[sm]) delete fEMCALMatrix[sm] ;
	fEMCALMatrix[sm] = NULL;
  }
}

void AliESDRun::SetDiamond(const AliESDVertex *vertex) {
  // set the interaction diamond
  if (vertex) {
    fDiamondXY[0]=vertex->GetX();
    fDiamondXY[1]=vertex->GetY();
    fDiamondZ=vertex->GetZ();
    Double32_t cov[6];
    vertex->GetCovMatrix(cov);
    fDiamondCovXY[0]=cov[0];
    fDiamondCovXY[1]=cov[1];
    fDiamondCovXY[2]=cov[2];
    fDiamondSig2Z=cov[5];
  }
}


//______________________________________________________________________________
void AliESDRun::Print(const Option_t *) const
{
  // Print some data members
  printf("Mean vertex in RUN %d: X=%.4f Y=%.4f Z=%.4f cm\n",
	 GetRunNumber(),GetDiamondX(),GetDiamondY(),GetDiamondZ());
  printf("Beam Type: %s (%d/%d - %d/%d), Energy: %.1f GeV\n",fBeamType.IsNull() ? "N/A":GetBeamType(),
	 GetBeamParticleA(0),GetBeamParticleZ(0),GetBeamParticleA(1),GetBeamParticleZ(1),
	 fBeamEnergy);
  printf("Magnetic field in IP= %f T | Currents: L3:%+.1f Dipole:%+.1f %s\n",
	 GetMagneticField(),fCurrentL3,fCurrentDip,TestBit(kUniformBMap) ? "(Uniform)":"");
  printf("Event from reconstruction version %d \n",fRecoVersion);
  
  printf("List of active trigger classes: ");
  for(Int_t i = 0; i < kNTriggerClasses; i++) {
    TNamed *str = (TNamed *)((fTriggerClasses).At(i));
    if (str) printf("%s ",str->GetName());
  }
  printf("Mean intenstity for interacting   : beam1:%+.3e beam2:%+.3e\n",fMeanBeamInt[0][0],fMeanBeamInt[1][0]);
  printf("Mean intenstity for non-intecting : beam1:%+.3e beam2:%+.3e\n",fMeanBeamInt[0][1],fMeanBeamInt[1][1]);
  printf("\n");
}

void AliESDRun::Reset() 
{
  // reset data members
  fRunNumber = 0;
  fPeriodNumber = 0;
  fRecoVersion = 0;
  fMagneticField = 0;
  fCurrentL3 = 0;
  fCurrentDip = 0;
  fBeamEnergy = 0;
  fBeamType = "";
  ResetBit(kBInfoStored|kUniformBMap|kConvSqrtSHalfGeV);
  for (Int_t i=0; i<2; i++) fDiamondXY[i]=0.;
  fDiamondCovXY[0]=fDiamondCovXY[2]=3.*3.;
  fDiamondCovXY[1]=0.;
  fDiamondZ=0.;
  fDiamondSig2Z=10.*10.;
  fTriggerClasses.Clear();
  fDetInDAQ   = 0;
  fDetInReco  = 0;
}

//______________________________________________________________________________
void AliESDRun::SetTriggerClass(const char* name, Int_t index)
{
  // Fill the trigger class name
  // into the corresponding array
  if (index >= kNTriggerClasses || index < 0) {
    AliError(Form("Index (%d) is outside the allowed range (0,49)!",index));
    return;
  }

  fTriggerClasses.AddAt(new TNamed(name,NULL),index);
}

//______________________________________________________________________________
const char* AliESDRun::GetTriggerClass(Int_t index) const
{
  // Get the trigger class name at
  // specified position in the trigger mask
  TNamed *trclass = (TNamed *)fTriggerClasses.At(index);
  if (trclass)
    return trclass->GetName();
  else
    return "";
}

//______________________________________________________________________________
TString AliESDRun::GetActiveTriggerClasses() const
{
  // Construct and return
  // the list of trigger classes
  // which are present in the run
  TString trclasses;
  for(Int_t i = 0; i < kNTriggerClasses; i++) {
    TNamed *str = (TNamed *)((fTriggerClasses).At(i));
    if (str) {
      trclasses += " ";
      trclasses += str->GetName();
      trclasses += " ";
    }
  }

  return trclasses;
}

//______________________________________________________________________________
TString AliESDRun::GetFiredTriggerClasses(ULong64_t mask) const
{
  // Constructs and returns the
  // list of trigger classes that
  // have been fired. Uses the trigger
  // class mask as an argument.
  // Works for first50
  TString trclasses;
  for(Int_t i = 0; i < kNTriggerClasses/2; i++) {
    if (mask & (1ull << i)) {
      TNamed *str = (TNamed *)((fTriggerClasses).At(i));
      if (str) {
	trclasses += " ";
	trclasses += str->GetName();
      trclasses += " ";
      }
    }
  }

  return trclasses;
}
//______________________________________________________________________________
TString AliESDRun::GetFiredTriggerClassesNext50(ULong64_t mask) const
{
  // Constructs and returns the
  // list of trigger classes that
  // have been fired. Uses the trigger
  // class mask as an argument.
  // Works for next50 classes
  TString trclasses;
  for(Int_t i = 0; i < kNTriggerClasses/2; i++) {
    if (mask & (1ull << i)) {
      TNamed *str = (TNamed *)((fTriggerClasses).At(i+50));
      if (str) {
	trclasses += " ";
	trclasses += str->GetName();
      trclasses += " ";
      }
    }
  }
  return trclasses;
}
//______________________________________________________________________________
TString AliESDRun::GetFiredTriggerClasses(ULong64_t masklow,ULong64_t maskhigh) const
{
 // Contruct and returns list of trigger classes for 100 classes
 TString trclasseslow;
 trclasseslow  = GetFiredTriggerClasses(masklow);
 TString trclasseshigh;
 trclasseshigh  = GetFiredTriggerClassesNext50(maskhigh);
 TString trclasses;
 trclasses = trclasseslow+trclasseshigh;
 return trclasses;
}
//______________________________________________________________________________
void AliESDRun::PrintAllTriggerClasses() const
{
  TString trclasses;
  for(Int_t i = 0; i < kNTriggerClasses; i++) {
      TNamed *str = (TNamed *)((fTriggerClasses).At(i));
      if (str) {
        printf("%03i:",i+1);
	printf("%s ",str->GetName());
      }else{
	//printf("NO ");
      }
  }
  printf("\n");
}
//______________________________________________________________________________
Bool_t AliESDRun::IsTriggerClassFired(ULong64_t mask, const char *name) const
{
  // Checks if the trigger class
  // identified by 'name' has been
  // fired. Uses the trigger class mask.
  // To be used for first classes 0-49

  TNamed *trclass = (TNamed *)fTriggerClasses.FindObject(name);
  if (!trclass) return kFALSE;

  Int_t iclass = fTriggerClasses.IndexOf(trclass);
  if (iclass < 0) return kFALSE;
  if (iclass >= 50) return kFALSE;

  if (mask & (1ull << iclass))
    return kTRUE;
  else
    return kFALSE;
}
//______________________________________________________________________________
Bool_t AliESDRun::IsTriggerClassFiredNext50(ULong64_t mask, const char *name) const
{
  // Checks if the trigger class
  // identified by 'name' has been
  // fired. Uses the trigger class mask.
  // To be used for first classes 50-99

  TNamed *trclass = (TNamed *)fTriggerClasses.FindObject(name);
  if (!trclass) return kFALSE;

  Int_t iclass = fTriggerClasses.IndexOf(trclass);
  if (iclass < 50) return kFALSE;
  if (iclass >= 100) return kFALSE;

  if (mask & (1ull << (iclass-50)))
    return kTRUE;
  else
    return kFALSE;
}
//______________________________________________________________________________
Bool_t AliESDRun::IsTriggerClassFired(ULong64_t masklow, ULong64_t maskhigh,const char *name) const
{
 return (IsTriggerClassFired(masklow,name) || IsTriggerClassFiredNext50(maskhigh,name));
}
//_____________________________________________________________________________
Bool_t AliESDRun::InitMagneticField() const
{
  // Create mag field from stored information
  //
  if (!TestBit(kBInfoStored)) {
    AliError("No information on currents, cannot create field from run header");
    return kFALSE;
  }
  //
  AliMagF* fld = (AliMagF*) TGeoGlobalMagField::Instance()->GetField();
  if (fld) {
    if (TGeoGlobalMagField::Instance()->IsLocked()) {
      if (fld->TestBit(AliMagF::kOverrideGRP)) {
	AliInfo("ExpertMode!!! Information on magnet currents will be ignored !");
	AliInfo("ExpertMode!!! Running with the externally locked B field !");
	return kTRUE;
      }
    }
    AliInfo("Destroying existing B field instance!");
    delete TGeoGlobalMagField::Instance();
  }
  //
  fld = AliMagF::CreateFieldMap(fCurrentL3,fCurrentDip,AliMagF::kConvLHC,
				TestBit(kUniformBMap), GetBeamEnergy(), GetBeamType(),
				GetBeamParticle(0),GetBeamParticle(1));
  if (fld) {
    TGeoGlobalMagField::Instance()->SetField( fld );
    TGeoGlobalMagField::Instance()->Lock();
    AliInfo("Running with the B field constructed out of the Run Header !");
    return kTRUE;
  }
  else {
    AliError("Failed to create a B field map !");
    return kFALSE;
  }
  //
}

//_____________________________________________________________________________
void AliESDRun::SetT0spread(Int_t i,Float_t t) 
{
  //
  // Setting the T0 spread value at index i 
  //

  if ( (i>=0) && (i<kT0spreadSize)) {
    fT0spread[i]=t;
  } else {
    AliError(Form("Index %d out of bound",i));
  }
  return;
}

//_____________________________________________________________________________
void AliESDRun::SetT0spread(Float_t *t) 
{
  //
  // Setting the T0 spread values
  //
  if (t == 0x0){
    AliError(Form("Null pointer passed"));
  }
  else{
    for (Int_t i=0;i<kT0spreadSize;i++) fT0spread[i]=t[i];
  }
  return;
}

