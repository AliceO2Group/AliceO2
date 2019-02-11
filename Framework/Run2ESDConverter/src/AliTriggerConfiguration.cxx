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
//
// This class which defines defines the Trigger Configuration 
//
// Trigger Configuration defines the trigger setup for a particular run
// We have default configurations for each running mode (Pb-Pb, p-p, p-A, Calibration, etc).
// It keep:
//   All the information conained in the CTP configuration file used
//   online during the data taking
//
// Configurations could be created and stored in local file.
// By default the configuration is loaded from the corresponding GRP entry
// inside the OCDB. There one can have one and only one configuration per run.
//
//   Example how to create a Trigger Configuration:
//
//   AliTriggerConfiguration config( "TEST", "Test Configuration" );
//
//   // Define a Cluster Detector
//   config.AddDetectorCluster( "VZERO ZDC MUON" );
//
//   // Define the trigger conditions (see AliTriggerCondition.cxx)
//   config.AddCondition( "VZERO_TEST1_L0 & MUON_SPlus_LPt_L0 & ZDC_TEST2_L0", // condition
//                         "VO1_M1_ZDC2",      // short name
//                         "Dummy",            // short description
//                          0x0100 );          // class mask (set one bit)
//
//   config.AddCondition( "VZERO_TEST2_L0 & MUON_SMinus_HPt_L0 & ZDC_TEST1_L0",
//                         "VO2_M3_ZDC1",
//                         "Dummy",
//                          0x0200 );
//
//   config.AddCondition( "VZERO_TEST3_L0 | MUON_Unlike_LPt_L0 | ZDC_TEST3_L0",
//                         "VO3_M1_ZDC3",
//                         "Dummy",
//                          0x0400 );
//   config.CheckInputsConditions("Config.C");
//   config.Print();
//
//   // save the configuration to file 
//   // (default file name $ALICE_ROOT/data/triggerConfigurations.root)
//   config.WriteConfiguration(); or config.WriteConfiguration( filename );
//
///////////////////////////////////////////////////////////////////////////////
#include <Riostream.h>

#include <TFile.h>
#include <TKey.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TObject.h>
#include <TROOT.h>
#include <TString.h>
#include <TSystem.h>
#include <TMath.h>

#include "AliLog.h"
#include "AliTriggerBCMask.h"
#include "AliTriggerClass.h"
#include "AliTriggerCluster.h"
#include "AliTriggerConfiguration.h"
#include "AliTriggerDescriptor.h"
#include "AliTriggerInput.h"
#include "AliTriggerInteraction.h"
#include "AliTriggerPFProtection.h"

using std::endl;
using std::cout;
using std::ifstream;
ClassImp(AliTriggerConfiguration)

const TString AliTriggerConfiguration::fgkConfigurationFileName("/data/triggerConfigurations.root");

//_____________________________________________________________________________
AliTriggerConfiguration::AliTriggerConfiguration():
  TNamed(),
  fInputs(),
  fInteractions(),
  fFunctions(),
  fPFProtections(),
  fMasks(),
  fDescriptors(),
  fClusters(),
  fClasses(),
  fVersion(0)
{
  // Default constructor
}

//_____________________________________________________________________________
AliTriggerConfiguration::AliTriggerConfiguration( TString & name, TString & description ):
  TNamed( name, description ),
  fInputs(),
  fInteractions(),
  fFunctions(),
  fPFProtections(),
  fMasks(),
  fDescriptors(),
  fClusters(),
  fClasses(),
  fVersion(0)
{
  // Constructor
}

//_____________________________________________________________________________
AliTriggerConfiguration::~AliTriggerConfiguration() 
{ 
  // Destructor
  fInputs.SetOwner();
  fInputs.Delete();
  fInteractions.SetOwner();
  fInteractions.Delete();
  fFunctions.SetOwner();
  fFunctions.Delete();
  fPFProtections.SetOwner();
  fPFProtections.Delete();
  fMasks.SetOwner();
  fMasks.Delete();
  fDescriptors.SetOwner();
  fDescriptors.Delete();
  fClusters.SetOwner(); 
  fClusters.Delete(); 
  fClasses.SetOwner(); 
  fClasses.Delete(); 
}

//_____________________________________________________________________________
Bool_t AliTriggerConfiguration::AddInput( AliTriggerInput* input )
{
  // Add a trigger input to
  // the list of the trigger inputs
  if (fInputs.GetEntries() < kNMaxInputs) {
    fInputs.AddLast( input );
    return kTRUE;
  }
  else {
    AliError("CTP can handle up to 64 inputs ! Impossible to add the required input !");
    return kFALSE;
  }
}

//_____________________________________________________________________________
AliTriggerInput* AliTriggerConfiguration::AddInput( TString &name, TString &det,
						    UChar_t level, UInt_t signature,
						    UChar_t number )
{
  // Add a trigger input to
  // the list of the trigger inputs
  AliTriggerInput *input = new AliTriggerInput(name,det,level,signature,number);
  if (!AddInput(input)) {
    delete input;
    return NULL;
  }
  else
    return input;
}
//_____________________________________________________________________________
AliTriggerInput* AliTriggerConfiguration::AddInput( TString &name, TString &det,
						    UChar_t level, UInt_t signature,
						    UInt_t indexCTP, UInt_t indexSwitch )
{
  // Add a trigger input to
  // the list of the trigger inputs
  AliTriggerInput *input = new AliTriggerInput(name,det,level,signature,indexCTP,indexSwitch);
  if (!AddInput(input)) {
    delete input;
    return NULL;
  }
  else
    return input;
}

//_____________________________________________________________________________
AliTriggerInteraction* AliTriggerConfiguration::AddInteraction(TString &name, TString &logic)
{
  // Add a trigger interaction object to
  // the list of the trigger interactions
  AliTriggerInteraction *interact = new AliTriggerInteraction(name,logic);
  if (!AddInteraction(interact)) {
    delete interact;
    return NULL;
  }
  else
    return interact;
}

//_____________________________________________________________________________
Bool_t  AliTriggerConfiguration::AddInteraction(AliTriggerInteraction *interact)
{
  // Add a trigger interaction object to
  // the list of the trigger interactions
  if (fInteractions.GetEntries() < kNMaxInteractions) {
    if (interact->CheckInputs(fInputs)) {
      fInteractions.AddLast( interact );
      return kTRUE;
    }
    else
      AliError("Invalid interaction ! Impossible to add it !");
  }
  else
    AliError("CTP can handle up to 2 interactions ! Impossible to add the required interaction !");

  return kFALSE;
}

//_____________________________________________________________________________
AliTriggerInteraction* AliTriggerConfiguration::AddFunction(TString &name, TString &logic)
{
  // Add a trigger function object to
  // the list of the trigger functions
  AliTriggerInteraction *func = new AliTriggerInteraction(name,logic);
  if (!AddFunction(func)) {
    delete func;
    return NULL;
  }
  else
    return func;
}

//_____________________________________________________________________________
Bool_t  AliTriggerConfiguration::AddFunction(AliTriggerInteraction *func)
{
  // Add a trigger function object to
  // the list of the trigger functions
  if (fFunctions.GetEntries() < kNMaxFunctions) {
    if (func->CheckInputs(fInputs)) {
      fFunctions.AddLast( func );
      return kTRUE;
    }
    else
      AliError("Invalid logical function ! Impossible to add it !");
  }
  else
    AliError("CTP can handle up to 4 logical functions ! Impossible to add the required interaction !");

  return kFALSE;
}

//_____________________________________________________________________________
Bool_t AliTriggerConfiguration::AddPFProtection( AliTriggerPFProtection* pfp )
{
  // Add a trigger past-future protection object to
  // the list of the trigger past-future protections
  if (fPFProtections.GetEntries() < kNMaxPFProtections) {
    //if (pfp->CheckInteractions(fInteractions)) {
    if (1) {
      fPFProtections.AddLast( pfp );
      return kTRUE;
    }
    else
      AliError("Invalid past-future protection ! Impossible to add it !");
  }
  else
    AliError("CTP can handle up to 4 past-future protections ! Impossible to add the required protection !");

  return kFALSE;
}

//_____________________________________________________________________________
AliTriggerBCMask* AliTriggerConfiguration::AddMask( TString &name, TString &mask )
{
  // Add a trigger bunch-crossing mask object to
  // the list of the trigger bunch-crossing masks
  AliTriggerBCMask *bcmask = new AliTriggerBCMask(name,mask);
  if (!AddMask(bcmask)) {
    delete bcmask;
    return NULL;
  }
  else
    return bcmask;
}

//_____________________________________________________________________________
Bool_t AliTriggerConfiguration::AddMask( AliTriggerBCMask* mask )
{
  // Add a trigger bunch-crossing mask object to
  // the list of the trigger bunch-crossing masks
  if (fMasks.GetEntries() < (kNMaxMasks)) {  
      fMasks.AddLast( mask );
      return kTRUE;
  }
  else
    AliError("CTP can handle up to 12 bunch-crossing masks ! Impossible to add the required mask !");

  return kFALSE;
}

//_____________________________________________________________________________
AliTriggerCluster* AliTriggerConfiguration::AddCluster( TString &name, UChar_t index, TString &detectors)
{
  // Add a trigger detector readout cluster to
  // the list of the trigger clusters
  AliTriggerCluster *clust = new AliTriggerCluster(name,index,detectors);
  if (!AddCluster(clust)) {
    delete clust;
    return NULL;
  }
  else
    return clust;

}

//_____________________________________________________________________________
Bool_t AliTriggerConfiguration::AddCluster( AliTriggerCluster* cluster )
{
  // Add a trigger detector readout cluster to
  // the list of the trigger clusters
  if (fClusters.GetEntries() < kNMaxClusters) {
    TString dets(cluster->GetDetectorsInCluster());
    if (!(dets.IsNull())) {
      fClusters.AddLast( cluster );
      return kTRUE;
    }
    else
      AliError("Empty trigger cluster ! Impossible to add it !");
  }
  else
    AliError("CTP can handle up to 6 different detector clusters ! Impossible to add the required cluster !");

  return kFALSE;
}

//_____________________________________________________________________________
TString AliTriggerConfiguration::GetActiveDetectors() const
{
  // Return an string with all active detector
  // from each cluster

   TString activeDet = "";

   Int_t nclus = fClusters.GetEntriesFast();
   if( !nclus ) return activeDet;
   
   for( Int_t j=0; j<nclus; ++j ) {
      TString detStr = ((AliTriggerCluster*)fClusters.At(j))->GetDetectorsInCluster();
      TObjArray* det = detStr.Tokenize(" ");
      Int_t ndet = det->GetEntriesFast();
      for( Int_t k=0; k<ndet; ++k ) {
         if( activeDet.Contains( ((TObjString*)det->At(k))->String() ) )continue;
         activeDet.Append( " " );
         activeDet.Append( ((TObjString*)det->At(k))->String() );
      }
      delete det;
   }
   return activeDet;
}

//_____________________________________________________________________________
TString AliTriggerConfiguration::GetTriggeringDetectors() const
{
  // Return an string with all detectors
  // used for triggering

   TString trDet = "";

   Int_t ninputs = fInputs.GetEntriesFast();
   if( !ninputs ) return trDet;
   
   for( Int_t j=0; j<ninputs; j++ ) {
      TString detStr = ((AliTriggerInput*)fInputs.At(j))->GetDetector();
      if( trDet.Contains( detStr ) ) continue;
      trDet.Append( " " );
      trDet.Append( detStr );
   }
   return trDet;
}

//_____________________________________________________________________________
TString AliTriggerConfiguration::GetTriggeringModules() const
{
   // Return an string with all detectors (modules in the AliRoot
  // simulation sense) used for triggering

   TString trDet = "";

   Int_t ninputs = fInputs.GetEntriesFast();
   if( !ninputs ) return trDet;
   
   for( Int_t j=0; j<ninputs; j++ ) {
      TString detStr = ((AliTriggerInput*)fInputs.At(j))->GetModule();
      if( trDet.Contains( detStr ) ) continue;
      trDet.Append( " " );
      trDet.Append( detStr );
   }
   return trDet;
}

//_____________________________________________________________________________
AliTriggerDescriptor* AliTriggerConfiguration::AddDescriptor( TString &name, TString &cond )
{
  // Add a trigger descriptor to
  // the list of the trigger descriptors
  AliTriggerDescriptor *desc = new AliTriggerDescriptor(name,cond);
  if (!AddDescriptor(desc)) {
    delete desc;
    return NULL;
  }
  else
    return desc;
}

//_____________________________________________________________________________
Bool_t AliTriggerConfiguration::AddDescriptor( AliTriggerDescriptor *desc )
{
  // Add a trigger descriptor to
  // the list of the trigger descriptors
  if (fDescriptors.GetEntries() < kNMaxClasses) {
    if (desc->CheckInputsAndFunctions(fInputs,fFunctions)) {
      fDescriptors.AddLast( desc );
      return kTRUE;
    }
    else
      AliError("Invalid trigger desciptor ! Impossible to add it !");
  }
  else
    AliError("CTP can handle up to 50 different descriptors ! Impossible to add the required descriptor !");

  return kFALSE;
}

//_____________________________________________________________________________
Bool_t AliTriggerConfiguration::AddClass( AliTriggerClass *trclass )
{
  // Add a trigger class to
  // the list of the trigger classes
  if (fClasses.GetEntries() < kNMaxClasses) {
    if (trclass->CheckClass(this)) {
      fClasses.AddLast( trclass );
      return kTRUE;
    }
    else
      AliError("Invalid trigger class ! Impossible to add it !");
  }
  else
    AliError("CTP can handle up to 50 different classes ! Impossible to add the required class !");

  return kFALSE;
}

//_____________________________________________________________________________
AliTriggerClass *AliTriggerConfiguration::AddClass( TString &name, UChar_t index,
						    AliTriggerDescriptor *desc, AliTriggerCluster *clus,
						    AliTriggerPFProtection *pfp, AliTriggerBCMask *mask,
						    UInt_t prescaler, Bool_t allrare)
{
  // Add a trigger class to
  // the list of the trigger classes
  if (!fDescriptors.FindObject(desc)) {
    AliError("Invalid descriptor ! Impossible to add the class !");
    return NULL;
  }
  if (!fClusters.FindObject(clus)) {
    AliError("Invalid cluster ! Impossible to add the class !");
    return NULL;
  }
  if (!fPFProtections.FindObject(pfp)) {
    AliError("Invalid past-future protection ! Impossible to add the class !");
    return NULL;
  }
  if (!fMasks.FindObject(mask)) {
    AliError("Invalid bunch-crossing mask ! Impossible to add the class !");
    return NULL;
  }
  AliTriggerClass* trclass = new AliTriggerClass( name,index,desc,clus,pfp,mask,prescaler,allrare );
  if (!AddClass(trclass)) {
    delete trclass;
    return NULL;
  }
  else
    return trclass;
}

//_____________________________________________________________________________
AliTriggerClass *AliTriggerConfiguration::AddClass( TString &name, UChar_t index,
						    TString &desc, TString &clus,
						    TString &pfp, TString &mask,
						    UInt_t prescaler, Bool_t allrare)
{
   // Add a new trigger class
  if (!fDescriptors.FindObject(desc)) {
    AliError("Invalid descriptor ! Impossible to add the class !");
    return NULL;
  }
  if (!fClusters.FindObject(clus)) {
    AliError("Invalid cluster ! Impossible to add the class !");
    return NULL;
  }
  if (!fPFProtections.FindObject(pfp)) {
    AliError("Invalid past-future protection ! Impossible to add the class !");
    return NULL;
  }
  if (!fMasks.FindObject(mask)) {
    AliError("Invalid bunch-crossing mask ! Impossible to add the class !");
    return NULL;
  }
  AliTriggerClass* trclass = new AliTriggerClass( this, name,index,desc,clus,pfp,mask,prescaler,allrare );
  if (!AddClass(trclass)) {
    delete trclass;
    return NULL;
  }
  else
    return trclass;
}

//_____________________________________________________________________________
Bool_t AliTriggerConfiguration::ProcessConfigurationLine(const char* line, Int_t& level)
{
    // processes one line of configuration

     TString strLine(line);

     if (strLine.BeginsWith("#")) return kTRUE;
     if (strLine.BeginsWith("PARTITION:")) {
       strLine.ReplaceAll("PARTITION:","");
       SetName(strLine.Data());
       return kTRUE;
     }
     if (strLine.BeginsWith("VERSION:")) {
       strLine.ReplaceAll("VERSION:","");
       fVersion = strLine.Atoi();
       return kTRUE;
     }
     if (strLine.BeginsWith("INPUTS:")) {
       level = 1;
       return kTRUE;
     }
     if (strLine.BeginsWith("INTERACTIONS:")) {
       level = 2;
       return kTRUE;
     }
     if (strLine.BeginsWith("DESCRIPTORS:")) {
       level = 3;
       return kTRUE;
     }
     if (strLine.BeginsWith("CLUSTERS:")) {
       level = 4;
       return kTRUE;
     }
     if (strLine.BeginsWith("PFS:")) {
       level = 5;
       return kTRUE;
     }
     if (strLine.BeginsWith("BCMASKS:")) {
       level = 6;
       return kTRUE;
     }
     if (strLine.BeginsWith("CLASSES:")) {
       level = 7;
       return kTRUE;
     }

     strLine.ReplaceAll("*",'!');
     strLine.ReplaceAll("~",'!');
     TObjArray *tokens = strLine.Tokenize(" \t");
     Int_t ntokens = tokens->GetEntriesFast();
     if (ntokens == 0)
     {
       delete tokens;
       return kTRUE;
     }
     switch (level) {
     case 1:
       // Read inputs
       if ((ntokens != 5) && (ntokens !=6)) {
	 AliError(Form("Invalid trigger input syntax (%s)!",strLine.Data()));
	 delete tokens;
	 return kFALSE;
       }
       if(ntokens==5){
         AddInput(((TObjString*)tokens->At(0))->String(),
		     ((TObjString*)tokens->At(1))->String(),
		     ((TObjString*)tokens->At(2))->String().Atoi(),
		     ((TObjString*)tokens->At(3))->String().Atoi(),
		     ((TObjString*)tokens->At(4))->String().Atoi());
       }else{		     
         AddInput(((TObjString*)tokens->At(0))->String(),
		     ((TObjString*)tokens->At(1))->String(),
		     ((TObjString*)tokens->At(2))->String().Atoi(),
		     ((TObjString*)tokens->At(3))->String().Atoi(),
		     ((TObjString*)tokens->At(4))->String().Atoi(),
		     ((TObjString*)tokens->At(5))->String().Atoi());
       }		     
       break;
     case 2:
       // Read interaction
       {
        TString inter;
        if (ntokens < 2) {
	 AliError(Form("Invalid trigger interaction syntax (%s)!",strLine.Data()));
	 delete tokens;
	 return kFALSE;
        } else if (ntokens == 2) {
         inter=((TObjString*)tokens->At(1))->String();
        } else {
	 AliWarning(Form("Trigger interaction syntax (%s)!",strLine.Data()));
	 for(Int_t i=1;i<ntokens;i++){
	    inter=inter+((TObjString*)tokens->At(i))->String();
	 }
        }
        AddInteraction(((TObjString*)tokens->At(0))->String(),inter);
        break;
       }
     case 3:
       // Read logical functions and descriptors
       if (ntokens < 2) {
	 if ((((TObjString*)tokens->At(0))->String().CompareTo("EMPTY") == 0) ||
	     (((TObjString*)tokens->At(0))->String().CompareTo("DTRUE") == 0) ||
	     (((TObjString*)tokens->At(0))->String().CompareTo("DEMPTY") == 0)) {
	   AddDescriptor(((TObjString*)tokens->At(0))->String(),
			 strLine.ReplaceAll(((TObjString*)tokens->At(0))->String(),""));
	   break;
	 }
	 else {
	   AliError(Form("Invalid trigger descriptor syntax (%s)!",strLine.Data()));
	   delete tokens;
	   return kFALSE;
	 }
       }
       if (((TObjString*)tokens->At(0))->String().BeginsWith("l0f")) {
	 // function
	 if(!AddFunction(((TObjString*)tokens->At(0))->String(),
			 strLine.ReplaceAll(((TObjString*)tokens->At(0))->String(),""))) {
	   delete tokens;
	   return kFALSE;
	 }
       }
       else {
	 if(!AddDescriptor(((TObjString*)tokens->At(0))->String(),
			   strLine.ReplaceAll(((TObjString*)tokens->At(0))->String(),""))) {
	   delete tokens;
	   return kFALSE;
	 }
       }
       break;
     case 4:
       {
         if (ntokens < 2) {
           AliError(Form("Invalid trigger cluster syntax (%s)!",strLine.Data()));
	   delete tokens;
	   return kFALSE;
         }
	 if (((TObjString*)tokens->At(1))->String().Atoi() <= 0) {
           AliError(Form("Invalid trigger cluster syntax (%s)!",strLine.Data()));
           return kFALSE;
         }
	 TString strTemp;
	 for(Int_t i = 2; i < ntokens; i++) {
	   strTemp += ((TObjString*)tokens->At(i))->String();
	   strTemp += " ";
	 }
	 AddCluster(((TObjString*)tokens->At(0))->String(),
			 ((TObjString*)tokens->At(1))->String().Atoi(),
			 strTemp);
       }
       break;
     case 5:
       {
	 AliTriggerPFProtection *pfp = NULL;
	 if ((((TObjString*)tokens->At(0))->String().CompareTo("NONE") == 0) ||
	     (((TObjString*)tokens->At(0))->String().CompareTo("NOPF") == 0)) {
	   pfp = new AliTriggerPFProtection(((TObjString*)tokens->At(0))->String());
	 }
	 else {
           if ((ntokens == 10) && (fVersion<=7)){ 
	    pfp = new AliTriggerPFProtection(((TObjString*)tokens->At(0))->String(),
					    ((TObjString*)tokens->At(1))->String(),
					    ((TObjString*)tokens->At(2))->String(),
					    ((TObjString*)tokens->At(3))->String());
	    pfp->SetNa1(((TObjString*)tokens->At(4))->String().Atoi());
	    pfp->SetNa2(((TObjString*)tokens->At(5))->String().Atoi());
	    pfp->SetNb1(((TObjString*)tokens->At(6))->String().Atoi());
	    pfp->SetNb2(((TObjString*)tokens->At(7))->String().Atoi());
	    pfp->SetTa(((TObjString*)tokens->At(8))->String().Atoi());
	    pfp->SetTb(((TObjString*)tokens->At(9))->String().Atoi());
	  }else if((ntokens == 13) && (fVersion<=7)){
	    UInt_t pfdef[12];
	    for(Int_t i=0;i<12;i++){
	       TString ss(((TObjString*)tokens->At(i+1))->String());
	       ss.Remove(0,2);
	       UInt_t num=0;
	       for(Int_t j=ss.Length()-1;j>=0;j--){
	        UInt_t nn=ss[j];
		if(nn >= (UInt_t)'0' && nn <= (UInt_t)'9')nn=nn-(UInt_t)'0'; else 
		if(nn >= (UInt_t)'A' && nn <= (UInt_t)'F')nn=10+nn-(UInt_t)'A'; else
		if(nn >= (UInt_t)'a' && nn <= (UInt_t)'f')nn=10+nn-(UInt_t)'a'; else{
  	     		AliError(Form("Invalid trigger pfs syntax (%s)!",strLine.Data()));
	     		//return kFALSE;
		}
		num=num+(1<<(ss.Length()-1-j)*4)*nn;
	        //cout << ss[j] << " 2 " << nn << " "  << num << endl;
	       }
	       pfdef[i]=num;
	    }   
	    pfp = new AliTriggerPFProtection(((TObjString*)tokens->At(0))->String(),pfdef);
	  }else if((ntokens==9) && (fVersion>7)){
	    // new LML0 PF
	    pfp = new AliTriggerPFProtection(
	    ((TObjString*)tokens->At(0))->String(),
	    ((TObjString*)tokens->At(1))->String(),
	    ((TObjString*)tokens->At(2))->String(),
	    ((TObjString*)tokens->At(3))->String().Atoi(),
	    ((TObjString*)tokens->At(4))->String().Atoi(),
	    ((TObjString*)tokens->At(5))->String().Atoi(),
	    ((TObjString*)tokens->At(6))->String().Atoi(),
	    ((TObjString*)tokens->At(7))->String().Atoi(),
	    ((TObjString*)tokens->At(8))->String().Atoi()
            );
	  }else{
  	     AliError(Form("Invalid trigger pfs syntax (%s)!",strLine.Data()));
	     //return kFALSE;
          }
	 }
	 AddPFProtection(pfp);
       }
       break;
     case 6:
         if (ntokens > 2) {
  	   AliError(Form("Invalid trigger bcmasks syntax (%s)!",strLine.Data()));
	   delete tokens;
	   return kFALSE;
         }
       if (((TObjString*)tokens->At(0))->String().CompareTo("NONE") == 0)
       {	 
         if(!AddMask(new AliTriggerBCMask(((TObjString*)tokens->At(0))->String()))) {
	   delete tokens;
	   return kFALSE;
	 }
       }
       else {
	 if(!AddMask(((TObjString*)tokens->At(0))->String(),((TObjString*)tokens->At(1))->String())) {
	   delete tokens;
	   return kFALSE;
	 }
       }
       break;
     case 7:
       {
         if ((ntokens !=8) && (ntokens != 10) && (ntokens != 11)) {
  	   AliError(Form("Invalid trigger class syntax (%s)!",strLine.Data()));
	   delete tokens;
	   return kFALSE;
         }
	 AliTriggerClass *trclass=0;
	 if(ntokens == 8)trclass = new AliTriggerClass(this,
			((TObjString*)tokens->At(0))->String(),((TObjString*)tokens->At(1))->String().Atoi(),
			((TObjString*)tokens->At(2))->String(),((TObjString*)tokens->At(3))->String(),
			((TObjString*)tokens->At(4))->String(),((TObjString*)tokens->At(5))->String(),
			((TObjString*)tokens->At(6))->String().Atoi(),(Bool_t)(((TObjString*)tokens->At(7))->String().Atoi()));
	 else{ trclass = new AliTriggerClass(this,
			((TObjString*)tokens->At(0))->String(),((TObjString*)tokens->At(1))->String().Atoi(),
			((TObjString*)tokens->At(2))->String(),((TObjString*)tokens->At(3))->String(),
			((TObjString*)tokens->At(4))->String(),
			((TObjString*)tokens->At(6))->String().Atoi(),(Bool_t)(((TObjString*)tokens->At(7))->String().Atoi()),
			(((TObjString*)tokens->At(8))->String().Atoi()),(((TObjString*)tokens->At(9))->String().Atoi()));
	   if(!trclass->SetMasks(this,((TObjString*)tokens->At(5))->String())) {
	     delete tokens;
	     return kFALSE;
	   }
	 }
         AddClass(trclass);
       }
     default:
       break;
     }
     delete tokens;

     return kTRUE;
}

//_____________________________________________________________________________
AliTriggerConfiguration* AliTriggerConfiguration::LoadConfiguration(TString & configuration)
{
   // Load one pre-created Configurations from database/file that match
   // with the input string 'configuration'
   // Ej: "Pb-Pb" or "p-p-DIMUON CALIBRATION-CENTRAL-BARREL"
  // By default the files are stored in GRP/CTP folder.
  // The filename is constructed as: GRP/CTP/<configuration>.cfg

   // Load the selected configuration
  TString filename;
  if (configuration.EndsWith(".cfg") ||
      configuration.EndsWith(".shuttle")) {
    filename = configuration;
  }
  else {
    filename = gSystem->Getenv("ALICE_ROOT");
    filename += "/GRP/CTP/";
    filename += configuration;
    filename += ".cfg";
  }

   if( gSystem->AccessPathName( filename.Data() ) ) {
      AliErrorClass( Form( "file (%s) not found", filename.Data() ) );
      return NULL;
   }


   ifstream *file = new ifstream ( filename.Data() );
   if (!*file) {
     AliErrorClass(Form("Error opening file (%s) !",filename.Data()));
     file->close();
     delete file;
     return NULL;
   }

   AliTriggerConfiguration *cfg = new AliTriggerConfiguration();

   Int_t level = 0;

   TString strLine;
   while (strLine.ReadLine(*file)) {
     if (cfg->ProcessConfigurationLine(strLine, level) == kFALSE)
     {
        delete cfg;
        cfg = 0;
        break;
     }
   }

   file->close();
   delete file;

   return cfg;
}

//_____________________________________________________________________________
AliTriggerConfiguration* AliTriggerConfiguration::LoadConfigurationFromString(const char* configuration)
{
   // Loads configuration given as parameter <configuration>

   if (!configuration)
     return 0;

   AliTriggerConfiguration *cfg = new AliTriggerConfiguration();

   Int_t level = 0;

   TObjArray* tokens = TString(configuration).Tokenize("\n");
   for (Int_t i=0; i<tokens->GetEntries(); i++)
   {
     TObjString* str = dynamic_cast<TObjString*>(tokens->At(i));
     if (!str)
       continue;

     if (cfg->ProcessConfigurationLine(str->String(), level) == kFALSE)
     {
        delete cfg;
        cfg = 0;
        break;
     }
   }

   delete tokens;

   return cfg;
}

//_____________________________________________________________________________
TObjArray* AliTriggerConfiguration::GetAvailableConfigurations( const char* filename )
{
   // Return an array of configuration in the file

   TString path;
   if( !filename[0] ) {
      path += gSystem->Getenv( "ALICE_ROOT" );
      path += fgkConfigurationFileName;
   }
   else
      path += filename;

   if( gSystem->AccessPathName( path.Data() ) ) {
      AliErrorGeneral( "AliTriggerConfiguration", Form( "file (%s) not found", path.Data() ) );
      return NULL;
   }

   TObjArray* desArray = new TObjArray();

   TFile file( path.Data(), "READ" );
   if( file.IsZombie() ) {
      AliErrorGeneral( "AliTriggerConfiguration", Form( "Error opening file (%s)", path.Data() ) );
      return NULL;
   }

   file.ReadAll();

   TKey* key;
   TIter next( file.GetListOfKeys() );
   while( (key = (TKey*)next()) ) {
      TObject* obj = key->ReadObj();
      if( obj->InheritsFrom( "AliTriggerConfiguration" ) ) {
         desArray->AddLast( obj );
      }
   }
   file.Close();

   return desArray;
}

//_____________________________________________________________________________
void AliTriggerConfiguration::WriteConfiguration( const char* filename )
{
   // Write the configuration
   TString path;
   if( !filename[0] ) {
      path += gSystem->Getenv("ALICE_ROOT");
      path += fgkConfigurationFileName;
   }
   else
      path += filename;

   TFile file( path.Data(), "UPDATE" );
   if( file.IsZombie() ) {
      AliErrorGeneral( "AliTriggerConfiguration", 
                        Form( "Can't open file (%s)", path.Data() ) );
      return;
   }

   Bool_t result = (Write( GetName(), TObject::kOverwrite ) != 0);
   if( !result )
      AliErrorGeneral( "AliTriggerConfiguration",
                        Form( "Can't write entry to file <%s>!", path.Data() ) );
   file.Close();
}

//_____________________________________________________________________________
Int_t AliTriggerConfiguration::GetClassIndexFromName(const char* className) const
{
   //const TObjArray& classes = cfg->GetClasses();
   Int_t nclasses = (Int_t)fClasses.GetEntriesFast();
   for (Int_t i=0;i<nclasses;i++) {
       AliTriggerClass* trgclass = (AliTriggerClass*)fClasses.At(i);
       if (TString(trgclass->GetName()).CompareTo(className) == 0) { 
          return trgclass->GetIndex();
       }
   }
   return -1;
}
//_____________________________________________________________________________
const char* AliTriggerConfiguration::GetClassNameFromIndex(Int_t classIndex) const
{
   Int_t nclasses = (Int_t)fClasses.GetEntriesFast();
   for (Int_t i=0;i<nclasses;i++) {
       AliTriggerClass* trgclass = (AliTriggerClass*)fClasses.At(i);
       if (trgclass->GetIndex() == classIndex) return trgclass->GetName();
   }
   return 0;
}
//_____________________________________________________________________________
AliTriggerClass* AliTriggerConfiguration::GetTriggerClass(Int_t classIndex) const
{
   Int_t nclasses = (Int_t)fClasses.GetEntriesFast();
   for (Int_t i=0;i<nclasses;i++) {
       AliTriggerClass* trgclass = (AliTriggerClass*)fClasses.At(i);
       if ( trgclass->GetIndex() == classIndex) return trgclass;
   }
   return 0;
}
//_____________________________________________________________________________
void AliTriggerConfiguration::Reset()
{
   for( Int_t j=0; j<fInputs.GetEntriesFast(); j++ )
     ((AliTriggerInput*)fInputs.At(j))->Reset();

   for( Int_t j=0; j<fClasses.GetEntriesFast(); j++ )
     ((AliTriggerClass*)fClasses.At(j))->Reset();
}

//_____________________________________________________________________________
void AliTriggerConfiguration::Print( const Option_t*  ) const
{
   // Print
  cout << "#################################################" << endl;
   cout << "Trigger Configuration:"  << endl;
   cout << "  Name:              " << GetName() << endl; 
   cout << "  Description:       " << GetTitle() << endl;
   cout << "  Version:           " << GetVersion() << endl;
   cout << "  Active Detectors:  " << GetActiveDetectors() << endl;
   cout << "  Trigger Detectors: " << GetTriggeringDetectors() << endl;

   cout << "#################################################" << endl;
   fInputs.Print();
   cout << "#################################################" << endl;
   fInteractions.Print();
   cout << "#################################################" << endl;
   fFunctions.Print();
   cout << "#################################################" << endl;
   fDescriptors.Print();
   cout << "#################################################" << endl;
   fClusters.Print();
   cout << "#################################################" << endl;
   fPFProtections.Print();
   cout << "#################################################" << endl;
   fMasks.Print();
   cout << "#################################################" << endl;
   fClasses.Print();
   cout << "#################################################" << endl;

   cout << endl;
}


//////////////////////////////////////////////////////////////////////////////
// Helper method

//_____________________________________________________________________________
Bool_t AliTriggerConfiguration::IsSelected( TString detName, TString& detectors ) const
{
   // check whether detName is contained in detectors
   // if yes, it is removed from detectors

   // check if all detectors are selected
   if( (detectors.CompareTo("ALL") == 0 ) ||
        detectors.BeginsWith("ALL ") ||
        detectors.EndsWith(" ALL") ||
        detectors.Contains(" ALL ") ) {
      detectors = "ALL";
      return kTRUE;
   }

   // search for the given detector
   Bool_t result = kFALSE;
   if( (detectors.CompareTo( detName ) == 0) ||
        detectors.BeginsWith( detName+" " ) ||
        detectors.EndsWith( " "+detName ) ||
        detectors.Contains( " "+detName+" " ) ) {
      detectors.ReplaceAll( detName, "" );
      result = kTRUE;
   }

   // clean up the detectors string
   while( detectors.Contains("  ") )  detectors.ReplaceAll( "  ", " " );
   while( detectors.BeginsWith(" ") ) detectors.Remove( 0, 1 );
   while( detectors.EndsWith(" ") )   detectors.Remove( detectors.Length()-1, 1 );

   return result;
}
