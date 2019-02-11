#ifndef ALITRIGGERCLASS_H
#define ALITRIGGERCLASS_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// This class represents the CTP class objects                               //
//                                                                           //
// The Class consists of Name, index in the trigger mask counted from 1,     //
// descriptor, cluster,past-future, mask, downscale, allrare,                //
// time group, time window                                                   //
//                                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TNamed.h>

class AliTriggerConfiguration;
class AliTriggerDescriptor;
class AliTriggerCluster;
class AliTriggerPFProtection;
class AliTriggerBCMask;

class AliTriggerClass : public TNamed {

public:
                          AliTriggerClass();
                          AliTriggerClass( TString & name, UChar_t index,
					   AliTriggerDescriptor *desc, AliTriggerCluster *clus,
					   AliTriggerPFProtection *pfp, AliTriggerBCMask *mask,
					   UInt_t prescaler, Bool_t allrare);
                          AliTriggerClass( AliTriggerConfiguration *config,
					   TString & name, UChar_t index,
					   TString &desc, TString &clus,
					   TString &pfp, TString &mask,
					   UInt_t prescaler, Bool_t allrare);
                          AliTriggerClass( AliTriggerConfiguration *config,
					   TString & name, UChar_t index,
					   TString &desc, TString &clus,
					   TString &pfp,
					   UInt_t prescaler, Bool_t allrare,
					   UInt_t timegroup, UInt_t timewindow);

                          AliTriggerClass( const AliTriggerClass& trclass );
               virtual   ~AliTriggerClass();
  AliTriggerClass&   operator=(const AliTriggerClass& trclass);

                  void    Reset() { fStatus = kFALSE; }

             ULong64_t    GetValue() const { return (fStatus) ? fClassMask : 0; }
             ULong64_t    GetValueNext50() const { return (fStatus) ? fClassMaskNext50 : 0; }
                Bool_t    GetStatus() const { return fStatus; }
               ULong64_t  GetMask() const { return fClassMask; }
               ULong64_t  GetMaskNext50() const { return fClassMaskNext50; }
	       	   Int_t  GetIndex() const {return (Int_t)fIndex;}
    AliTriggerDescriptor* GetDescriptor() const { return fDescriptor; }
       AliTriggerCluster* GetCluster() const { return fCluster; }
        AliTriggerBCMask* GetBCMask() const { return fMask[0]; }
	           UInt_t GetTimeGroup() const { return fTimeGroup; }
	           UInt_t GetTimeWindow() const { return fTimeGroup; }
		   UInt_t GetPrescaler() const { return fPrescaler; }
		   Int_t GetDownscaleFactor(Double_t &ds) const;

		   Bool_t SetMasks(AliTriggerConfiguration *config,TString &mask);
                    void  Trigger( const TObjArray& inputs , const TObjArray& functions);
		    void  Print( const Option_t* ) const;

                  Bool_t  CheckClass(AliTriggerConfiguration *config) const;
		  Bool_t  IsActive( const TObjArray& inputs, const TObjArray& functions) const;
		  enum {kNMaxMasks = 13};  // CTP handles up to 12 different BC masks + NONE

private:
	       ULong64_t  fClassMask;    // trigger mask (1<< (index-1))
	       ULong64_t  fClassMaskNext50; // trigger mask (1<< (index-1))
	       	 UChar_t  fIndex;        // position of class in mask
    AliTriggerDescriptor* fDescriptor;   // pointer to the descriptor
       AliTriggerCluster* fCluster;      // pointer to the cluster
  AliTriggerPFProtection* fPFProtection; // pointer to the past-future protection
        AliTriggerBCMask* fMask[kNMaxMasks];         // array of pinters pointer to bunch-crossing mask
                  UInt_t  fPrescaler;    // Downscaling factor
                  Bool_t  fAllRare;      // All or Rare trigger
		  Bool_t  fStatus;       //! true = Condition has been satisfied after Trigger
		  UInt_t  fTimeGroup;    // time group
		  UInt_t  fTimeWindow;   // the size of time window for its group

  ClassDef( AliTriggerClass, 6 )  // Define a trigger class object
};

#endif
