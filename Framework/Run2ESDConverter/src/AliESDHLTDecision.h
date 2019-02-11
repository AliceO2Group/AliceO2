//-*- Mode: C++ -*-
// $Id$
#ifndef ALIESDHLTDECISION_H
#define ALIESDHLTDECISION_H
/* This file is property of and copyright by the ALICE HLT Project        *
 * ALICE Experiment at CERN, All rights reserved.                         *
 * See cxx source for full Copyright notice                               */

/// @file   AliESDHLTDecision.h
/// @author matthias.richter@ift.uib.no
/// @date   23 Nov 2009
/// @brief  Container for HLT decision within the ESD

#include "TNamed.h"
#include "TArrayI.h"
#include "TArrayL64.h"
#include "TClonesArray.h"

class AliESDHLTDecision : public TNamed
{
 public:
  /// constructor
  AliESDHLTDecision();
  /// constructor
  AliESDHLTDecision(bool result, const char* description);
  /// copy constructor
  AliESDHLTDecision(const AliESDHLTDecision& src);
  /// assignment operator
  AliESDHLTDecision& operator=(const AliESDHLTDecision& src);
  /// destructor
  ~AliESDHLTDecision();

  /// set trigger result
  void SetTriggerResult(bool result) {SetBit(kTriggerResult, result);}

  /// set the active HLT trigger items
  int SetTriggerItems(TArrayI items) {fTriggerItems=items; return 0;}

  /// set the event counters for HLT trigger items
  int SetCounters(TArrayL64 counters) {fCounters=counters; return 0;}

  /// get the description og the global trigger decision
  const char* GetDescription() const;

  /// trigger result
  bool Result() const { return TestBit(kTriggerResult) == 1; }

  /// Inherited from TObject. Return the result of the trigger.
  /// @return   "0" or "1" (note: its a string)
  virtual Option_t *GetOption() const {return Result()?"1":"0";}

  /// check whether the HLT global trigger has fired, or
  /// for a specific HLT trigger class if specified
  Bool_t    IsTriggerFired(const char* name=NULL) const;

  /// Inherited from TObject. Print Information.
  virtual void Print(Option_t* option = "") const;

  /// Inherited from TObject. Copy this to the specified object.
  virtual void Copy(TObject &object) const;
  
  /// Inherited from TObject. Create a new clone.
  virtual TObject *Clone(const char *newname="") const;

  enum {
    /// trigger result true/false
    kTriggerResult =       BIT(15),
    /// indicate trigger decision object
    kTriggerDecision =     BIT(16)
  };

  static const char* Name() {return fgkName;}

 protected:
 private:
  /// short info about the input objects stored in TNamed
  /// trigger name, description (title), 
  /// type: TObject bit 15 set for trigger decisions
  /// decision: TObject bit 16 indicates true/false
  TClonesArray fInputObjectInfo; // TNamed objects

  /// array of fired trigger item ids from the HLT trigger menu
  TArrayI fTriggerItems; // list of fired trigger items

  /// trigger counters for HLT trigger items
  TArrayL64 fCounters; // event counters for trigger items

  /// object name for the HLT decision
  static const char* fgkName; // the name of the HLT decision

  ClassDef(AliESDHLTDecision, 1)
};
#endif //ALIESDHLTDECISION_H
