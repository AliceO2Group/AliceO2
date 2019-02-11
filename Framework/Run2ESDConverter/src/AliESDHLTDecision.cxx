// $Id$
//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Matthias.Richter@ift.uib.no                           *
//*                  for The ALICE HLT Project.                            *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

/// @file   AliESDHLTDecision.cxx
/// @author matthias.richter@ift.uib.no
/// @date   23 Nov 2009
/// @brief  Container for HLT decision within the ESD
///
/// A container for the HLT trigger decision stored in the ESD.
/// The HLT trigger decision is evaluated by the HLTGlobalTrigger component
/// according to different trigger inputs and the HLT trigger menu.
 
#include "AliESDHLTDecision.h"
#include <iostream>

using std::endl;
using std::cout;
ClassImp(AliESDHLTDecision)

AliESDHLTDecision::AliESDHLTDecision()
  : TNamed(fgkName, "")
  , fInputObjectInfo(TNamed::Class())
  , fTriggerItems()
  , fCounters()
{
  /// constructor
}

const char* AliESDHLTDecision::fgkName="HLTGlobalTrigger";

AliESDHLTDecision::AliESDHLTDecision(bool result, const char* description)
  : TNamed(fgkName, description)
  , fInputObjectInfo(TNamed::Class())
  , fTriggerItems()
  , fCounters()
{
  /// constructor
  SetBit(kTriggerResult, result);
}

AliESDHLTDecision::AliESDHLTDecision(const AliESDHLTDecision& src)
  : TNamed(src)
  , fInputObjectInfo(src.fInputObjectInfo)
  , fTriggerItems(src.fTriggerItems)
  , fCounters(src.fCounters)
{
  /// copy constructor, performs a deep copy
}

AliESDHLTDecision& AliESDHLTDecision::operator=(const AliESDHLTDecision& src)
{
  /// assignment operator
  TNamed::operator=(src);

  fInputObjectInfo=src.fInputObjectInfo;
  fTriggerItems=src.fTriggerItems;
  fCounters=src.fCounters;

  return *this;
}

AliESDHLTDecision::~AliESDHLTDecision()
{
  /// destructor
  fInputObjectInfo.Delete();
}

const char* AliESDHLTDecision::GetDescription() const
{
  /// get the description of the global trigger decision
  return GetTitle();
}

Bool_t    AliESDHLTDecision::IsTriggerFired(const char* name) const
{
  /// check whether the HLT global trigger has fired, or
  /// for a specific HLT trigger class if specified

  // TODO: the complete functionality must be implemented
  // The HLT global trigger evaluates the trigger decision
  // according to the trigger input and the trigger menu. It
  // supports priority groups, allowing items to take precedence
  // over others. The simplest scheme is an 'OR' of all items.
  // This is implemented here, and the full and correct handling
  // needs to be implemented.
  Option_t* option=this->GetOption();
  if (option==NULL || *option!='1') return kFALSE;

  if (name) {
    TString description=GetDescription();
    Int_t index=description.Index(name);
    if (index<0) return kFALSE;
    index+=strlen(name);
    if (index>=description.Length()) return kFALSE;
    if (description[index]!=0 && description[index]!=' ') return kFALSE;
  }
  return kTRUE;
}

void AliESDHLTDecision::Print(Option_t* option ) const
{
  /// Inherited from TObject. Print Information.
  TString opt(option);
  if (opt.Contains("compact"))
  {
    cout << "Global Trigger " << GetName() << ": result = " << TestBit(kTriggerResult) << endl;
    cout << "    Description = \"" << GetDescription() << "\"" << endl;
  }
  else if (opt.Contains("short"))
  {
    cout << "Global Trigger " << GetName() << ": result = " << TestBit(kTriggerResult) << endl;
    cout << "    Description = \"" << GetDescription() << "\"" << endl;
    cout << "#################### Input trigger decisions ####################" << endl;
    TIter next(&fInputObjectInfo);
    TObject* object=NULL;
    int count=0;
    while ((object=next())) {
      if (object->TestBit(kTriggerDecision)) {
	count++;
	cout << "Trigger " << object->GetName() << ": result = " << object->TestBit(kTriggerResult) << endl;
	cout << "    Description = \"" << object->GetTitle() << "\"" << endl;
      }
    }
    if (count==0) {
      cout << "(none)" << endl;
    }
  }
  else if (opt.Contains("counters"))
  {
    cout << "Counter\tValue" << endl;
    for (Int_t i = 0; i < fCounters.GetSize(); i++)
    {
      cout << i << "\t" << fCounters[i] << endl;
    }
    if (fCounters.GetSize() == 0)
    {
      cout << "(none)" << endl;
    }
  }
  else
  {
    TObject* object=NULL;
    cout << "Global Trigger " << GetName() << ": result = " << TestBit(kTriggerResult) << endl;
    cout << "    Description = \"" << GetDescription() << "\"" << endl;
    cout << "#################### Input trigger decisions ####################" << endl;
    TIter next(&fInputObjectInfo);
    int count=0;
    while ((object=next())) {
      if (object->TestBit(kTriggerDecision)) {
	count++;
	cout << "Trigger " << object->GetName() << ": result = " << object->TestBit(kTriggerResult) << endl;
	cout << "    Description = \"" << object->GetTitle() << "\"" << endl;
      }
    }
    if (count==0) {
      cout << "(none)" << endl;
    }
    cout << "###################### Other input objects ######################" << endl;
    count=0;
    next.Reset();
    while ((object=next())) {
      if (!object->TestBit(kTriggerDecision)) {
	cout << "------------------------ Input object " << count << " ------------------------" << endl;
	object->Print(option);
	count++;
      }
    }
    if (count==0) {
      cout << "(none)" << endl;
    }
    cout << "#################### Event class counters ####################" << endl;
    cout << "Counter\tValue" << endl;
    for (Int_t i = 0; i < fCounters.GetSize(); i++)
    {
      cout << i << "\t" << fCounters[i] << endl;
    }
    if (fCounters.GetSize() == 0)
    {
      cout << "(none)" << endl;
    }
  }
}

void AliESDHLTDecision::Copy(TObject &object) const
{
  /// Inherited from TObject. Copy this to the specified object.
  if (object.IsA() != IsA()) return;

  AliESDHLTDecision& target=dynamic_cast<AliESDHLTDecision&>(object);
  target=*this;
}

TObject *AliESDHLTDecision::Clone(const char */*newname*/) const
{
  /// Inherited from TObject. Create a new clone.
  return new AliESDHLTDecision(*this);
}
