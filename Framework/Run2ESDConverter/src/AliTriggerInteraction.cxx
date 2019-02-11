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
// This class which defines the trigger interaction objects
//
//
///////////////////////////////////////////////////////////////////////////////

#include <Riostream.h>
#include <TObjArray.h>
#include <TObjString.h>

#include "AliTriggerInteraction.h"
#include "AliTriggerInput.h"
#include "AliExpression.h"
#include "AliLog.h"

using std::endl;
using std::cout;
ClassImp(AliTriggerInteraction)

//_____________________________________________________________________________
AliTriggerInteraction::AliTriggerInteraction():
  TNamed()
{
  // Default constructor
}

//_____________________________________________________________________________
AliTriggerInteraction::AliTriggerInteraction( TString & name, TString &logic ):
  TNamed( name, logic )
{
  // Constructor
}
//_____________________________________________________________________________
AliTriggerInteraction::~AliTriggerInteraction() 
{ 
  // Destructor
}
//_____________________________________________________________________________
AliTriggerInteraction::AliTriggerInteraction( const AliTriggerInteraction& interact ):
  TNamed( interact )
{
   // Copy constructor
}

//______________________________________________________________________________
AliTriggerInteraction& AliTriggerInteraction::operator=(const AliTriggerInteraction& interact)
{
   // AliTriggerInteraction assignment operator.

   if (this != &interact) {
      TNamed::operator=(interact);
   }
   return *this;
}

//_____________________________________________________________________________
Bool_t AliTriggerInteraction::CheckInputs(const TObjArray &inputs) const
{
  // Check the existance of trigger inputs
  // and the logic used.
  // Return false in case of wrong interaction
  // definition.

  TString logic( GetTitle() );
  TObjArray* tokens = logic.Tokenize(" !&|()\t");

  Int_t ntokens = tokens->GetEntriesFast();
  for( Int_t i=0; i<ntokens; i++ ) {
    TObjString* iname = (TObjString*)tokens->At( i );

    AliTriggerInput *inp = (AliTriggerInput*)inputs.FindObject(iname->String().Data());
    if (!inp) {
      AliError( Form( "The trigger input (%s) is not available for Interaction (%s)",
		      iname->String().Data(), GetName() ) );
      delete tokens;
      return kFALSE;
    }
    if (inp->GetMask() == 0 || inp->GetMask() > (1<<24)) { // New l0f can use all inputs
      AliError( Form( "The trigger input (%s) is not among the first 4 trigger inputs used to create interactions. Interaction (%s) is invalid",
		      iname->String().Data(), GetName() ) ); 
      delete tokens;
      return kFALSE;
    }
  }

  delete tokens;
  return kTRUE;
}

//_____________________________________________________________________________
Bool_t AliTriggerInteraction::IsActive(const TObjArray &inputs) const
{
  // Check if the trigger inputs
  // are active
  // Return false in one or more inputs
  // are disabled

  TString logic( GetTitle() );
  TObjArray* tokens = logic.Tokenize(" !&|()\t");

  Int_t ntokens = tokens->GetEntriesFast();
  for( Int_t i=0; i<ntokens; i++ ) {
    TObjString* iname = (TObjString*)tokens->At( i );

    AliTriggerInput *inp = (AliTriggerInput *)inputs.FindObject(iname->String());
    if (!inp) {
      AliError( Form( "The trigger input (%s) is not available for Interaction (%s)",
		      iname->String().Data(), GetName() ) );
      delete tokens;
      return kFALSE;
    }
    else {
      if (!inp->IsActive()) {
	AliWarning(Form("The interaction/function (%s) will be disabled, because the input (%s) is disabled",
			GetName(),iname->String().Data()));
	delete tokens;
	return kFALSE;
      }
    }
  }

  delete tokens;
  return kTRUE;
}

//_____________________________________________________________________________
Bool_t AliTriggerInteraction::Trigger(const TObjArray& inputs ) const
{
  // Check if the inputs satify the interaction expression condition 
  AliExpression* exp = new AliExpression( GetTitle() );
  Bool_t status = exp->Value( inputs );
  delete exp;
  return status;
}

//_____________________________________________________________________________
void AliTriggerInteraction::Print( const Option_t* ) const
{
   // Print
  cout << "Trigger Interaction:" << endl;
  cout << "  Name:              " << GetName() << endl;
  cout << "  Logic:             " << GetTitle() << endl;
}
