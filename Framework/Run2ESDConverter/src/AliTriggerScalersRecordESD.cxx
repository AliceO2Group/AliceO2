/*************************************************************************
* Copyright(c) 1998-2008, ALICE Experiment at CERN, All rights reserved. *
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


#include <Riostream.h>
#include <TObjArray.h>
#include <AliTimeStamp.h>
#include "AliLog.h"  
#include "AliTriggerScalersESD.h"
#include "AliTriggerScalersRecordESD.h"

using std::endl;
using std::cout;
ClassImp( AliTriggerScalersRecordESD )
//_____________________________________________________________________________
AliTriggerScalersRecordESD::AliTriggerScalersRecordESD():
TObject(),
fTimestamp(),
fScalers(),
fTimeGroup(0)
{
}

//_____________________________________________________________________________
void AliTriggerScalersRecordESD::AddTriggerScalers( AliTriggerScalersESD* scaler ) 
{ 
  fScalers.AddLast( scaler ); 
}

//_____________________________________________________________________________
void AliTriggerScalersRecordESD::AddTriggerScalers( UChar_t classIndex, ULong64_t LOCB, ULong64_t LOCA,        
                                         ULong64_t L1CB, ULong64_t L1CA, ULong64_t L2CB, ULong64_t L2CA )
{
    AddTriggerScalers( new AliTriggerScalersESD( classIndex, LOCB, LOCA, L1CB, L1CA, L2CB, L2CA ) );
} 

//_____________________________________________________________________________
AliTriggerScalersRecordESD::AliTriggerScalersRecordESD( const AliTriggerScalersRecordESD &rec ) :
TObject(rec),
fTimestamp(rec.fTimestamp),
fScalers(),
fTimeGroup(rec.fTimeGroup)
{
//copy constructor
for (Int_t i = 0; i < rec.fScalers.GetEntriesFast(); i++) {
    if (rec.fScalers[i]) fScalers.Add(rec.fScalers[i]->Clone());
  }
}
//_____________________________________________________________________________
AliTriggerScalersRecordESD& AliTriggerScalersRecordESD:: operator=(const AliTriggerScalersRecordESD& rec)
{
//asignment operator
if(&rec == this) return *this;
((TObject *)this)->operator=(rec);
fTimestamp=rec.fTimestamp;
fScalers.Delete();
for (Int_t i = 0; i < rec.fScalers.GetEntriesFast(); i++) {
    if (rec.fScalers[i]) fScalers.Add(rec.fScalers[i]->Clone());
  }
fTimeGroup=rec.fTimeGroup;
return *this;
} 

//_____________________________________________________________________________
void AliTriggerScalersRecordESD::Reset()
{
fScalers.SetOwner();
fScalers.Clear();
fTimeGroup=0;
} 

//_____________________________________________________________________________
const AliTriggerScalersESD* AliTriggerScalersRecordESD::GetTriggerScalersForClass( const Int_t classindex ) const
{
   // Find Trigger scaler with class ID = classmask using a binary search. 

   Int_t   base, last;
   AliTriggerScalersESD *op2 = NULL;
   
   base = 0;
   last = fScalers.GetEntriesFast();

   while (base < last) {
      op2 = (AliTriggerScalersESD *)fScalers.At(base);
      if( op2->GetClassIndex()  == classindex ) return op2;
      base++;
   }
   return op2;   
}
                                      
//_____________________________________________________________________________
void AliTriggerScalersRecordESD::Print( const Option_t* ) const
{
   // Print
  cout << "Trigger Scalers Record, time group: "<< fTimeGroup << endl;
  fTimestamp.Print();
  for( Int_t i=0; i<fScalers.GetEntriesFast(); ++i ) 
     ((AliTriggerScalersESD*)fScalers.At(i))->Print();
}
