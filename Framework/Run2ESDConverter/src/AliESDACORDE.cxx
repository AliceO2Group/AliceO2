

// Last update: October 2th 2009 

#include "AliESDACORDE.h"

ClassImp(AliESDACORDE)

AliESDACORDE::AliESDACORDE():TObject()
{
 //Default constructor
	for(Int_t i=0;i<60;i++)
	{
		fACORDEBitPattern[i] = 0;
	}
}


AliESDACORDE::AliESDACORDE(const AliESDACORDE &o)
  :TObject(o)

{	
	//Default constructor
	for(Int_t i=0;i<60;i++)
	{
		fACORDEBitPattern[i] = o.fACORDEBitPattern[i];
	}
}


AliESDACORDE::AliESDACORDE(Bool_t* MACORDEBitPattern):TObject()
{

	//Constructor

	for(Int_t i=0;i<60;i++)
	{
		fACORDEBitPattern[i] = MACORDEBitPattern[i];
	}
}

AliESDACORDE& AliESDACORDE::operator=(const AliESDACORDE& o)
{
// Copy Constructor
	if(this==&o)return *this;
	TObject::operator=(o);

	// Assignment operator
	for(Int_t i=0; i<60; i++)
	{
		fACORDEBitPattern[i] = o.fACORDEBitPattern[i];
	}
	
	return *this;
}


Bool_t AliESDACORDE::GetHitChannel(Int_t i) const
{
	return fACORDEBitPattern[i];
}

void AliESDACORDE::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDACORDE *robj = dynamic_cast<AliESDACORDE*>(&obj);
  if(!robj)return; // not an AliESDACRDE
  *robj = *this;

}


