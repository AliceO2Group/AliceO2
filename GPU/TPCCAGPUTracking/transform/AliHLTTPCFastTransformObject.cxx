//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Sergey Gorbunov <sergey.gorbunov@cern.ch>             *
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

/** @file   AliHLTTPCFastTransformObject.cxx
    @author Sergey Gorbubnov
    @date   
    @brief 
*/


#include "AliHLTTPCFastTransformObject.h"
#include "TCollection.h"
#include "TIterator.h"
 
ClassImp(AliHLTTPCFastTransformObject); //ROOT macro for the implementation of ROOT specific class methods


AliHLTTPCFastTransformObject::AliHLTTPCFastTransformObject()
  :
  TObject(),
  fVersion(0),
  fLastTimeBin(0),
  fTimeSplit1(0),
  fTimeSplit2(0),
  fAlignment(0),
  fReverseTransformInfo()
{
  // constructor
  for (int i = 0;i < fkNSec;i++) fSectorInit[i] = false;
  
  Reset();
}



void  AliHLTTPCFastTransformObject::Reset()
{
  // Deinitialisation
  fLastTimeBin = 0.;
  fTimeSplit1 = 0.;
  fTimeSplit2 = 0.;  
  for( Int_t i=0; i<fkNSplinesIn + fkNSplinesOut; i++) fSplines[i].Reset();
  fAlignment.Set(0);
}

AliHLTTPCFastTransformObject::AliHLTTPCFastTransformObject( const AliHLTTPCFastTransformObject &o )
  :
  TObject(o),
  fVersion(0),
  fLastTimeBin(o.fLastTimeBin),
  fTimeSplit1(o.fTimeSplit1),
  fTimeSplit2(o.fTimeSplit2),
  fAlignment(o.fAlignment),
  fReverseTransformInfo()
{ 
  // constructor    
  for( Int_t i=0; i<fkNSplinesIn + fkNSplinesOut; i++){
    fSplines[i] = o.fSplines[i];
  }
}

AliHLTTPCFastTransformObject& AliHLTTPCFastTransformObject::operator=( const AliHLTTPCFastTransformObject &o)
{
  // assignment operator
   new (this) AliHLTTPCFastTransformObject( o );
   return *this;
}

void AliHLTTPCFastTransformObject::Merge(const AliHLTTPCFastTransformObject& obj)
{
  for (int i = 0;i < fkNSecIn;i++)
    {
      if (!obj.IsSectorInit(i)) continue;
      for(int iRow=0;iRow<fkNRowsIn;iRow++)
	{
	  for(int iSpline=0;iSpline<3;iSpline++)
	    {
	      GetSplineInNonConst(i, iRow, iSpline) = obj.GetSplineIn(i, iRow, iSpline);
	    }
	}
      fSectorInit[i] = true;
    }

  for (int i = 0;i < fkNSecOut;i++)
    {
      if (!obj.IsSectorInit(fkNSecIn+i)) continue;
      for(int iRow=0;iRow<fkNRowsOut;iRow++)
	{
	  for(int iSpline=0;iSpline<3;iSpline++)
	    {
	      GetSplineOutNonConst(i, iRow, iSpline) = obj.GetSplineOut(i, iRow, iSpline);
	    }
	}
      fSectorInit[fkNSecIn+i] = true;
    }
}

Long64_t AliHLTTPCFastTransformObject::Merge(TCollection* list)
{
	if (list == NULL || list->GetSize() == 0) return(0); //Nothing to do!
	TIterator* iter = list->MakeIterator();
	TObject* obj;
	int nMerged = 0;
	while (obj = iter->Next())
	{
		AliHLTTPCFastTransformObject* mergeObj = dynamic_cast<AliHLTTPCFastTransformObject*>(obj);
		if (mergeObj && mergeObj != this)
		{
			Merge(*mergeObj);
			nMerged++;
		}
	}
	return(nMerged);
}
