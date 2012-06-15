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

/** @file   AliHLTTPCFastTransform.cxx
    @author Sergey Gorbubnov
    @date   
    @brief 
*/


#include "AliHLTTPCFastTransform.h"
#include "AliTPCTransform.h"
#include "AliTPCParam.h"
#include "AliTPCcalibDB.h"
 
#include <iostream>
#include <iomanip>

using namespace std;

ClassImp(AliHLTTPCFastTransform); //ROOT macro for the implementation of ROOT specific class methods

AliHLTTPCFastTransform* AliHLTTPCFastTransform::fgInstance = 0;


void AliHLTTPCFastTransform::Terminate()
{
  //
  // Singleton implementation
  // Deletes the instance of this class and sets the terminated flag, instances cannot be requested anymore
  // This function can be called several times.
  //
  
  if( fgInstance ){
    delete fgInstance;
    fgInstance = 0;
  }
}

AliHLTTPCFastTransform::AliHLTTPCFastTransform()
:
  fOrigTransform(0),
  fLastTimeStamp(-1),
  fLastTimeBin(600),
  fTimeBorder1(100),
  fTimeBorder2(500)
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt    
  for( Int_t i=0; i<72; i++)
    for( Int_t j=0; j<100; j++ ) fRows[i][j] = 0;
  Init();
}

AliHLTTPCFastTransform::~AliHLTTPCFastTransform() 
{ 
  // see header file for class documentation
  for( Int_t i=0; i<72; i++)
    for( Int_t j=0; j<100; j++ ) delete fRows[i][j]; 

  if( fgInstance == this ) fgInstance = 0;
}


Int_t  AliHLTTPCFastTransform::Init( AliTPCTransform *transform, Int_t TimeStamp )
{
  // Initialisation

  if( !transform ){
    AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();  
    if(!pCalib ) return 1;
    transform = pCalib->GetTransform();
  }

  if( fOrigTransform != transform ){
    fOrigTransform = transform;
    fLastTimeStamp = -1;
  }

  SetCurrentTimeStamp( TimeStamp );
  return 0;
}


void AliHLTTPCFastTransform::SetCurrentTimeStamp( Int_t TimeStamp )
{
  // Set the current time stamp
  if( fLastTimeStamp>=0 && TMath::Abs(fLastTimeStamp - TimeStamp ) <60 ) return;

  if( !fOrigTransform ) return;
  
  if( TimeStamp>=0 ){
    fOrigTransform->SetCurrentTimeStamp( TimeStamp );
    fLastTimeStamp = TimeStamp;
  } else fLastTimeStamp = fOrigTransform->GetCurrentTimeStamp();


  AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();  
  if(!pCalib ) return ;   

  AliTPCParam *par = pCalib->GetParameters(); 
  if( !par ) return ;

  // find last calibrated time bin
  
  Int_t nTimeBins = par->GetMaxTBin();
  Int_t is[]={0};
  bool sign = 0;
  for( fLastTimeBin=0; fLastTimeBin<nTimeBins; fLastTimeBin++){  
    Double_t xx[]={0,0,fLastTimeBin};
    fOrigTransform->Transform(xx,is,0,1);
    bool s = (xx[2]>=0);
    if( fLastTimeBin==0 ) sign = s;
    else if( sign!=s ){
      fLastTimeBin--;
      break;    
    }
  }
  fTimeBorder1 = 60;
  fTimeBorder2 = fLastTimeBin - 100;

  for( Int_t i=0; i<72; i++ )
    for( Int_t j=0; j<100; j++ ) if( fRows[i][j] ) InitRow(i,j);
}

Int_t AliHLTTPCFastTransform::InitRow( Int_t iSector, Int_t iRow )
{
  // see header file for class documentation
  
  if( iSector<0 || iSector>=72 || iRow<0 || iRow>=100 || !fOrigTransform) return 1;

  AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();  
  if(!pCalib ) return 1;   

  AliTPCParam *par = pCalib->GetParameters(); 
  if( !par ) return 1;

  Int_t nPads = par->GetNPads(iSector,iRow);

  if( !fRows[iSector][iRow] ){
    fRows[iSector][iRow] = new AliHLTTPCFastTransform::AliRowTransform;
  }

  fRows[iSector][iRow]->fSpline[0].Init(         0.5, nPads-1+0.5, 15, 0,  fTimeBorder1,         5);
  fRows[iSector][iRow]->fSpline[1].Init(         0.5, nPads-1+0.5, 15, fTimeBorder1, fTimeBorder2,         10);
  fRows[iSector][iRow]->fSpline[2].Init(         0.5, nPads-1+0.5, 15, fTimeBorder2, fLastTimeBin,         5);

  for( Int_t i=0; i<3; i++){    
    Int_t is[]={iSector};
    for( Int_t j=0; j<fRows[iSector][iRow]->fSpline[i].GetNPoints(); j++){
      Float_t pad, time;
      fRows[iSector][iRow]->fSpline[i].GetAB(j,pad,time);
      Double_t xx[]={iRow,pad,time};
      fOrigTransform->Transform(xx,is,0,1);
      fRows[iSector][iRow]->fSpline[i].Fill(j,xx);    
    }
  }
  return 0;
}



Int_t AliHLTTPCFastTransform::GetRowSize( Int_t iSec, Int_t iRow ) const 
{ 
  // see header file for class documentation
  Int_t s = sizeof(AliHLTTPCFastTransform::AliRowTransform);
  if( fRows[iSec][iRow] ) for( Int_t i=0; i<3; i++) s+=fRows[iSec][iRow]->fSpline[i].GetMapSize();
  return s;
}

Int_t AliHLTTPCFastTransform::GetSize() const
{ 
  // see header file for class documentation
  Int_t s = sizeof(AliHLTTPCFastTransform);
  for( Int_t i=0; i<72; i++ )
    for( Int_t j=0; j<100; j++ ) if( fRows[i][j] ){
	s+= sizeof(AliHLTTPCFastTransform::AliRowTransform);
	for( Int_t k=0; k<3; k++) fRows[i][j]->fSpline[k].GetMapSize();
      }
  return s;
}
