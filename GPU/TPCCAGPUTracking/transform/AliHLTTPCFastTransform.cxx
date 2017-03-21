//************************OB**************************************************
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
#include "AliTPCRecoParam.h"
#include "AliTPCcalibDB.h"
#include "AliHLTTPCFastTransformObject.h"
#include "AliHLTTPCDataCompressionComponent.h"
#include "AliHLTTPCClusterStatComponent.h"
#include "AliHLTTPCGeometry.h"

#include <iostream>
#include <iomanip>

using namespace std;

ClassImp(AliHLTTPCFastTransform); //ROOT macro for the implementation of ROOT specific class methods


AliHLTTPCFastTransform::AliHLTTPCFastTransform()
:
  fMinInitSec(0),
  fMaxInitSec(fkNSec),
  fError(),
  fInitialisationMode(-1),
  fOrigTransform(0),
  fLastTimeStamp(-1),
  fLastTimeBin(600),
  fTimeBorder1(100),
  fTimeBorder2(500),
  fAlignment(NULL),
  fReverseTransformInfo(),
  fUseCorrectionMap(false)
{
  // see header file for class documentation
  // or
  // refer to README to build package
  // or
  // visit http://web.ift.uib.no/~kjeks/doc/alice-hlt    
  for( Int_t i=0; i<fkNSec; i++)
    for( Int_t j=0; j<fkNRows; j++ ) fRows[i][j] = NULL;
}

AliHLTTPCFastTransform::~AliHLTTPCFastTransform() 
{ 
  // see header file for class documentation
  DeInit();
}


void  AliHLTTPCFastTransform::DeInit()
{
  // Deinitialisation

  for( Int_t i=0; i<fkNSec; i++){
    for( Int_t j=0; j<fkNRows; j++ ){
      delete fRows[i][j]; 
      fRows[i][j] = 0;
    }
  }
  fOrigTransform = NULL;
  fLastTimeStamp = -1;
  delete[] fAlignment;
  fAlignment = NULL;
  fError = "";
  fInitialisationMode = -1;
}


Int_t  AliHLTTPCFastTransform::Init( AliTPCTransform *transform, Long_t TimeStamp )
{
  // Initialisation 

  DeInit();
  fInitialisationMode = 1;

  AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();  
  if(!pCalib ) return Error( -1, "AliHLTTPCFastTransform::Init: No TPC calibration instance found");

  AliTPCParam *tpcParam = pCalib->GetParameters(); 
  if( !tpcParam ) return Error( -2, "AliHLTTPCFastTransform::Init: No TPCParam object found");

  if( !transform ) transform = pCalib->GetTransform();    
  if( !transform ) return Error( -3, "AliHLTTPCFastTransform::Init: No TPC transformation found");
 
  tpcParam->Update();
  tpcParam->ReadGeoMatrices();  


  // at the moment initialise all the rows
  
  int nSec = tpcParam->GetNSector();
  if( nSec>fkNSec ) nSec = fkNSec;

  for( Int_t i=0; i<nSec; i++ ){     
    int nRows = tpcParam->GetNRow(i);
    if( nRows>fkNRows ) nRows = fkNRows;
    for( int j=0; j<nRows; j++){
      if( !fRows[i][j] ) fRows[i][j] = new AliHLTTPCFastTransform::AliRowTransform;
      if( !fRows[i][j] ) return Error( -4, "AliHLTTPCFastTransform::Init: Not enough memory");
    }
  }

  const AliTPCRecoParam *rec = transform->GetCurrentRecoParam();

  if( !rec ) return Error( -5, "AliHLTTPCFastTransform::Init: No TPC Reco Param set in transformation");
  fUseCorrectionMap = rec->GetUseCorrectionMap();
  if (fUseCorrectionMap) transform->SetCorrectionMapMode(kTRUE); //If the simulation set this to false to simulate distortions, we need to reverse it for the transformation

  fOrigTransform = transform;

  if( rec->GetUseSectorAlignment() && (!pCalib->HasAlignmentOCDB()) ){

    fAlignment = new Float_t [fkNSec*21];
    for( Int_t iSec=0; iSec<fkNSec; iSec++ ){
      Float_t *t = fAlignment + iSec*21, *r = t+3, *v = t+12;      
      for( int i=0; i<21; i++ ) t[i]=0.f;
      r[0] = r[4] = r[8] = 1.f;
      v[0] = v[4] = v[8] = 1.f;
    }

    for( Int_t iSec=0; iSec<nSec; iSec++ ){
      Float_t *t = fAlignment + iSec*21, *r = t+3, *v = t+12;
      TGeoHMatrix  *alignment = tpcParam->GetClusterMatrix( iSec );
      if ( alignment ){
	const Double_t *tr = alignment->GetTranslation();
	const Double_t *rot = alignment->GetRotationMatrix();
	if( tr && rot ){
	  for( int i=0; i<3; i++ ) t[i] = tr[i];
	  for( int i=0; i<9; i++ ) r[i] = rot[i];
	  CalcAdjugateRotation(r,v,1);
	}
      }
    } 
  }
  
  return SetCurrentTimeStamp( TimeStamp );
}


Int_t AliHLTTPCFastTransform::WriteToObject( AliHLTTPCFastTransformObject &obj )
{
  //
  // write fast transformation to ROOT object to store it in database
  //
  obj.Reset();

  if( obj.GetNSec() < fkNSec ) return Error( -10, "AliHLTTPCFastTransform::WriteToObject: Wrong N Sectors in object");

  obj.SetLastTimeBin( fLastTimeBin );
  obj.SetTimeSplit1( fTimeBorder1 );
  obj.SetTimeSplit2( fTimeBorder2 );

  TArrayF &alignment =obj.GetAlignmentNonConst();
  if( !fAlignment ) alignment.Set(0);
  else{
    alignment.Set( fkNSec*21 );
    for( Int_t i=0; i<fkNSec*21; i++ ) alignment[i] = fAlignment[i];
  }

  for( Int_t iSec=fMinInitSec; iSec<fMaxInitSec; iSec++ ){
    for( int iRow=0; iRow<fkNRows; iRow++){
      if( !fRows[iSec][iRow] ) break;
      if( ( iSec<obj.GetNSecIn() && iRow>=obj.GetNRowsIn() ) || ( iSec>=obj.GetNSecIn() && iRow>=obj.GetNRowsOut() ) ){
	  return Error( -10, "AliHLTTPCFastTransform::WriteToObject: Wrong N Rows in object");
	}
      for( int iSpline=0; iSpline<3; iSpline++ ){
	AliHLTTPCSpline2D3D & spline = fRows[iSec][iRow]->fSpline[iSpline];
	AliHLTTPCSpline2D3DObject& splineObj = obj.GetSplineNonConst( iSec, iRow, iSpline );
	spline.WriteToObject( splineObj );
      }
    }
    obj.SetInitSec(iSec, true);
  }
  obj.SetReverseTransformInfo(fReverseTransformInfo);

  return 0;
}



Int_t AliHLTTPCFastTransform::ReadFromObject( const AliHLTTPCFastTransformObject &obj )
{
  //
  // read fast transformation from ROOT object in database
  //

  if (fgkUseOrigTransform) return 0;
  DeInit();
  fInitialisationMode = 0;

  if( obj.GetNSec() > fkNSec ) return Error( -10, "AliHLTTPCFastTransform::ReadFromObject: Wrong N Sectors in object");

  fOrigTransform = NULL;
  fLastTimeStamp = 0;

  fLastTimeBin = obj.GetLastTimeBin();
  fTimeBorder1 = obj.GetTimeSplit1();
  fTimeBorder2 = obj.GetTimeSplit2();

  AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();  
  if(!pCalib ) return Error( -1, "AliHLTTPCFastTransform::Init: No TPC calibration instance found");

  AliTPCParam *tpcParam = pCalib->GetParameters(); 
  if( !tpcParam ) return Error( -2, "AliHLTTPCFastTransform::Init: No TPCParam object found");

  tpcParam->Update();
  tpcParam->ReadGeoMatrices();

  int nSec = tpcParam->GetNSector();
  if( nSec>fkNSec ) nSec = fkNSec;
  
  for( Int_t iSec=0; iSec<nSec; iSec++ ){

    if (obj.IsSectorInit(iSec) == false) return Error( -10, "AliHLTTPCFastTransform::ReadFromObject: Cannot initialize from Object that does not contain full transform map!");

    int nRows = tpcParam->GetNRow(iSec);

    if( nRows>fkNRows ) return Error( -10, Form("AliHLTTPCFastTransform::ReadFromObject: N of rows in the fast transformation is too low: %d, needed %d for sector %d", fkNRows, nRows , iSec) );

    if( iSec<obj.GetNSecIn() && nRows>obj.GetNRowsIn() ) return Error( -10, Form( "AliHLTTPCFastTransform::ReadFromObject: N of rows in the fast transformation obect is too low: %d, needed %d for sector %d", obj.GetNRowsIn(), nRows , iSec) );

    if( iSec>=obj.GetNSecIn() && nRows>obj.GetNRowsOut() ) return Error( -10, Form("AliHLTTPCFastTransform::ReadFromObject: N of rows in the fast transformation obect is too low: %d, needed %d for sector %d", obj.GetNRowsOut(), nRows , iSec) );
  
    for( int iRow=0; iRow<nRows; iRow++){
      if( !fRows[iSec][iRow] ) fRows[iSec][iRow] = new AliHLTTPCFastTransform::AliRowTransform;
      if( !fRows[iSec][iRow] ) return Error( -4, "AliHLTTPCFastTransform::ReadFromObject: Not enough memory");
      for( int iSpline=0; iSpline<3; iSpline++ ){
	AliHLTTPCSpline2D3D & spline = fRows[iSec][iRow]->fSpline[iSpline];
	const AliHLTTPCSpline2D3DObject& splineObj = obj.GetSpline( iSec, iRow, iSpline );
	spline.ReadFromObject( splineObj );
      }
    }
  }
  
  const TArrayF &alignment =obj.GetAlignment();
  delete[] fAlignment;
  fAlignment = 0;  
  if( alignment.GetSize() == fkNSec*21 ){
    fAlignment = new Float_t [fkNSec*21];
    for( Int_t i=0; i<fkNSec*21; i++ ) fAlignment[i] = alignment[i];
  }
  
  fReverseTransformInfo = obj.GetReverseTransformInfo();
  return 0;
}



bool AliHLTTPCFastTransform::CalcAdjugateRotation(const Float_t *mA, Float_t *mB, bool bCheck)
{
  // check rotation matrix and adjugate for consistency
  //
  // ( for a rotation matrix inverse == transpose )
  //

  mB[0] = mA[0];
  mB[1] = mA[3];
  mB[2] = mA[6];

  mB[3] = mA[1];
  mB[4] = mA[4];
  mB[5] = mA[7];

  mB[6] = mA[2];
  mB[7] = mA[5];
  mB[8] = mA[8];

  if (bCheck) {
    for (int r=0; r<3; r++) {
      for (int c=0; c<3; c++) {
	float a=0.;
	float expected=0.;
	if (r==c) expected=1.;
	for (int i=0; i<3; i++) {
	  a+=mA[3*r+i]*mB[c+(3*i)];
	}
	if (TMath::Abs(a-expected)>0.00001) {
	  std::cout << "inconsistent adjugate at " << r << c << ": " << a << " " << expected << std::endl;
	  for( int i=0; i<9; i++ ) mB[i] = 0;
	  mB[0] = mB[4] = mB[8] = 1;
	  return false;
	}
      }
    }
  }
  return true;
}



Int_t AliHLTTPCFastTransform::SetCurrentTimeStamp( Long_t TimeStamp )
{
  // Set the current time stamp
  
  if( fInitialisationMode!=1 ){
    fLastTimeStamp = TimeStamp;
    return 0;
  }

  Long_t lastTS = fLastTimeStamp;
  fLastTimeStamp = -1; // deinitialise

  if( !fOrigTransform ) return Error( -1, "AliHLTTPCFastTransform::SetCurrentTimeStamp: TPC transformation has not been set properly"); 

  AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();  
  if(!pCalib ) return Error( -2, "AliHLTTPCFastTransform::SetCurrentTimeStamp: No TPC calibration found"); 
  
  AliTPCParam *tpcParam = pCalib->GetParameters(); 
  if( !tpcParam ) return  Error( -3, "AliHLTTPCFastTransform::SetCurrentTimeStamp: No TPCParam object found"); 
  
  
  if( TimeStamp<0  ) return 0; 

  fLastTimeStamp = lastTS;

  if( fLastTimeStamp>=0 && TMath::Abs(fLastTimeStamp - TimeStamp ) <60 ) return 0;
 
   
  if (fUseCorrectionMap) fOrigTransform->SetCorrectionMapMode(kTRUE); //If the simulation set this to false to simulate distortions, we need to reverse it for the transformation
  fOrigTransform->SetCurrentTimeStamp( static_cast<UInt_t>(TimeStamp) );
  fLastTimeStamp = TimeStamp;  
  
  if (fgkUseOrigTransform) return(0);

  // find last calibrated time bin
  
  Int_t nTimeBins = tpcParam->GetMaxTBin();
  Int_t is[]={0};
  bool sign = 0;
  for( fLastTimeBin=0; fLastTimeBin<nTimeBins; fLastTimeBin++){
    // static cast is okay since fLastTimeBin has limited value range  
    Double_t xx[]={0,0,static_cast<Double_t>(fLastTimeBin)};
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
  
  int nSec = tpcParam->GetNSector();
  if( nSec>fkNSec ) nSec = fkNSec;

  for( Int_t i=fMinInitSec; i<fMaxInitSec; i++ ){     
    int nRows = tpcParam->GetNRow(i);
    if( nRows>fkNRows ) nRows = fkNRows;
    for( int j=0; j<nRows; j++){
      if( fRows[i][j] ){
	int err = InitRow(i,j);
	if( err!=0 ) return err;
      }
    }
  }

  AliHLTTPCReverseTransformInfoV1* info = fOrigTransform->GetReverseTransformInfo();

  AliHLTTPCDataCompressionComponent::CalculateDriftTimeTransformation(*this, 0, 0, info->fDriftTimeFactorA, info->fDriftTimeOffsetA, info);
  AliHLTTPCDataCompressionComponent::CalculateDriftTimeTransformation(*this, 18, 0, info->fDriftTimeFactorC, info->fDriftTimeOffsetC, info);
  
  TLinearFitter fitter(3,"hyp2");
  int nPoints = 0;
  for(int i = 0;i < 36;i+=4)
  {
    for (int j = 0;j < 159;j += 30)
    {
      for (int k = 0;k < AliHLTTPCGeometry::GetNPads(j);k += AliHLTTPCGeometry::GetNPads(j) / 10)
      {
        for (int l = 0;l < fLastTimeBin;l += fLastTimeBin / 10)
        {
	  int sector, sectorrow;
          if (j < AliHLTTPCGeometry::GetNRowLow())
          {
            sector = i;
            sectorrow = j;
          }
          else
          {
            sector = i + 36;
            sectorrow = j - AliHLTTPCGeometry::GetNRowLow();
          }
          float xyz[3];
          Transform(sector, sectorrow, k, l, xyz);
          float xyz2[3];
          AliHLTTPCClusterStatComponent::TransformForward(i, j, k, l, xyz2, info);
          Double_t x[2] = {xyz[0], AliHLTTPCGeometry::GetZLength() - fabs(xyz[2])}; 
          fitter.AddPoint(x, xyz[1] - xyz2[1]);
          nPoints++;
        }
      }
    }
  }
  fitter.Eval();
  TVectorD param(3);
  fitter.GetParameters(param);
  info->fCorrectY1 = param[0];
  info->fCorrectY2 = param[1];
  info->fCorrectY3 = param[2];

  fReverseTransformInfo = *info;
  delete info;
  return 0;
}


Int_t AliHLTTPCFastTransform::InitRow( Int_t iSector, Int_t iRow )
{
  // see header file for class documentation
  
  AliTPCcalibDB* pCalib=AliTPCcalibDB::Instance();  

  if( iSector<0 || iSector>=fkNSec || iRow<0 || iRow>=fkNRows || !fOrigTransform || (fLastTimeStamp<0) ||
      !fRows[iSector][iRow] || !pCalib || !pCalib->GetParameters() ){
    return Error( -1, "AliHLTTPCFastTransform::InitRow: Internal error");
  }

  AliTPCParam *tpcParam = pCalib->GetParameters(); 

  Int_t nPads = tpcParam->GetNPads(iSector,iRow);
  
  if( nPads<2 ){
    return Error( -2, Form("AliHLTTPCFastTransform::InitRow: Wrong NPads=%d for sector %d row %d",nPads,iSector,iRow ));
  }

  fRows[iSector][iRow]->fSpline[0].Init(         0.5, nPads-1+0.5, 15, 0,  fTimeBorder1,         5);
  fRows[iSector][iRow]->fSpline[1].Init(         0.5, nPads-1+0.5, 15, fTimeBorder1, fTimeBorder2,         10);
  fRows[iSector][iRow]->fSpline[2].Init(         0.5, nPads-1+0.5, 15, fTimeBorder2, fLastTimeBin,         5);

  for( Int_t i=0; i<3; i++){    
    Int_t is[]={iSector};
    for( Int_t j=0; j<fRows[iSector][iRow]->fSpline[i].GetNPoints(); j++){
      Float_t pad, time;
      fRows[iSector][iRow]->fSpline[i].GetAB(j,pad,time);
      Double_t xx[]={static_cast<Double_t>(iRow),pad,time};
      fOrigTransform->Transform(xx,is,0,1);
      fRows[iSector][iRow]->fSpline[i].Fill(j,xx);    
    }
    fRows[iSector][iRow]->fSpline[i].Consolidate();
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
  for( Int_t i=0; i<fkNSec; i++ )
    for( Int_t j=0; j<fkNRows; j++ ) if( fRows[i][j] ){
	s+= sizeof(AliHLTTPCFastTransform::AliRowTransform);
	for( Int_t k=0; k<3; k++) fRows[i][j]->fSpline[k].GetMapSize();
      }
  return s;
}


void AliHLTTPCFastTransform::Print(const char* /*option*/) const
{
  // print info
  ios::fmtflags coutflags=std::cout.flags(); // backup cout status flags
  
  if( !fAlignment ){
    std::cout << "AliHLTTPCFastTransform: no alignment transformation";
  } else {
    for( int iSec=0; iSec<fkNSec; iSec++ ){
      std::cout << "AliHLTTPCClusterTransformation for sector " << iSec << std::endl;
      
      const Float_t *mT = fAlignment + iSec*21;
      const Float_t *mR = mT + 3;
      const Float_t *mV = mT + 3+9;
      
      std::cout.setf(ios_base::showpos|ios_base::showpos|ios::right);
      std::cout << "  translation: " << std::endl;
      int r=0;
      for (r=0; r<3; r++) {
	std::cout << setw(7) << fixed << setprecision(2);
	cout << "  " << mT[r] << std::endl;
      }
      std::cout << "  rotation and adjugated rotation: " << std::endl;
      for (r=0; r<3; r++) {
	int c=0;
	std::cout << setw(7) << fixed << setprecision(2);
	for (c=0; c<3; c++) std::cout << "  " << mR[3*r+c];
	std::cout << "      ";
	for (c=0; c<3; c++) std::cout << "  " << mV[3*r+c];
	std::cout << endl;
      }
    }
  }
  std::cout.flags(coutflags); // restore the original flags
}
