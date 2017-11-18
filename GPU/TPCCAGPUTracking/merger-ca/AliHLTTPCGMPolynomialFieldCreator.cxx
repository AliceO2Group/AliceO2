// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Vito Nordloh <vito.nordloh@vitonordloh.de>              *
//                  Sergey Gorbunov <sergey.gorbunov@fias.uni-frankfurt.de> *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include "AliHLTTPCGMPolynomialFieldCreator.h"
#include "AliHLTTPCGMPolynomialField.h"
#include <cmath>
  

int AliHLTTPCGMPolynomialFieldCreator::GetPolynomialField( StoredField_t fieldType, float nominalFieldkG, AliHLTTPCGMPolynomialField &field )
{
  //
  // get pre-calculated polynomial field approximation of the TPC region
  // returns -1 when the polynomial field is not exist
  // returns -2 if number of coefficients in AliHLTTPCGMPolynomialField is not 10
  //

  const int kM = 10;  

  //
  // polynomial coefficients for the Uniform Bz field 
  //

  const float kSolUBx[kM] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
  const float kSolUBy[kM] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
  const float kSolUBz[kM] = {1., 0., 0., 0., 0., 0., 0., 0., 0., 0. };


  //
  // polynomial coefficients for 2kG field
  //
  
  const float kSol2Bx[kM] = { 8.25026654638e-06,
			      2.73111226079e-07, 8.09913785815e-07, -4.43062708655e-06,
			      -1.12499973781e-08, 3.94054833208e-09, 2.66427264251e-07, -6.30059693307e-09, 2.79869932784e-10, 1.15630518494e-08 };

  const float kSol2By[kM] = {-1.62876094691e-04,
			      8.20370075871e-07, -2.60450360656e-06, 5.25321956957e-06,
			      1.18615373079e-09, -1.44053808881e-08, 1.92043728142e-10, -2.99749697286e-10, 2.66646878799e-07, -1.15439746651e-09 };

  const float kSol2Bz[kM] = { 9.99487757683e-01,
			     -5.67969527765e-06, 4.76676314065e-06, 2.46677245741e-06,
			      1.46798569745e-07, 5.39603639549e-10, 2.86027042051e-08, 1.45939324625e-07, -2.48197662422e-09, -2.47860867830e-07 };
 
  //
  // polynomial coefficients for 5kG field
  //

  const float kSol5Bx[kM] = {-2.58322252193e-05,
			       2.25564940592e-06, -4.14718357433e-08, -2.75251750281e-06,
			      -8.72029382037e-09,  1.72417402577e-09,  3.19352068345e-07, -3.28086002810e-09,  5.64790381130e-10,  8.92192542068e-09 };
  const float kSol5By[kM] = { 6.37950097371e-06,
			      -4.46194050596e-08,  9.01212274584e-07,  8.26001087262e-06,
			       7.99017740860e-10, -7.45108241773e-09,  4.81764572680e-10,  8.35443714209e-10,  3.14677095048e-07, -1.18421328299e-09 };  
  const float kSol5Bz[kM] = { 9.99663949013e-01,
			      -3.54553162651e-06,  7.73496958573e-06, -2.90551361104e-06,
			       1.69738939348e-07,  5.00871899511e-10,  2.10037196524e-08,  1.66827078374e-07, -2.64136179595e-09, -3.02637317873e-07 };

  
  const double kCLight = 0.000299792458;

  field.Reset();

  // check if AliHLTTPCGMPolynomialField class is ok

  if( AliHLTTPCGMPolynomialField::fkM != kM ){
    return -2;
  }
  
  // check which field map is in use
  
  const float *cBx, *cBy, *cBz;

  double nominalBz = nominalFieldkG * kCLight;
  
  if( fieldType == AliHLTTPCGMPolynomialFieldCreator::kUniform) {
    cBx = kSolUBx;
    cBy = kSolUBy;
    cBz = kSolUBz;
  } else if( fieldType == AliHLTTPCGMPolynomialFieldCreator::k2kG) {     
    cBx = kSol2Bx;
    cBy = kSol2By;
    cBz = kSol2Bz;
  } else if( fieldType == AliHLTTPCGMPolynomialFieldCreator::k5kG) {
    cBx = kSol5Bx;
    cBy = kSol5By;
    cBz = kSol5Bz;
  } else { // field is not known
    return -1;
  }

  float Bx[kM], By[kM], Bz[kM];
  for( int i=0; i<kM; i++ ){
    Bx[i] = nominalBz*cBx[i];
    By[i] = nominalBz*cBy[i];
    Bz[i] = nominalBz*cBz[i];
  }
  field.Set( nominalBz, Bx, By, Bz );

  return 0;
}


int AliHLTTPCGMPolynomialFieldCreator::GetPolynomialField( float nominalFieldkG, AliHLTTPCGMPolynomialField &field )
{
  //
  // get closest pre-calculated polynomial field approximation of the TPC region  for the given field value nominalFieldkG
  // returns !=0 in case of error 
  //
  // check which field map is in use
  
  field.Reset();

  StoredField_t type = kUnknown;
 
  if( fabs(fabs(nominalFieldkG) - 5.00668 ) <= fabs( fabs(nominalFieldkG) - 2.) ){
    type = k5kG;
  } else {
    type = k2kG;
  }
  
  return GetPolynomialField( type, nominalFieldkG, field ); 
}





/******************************************************************************************
 *
 *  the following code only works inside AliRoot framework with initialised magnetic field  
 *
 *******************************************************************************************/



#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)


#include "AliHLTTPCPolynomFit.h"

#include <cmath>

#include "AliTracker.h"
#include "AliHLTTPCGeometry.h"
#include "TGeoGlobalMagField.h"
#include "AliMagF.h"

#include "TFile.h"
#include "TMath.h"
#include "TNtuple.h"
#include "Riostream.h"

#include "TMatrixD.h"
#include "TH1F.h"
#include "TStyle.h"


int AliHLTTPCGMPolynomialFieldCreator::GetPolynomialField( AliHLTTPCGMPolynomialField &field ) 
{
  //
  // get pre-calculated polynomial field approximation of the TPC region appropriate for the current AliTracker field map (if exists)
  // returns !=0 error when the polynomial field is not exist
  //

  // check which field map is in use
  
  field.Reset();

  AliMagF* fld = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
   
  if( !fld ) return -1;
  
  AliMagF::BMap_t mapType = fld->GetMapType();

  StoredField_t type = kUnknown;

  if( fld->IsUniform() ){
    type = kUniform;
  } else if(mapType == AliMagF::k2kG) {
    type = k2kG;
  } else if(mapType == AliMagF::k5kG) {
    type = k5kG;
  }
  
  return GetPolynomialField( type, AliTracker::GetBz(), field );
}


int AliHLTTPCGMPolynomialFieldCreator::FitField( AliMagF* inputFld, AliHLTTPCGMPolynomialField &polyField )
{
  //
  // Fit magnetic field with polynoms
  //

  const double kCLight = 0.000299792458;
  const double kAlmost0Field = 1.e-13;

  int step = 1; 
    
  polyField.Reset();

  AliMagF* fld = inputFld;
    
  if( !fld ){
    //fld = new AliMagF("Fit", "Fit", 1., 1., AliMagF::k2kG);
    fld = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
  }
  if( !fld ) return -1;
  
  const double sectorAngleShift = 10./180.*TMath::Pi();
  const double sectorAngle = 20./180.*TMath::Pi();
  const int nRows = AliHLTTPCGeometry::GetNRows();

  double xMin = AliHLTTPCGeometry::Row2X(0);
  double xMax = AliHLTTPCGeometry::Row2X(nRows-1);
  double rMin = xMin;
  double rMax = xMax/TMath::Cos(sectorAngle/2.);

  double dA = 1./rMax; // angular step == 1 cm at outer radius
  dA*=step;  
  int nSectorParticles = (int) (sectorAngle/dA);
  if( nSectorParticles < 1 ) nSectorParticles = 1;
  dA = sectorAngle/nSectorParticles;

  double dZ = 1.*step; // step in z == 1 cm
  
  double zMin = -AliHLTTPCGeometry::GetZLength();
  double zMax =  AliHLTTPCGeometry::GetZLength();

  double alMin = -sectorAngle/2.;
  double alMax =  sectorAngle/2. - 0.5*dA;

  Double_t solenoidBzkG = fld->SolenoidField();
  Double_t solenoidBzkGInv = (TMath::Abs(solenoidBzkG) > kAlmost0Field ) ?1./solenoidBzkG :0. ;
  
  cout << "solenoidBz = " << solenoidBzkG <<" kG"<<endl;  
  
  const int M = AliHLTTPCGMPolynomialField::fkM;
  AliHLTTPCPolynomFit fitBx(M);
  AliHLTTPCPolynomFit fitBy(M);
  AliHLTTPCPolynomFit fitBz(M);
  
  for( int sector=0; sector<18; sector++){
    cout << "sector = " << sector << endl;
    double asec = sectorAngleShift + sector*sectorAngle;
    double cs = TMath::Cos(asec);
    double ss = TMath::Sin(asec);
    for( double al=alMin; al<alMax; al+=dA ){
      cout<<"angle "<<al/TMath::Pi()*180.<<" grad "<<endl;
      double tg = TMath::Tan(al);
      for( int row=0; row<AliHLTTPCGeometry::GetNRows(); row++){
	double xl = AliHLTTPCGeometry::Row2X(row);
	double yl = xl*tg;
	double x = xl*cs - yl*ss;
	double y = xl*ss + yl*cs;
	//cout<<"sector = "<<sector<<" al = "<<al/TMath::Pi()*180.<<" xl "<<xl<<" yl "<<yl<<endl;
	
	for( double z=zMin; z<=zMax; z+=dZ ){ // 1 cm step in Z
	  Double_t xyz[3] = {x,y,z};
	  Double_t B[3] = {0.,0.,0.};
	  if(fld->IsUniform()) {
	    B[0] = B[1] = 0.;
	    B[2] = fld->SolenoidField();
	  } else {
	    fld->Field(xyz, B);
	  }
	  B[0]*=solenoidBzkGInv;
	  B[1]*=solenoidBzkGInv;
	  B[2]*=solenoidBzkGInv;	  

	  float f[M];
	  AliHLTTPCGMPolynomialField::GetPolynoms(x,y,z,f);	  
	  fitBx.AddMeasurement( f, B[0]);
	  fitBy.AddMeasurement( f, B[1]);
	  fitBz.AddMeasurement( f, B[2]);   
	  
	}
      }
    }
  }

  // field coefficients
  float cX[M];
  float cY[M];
  float cZ[M];

  int errX = fitBx.Fit( cX );
  int errY = fitBy.Fit( cY );
  int errZ = fitBz.Fit( cZ );
  
  if( errX!=0 || errY!=0 || errZ!=0 ){
    cout<<"Fit of polynamial field failed!!!"<<endl;
    if( fld != inputFld) delete fld;
    return -1;
  }

  AliHLTTPCGMPolynomialField fittedField;
  fittedField.Set( 1., cX, cY, cZ );

  
  // scale result
  double nominalBz = solenoidBzkG * kCLight;

  for( int i=0; i<M; i++ ){
    cX[i] = nominalBz * cX[i];
    cY[i] = nominalBz * cY[i];
    cZ[i] = nominalBz * cZ[i];
  }
  polyField.Set( nominalBz, cX, cY, cZ );
  
  gStyle->SetOptStat(1111111);

  TH1F histBx("Performance B_x", "Error B_x", 1000, -0.005, 0.005);
  TH1F histBy("Performance B_y", "Error B_y", 1000, -0.005, 0.005);
  TH1F histBz("Performance B_z", "Error B_z", 1000, -0.005, 0.005);

  for( int sector=0; sector<18; sector++){
    cout << "check quality: sector = " << sector << endl;
    double asec = sectorAngleShift + sector*sectorAngle;
    double cs = TMath::Cos(asec);
    double ss = TMath::Sin(asec);
    for( double al=alMin; al<alMax; al+=dA ){    
     cout<<"check quality: angle "<<al/TMath::Pi()*180.<<" grad "<<endl;
      double tg = TMath::Tan(al);
      for( int row=0; row<AliHLTTPCGeometry::GetNRows(); row++){
	double xl = AliHLTTPCGeometry::Row2X(row);
	double yl = xl*tg;
	double x = xl*cs - yl*ss;
	double y = xl*ss + yl*cs;		
	for( double z=zMin; z<=zMax; z+=dZ ){
	  Double_t xyz[3] = {x,y,z};
	  Double_t B[3];
	  if(fld->IsUniform()) {
	    B[0] = B[1] = 0.;
	    B[2] = fld->SolenoidField();
	  } else {
	    fld->Field(xyz, B);
	  }
	  B[0]*=solenoidBzkGInv;
	  B[1]*=solenoidBzkGInv;
	  B[2]*=solenoidBzkGInv;	  
          float approxB[3];
          fittedField.GetField(x, y, z, approxB);

          histBx.Fill(approxB[0] - B[0]);
          histBy.Fill(approxB[1] - B[1]);
          histBz.Fill(approxB[2] - B[2]);
        }
      }
    }
  }

  TFile file("FieldFitStat.root", "RECREATE");
  file.cd();

  histBx.Write();
  histBy.Write();
  histBz.Write();

  file.Write();
  file.Close();

  cout<<"Fitted polynomial field: "<<endl;
  fittedField.Print();

  return 0;
}

#endif
