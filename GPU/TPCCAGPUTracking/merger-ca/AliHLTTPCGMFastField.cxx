// $Id: AliHLTTPCGMFastField.cxx 41769 2010-06-16 13:58:00Z sgorbuno $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
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

#include "AliHLTTPCGMFastField.h"
#include <cmath>


#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)

#include "AliTracker.h"
#include "AliHLTTPCGeometry.h"
#include "TFile.h"
#include "TMath.h"
#include "TNtuple.h"
#include "Riostream.h"
#include "AliExternalTrackParam.h"

void AliHLTTPCGMFastField::DumpField( const char *fileName ) const
{

  const double sectorAngleShift = 10./180.*TMath::Pi();
  const double sectorAngle = 20./180.*TMath::Pi();
  const int nRows = AliHLTTPCGeometry::GetNRows();
  
  double xMin = AliHLTTPCGeometry::Row2X(0);
  double xMax = AliHLTTPCGeometry::Row2X(nRows-1);
  double rMin = xMin;
  double rMax = xMax/TMath::Cos(sectorAngle/2.);  
  double dA = 1./rMax; // angular step == 1 cm at outer radius
  int nSectorParticles = (int) (sectorAngle/dA);
  dA = sectorAngle/nSectorParticles;
    
  double zMin = -AliHLTTPCGeometry::GetZLength();
  double zMax =  AliHLTTPCGeometry::GetZLength();

  double alMin = -sectorAngle/2.;
  double alMax =  sectorAngle/2. - 0.5*dA;
     
  Double_t solenoidBz = AliTracker::GetBz();
  
  cout<<"solenoidBz "<<solenoidBz<<" almost0 = "<<0.5*kAlmost0Field<<endl;

  TFile *file = new TFile(fileName,"RECREATE");
  file->cd();
  TNtuple *nt = new TNtuple("field","field","x:y:z:Bx:By:Bz");
     
  for( int sector=0; sector<18; sector++){
    double asec = sectorAngleShift + sector*sectorAngle;
    double cs = TMath::Cos(asec);
    double ss = TMath::Sin(asec);
    for( double al=alMin; al<alMax; al+=dA ){
      double tg = TMath::Tan(al);
      for( int row=0; row<AliHLTTPCGeometry::GetNRows(); row++){
	double xl = AliHLTTPCGeometry::Row2X(row);
	double yl = xl*tg;
	double x = xl*cs - yl*ss;
	double y = xl*ss + yl*cs;
	cout<<"sector = "<<sector<<" al = "<<al/TMath::Pi()*180.<<" xl "<<xl<<" yl "<<yl<<endl;
	
	for( double z=zMin; z<=zMax; z+=1. ){ // 1 cm step in Z
	  Double_t xyz[3] = {x,y,z};
	  Double_t B[3];
	  AliTracker::GetBxByBz(xyz,B);	  
	  B[0]/=solenoidBz;
	  B[1]/=solenoidBz;
	  B[2]/=solenoidBz;	  
	  nt->Fill(x,y,z,B[0],B[1],B[2]);
	}
      }
    }
  }
  nt->Write();
  file->Write();
  file->Close();
  delete file;
  //delete nt;  
}
 
#else

void AliHLTTPCGMFastField::DumpField( const float *) const
{
  // do nothing
}

#endif
