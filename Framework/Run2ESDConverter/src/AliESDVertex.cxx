/**************************************************************************
 * Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
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

//-----------------------------------------------------------------
//           Implementation of the Primary Vertex class
//           for the Event Data Summary class
//           This class contains the Primary Vertex
//           of the event coming from reconstruction
// Origin: A.Dainese, andrea.dainese@lnl.infn.it
//-----------------------------------------------------------------

//---- standard headers ----
#include "Riostream.h"
//---- Root headers --------
#include <TMath.h>
#include <TROOT.h>
#include <TMatrixDSym.h>
//---- AliRoot headers -----
#include "AliESDVertex.h"
#include "AliVTrack.h"
#include "AliLog.h"

ClassImp(AliESDVertex)

//--------------------------------------------------------------------------
AliESDVertex::AliESDVertex() :
  AliVertex(),
  fCovXX(0.005*0.005),
  fCovXY(0),
  fCovYY(0.005*0.005),
  fCovXZ(0),
  fCovYZ(0),
  fCovZZ(5.3*5.3),
  fChi2(0),
  fID(-1),   // ID=-1 means the vertex with the biggest number of contributors 
  fBCID(AliVTrack::kTOFBCNA)
{
  //
  // Default Constructor, set everything to 0
  //
  SetToZero();
}

//--------------------------------------------------------------------------
AliESDVertex::AliESDVertex(Double_t positionZ,Double_t sigmaZ,
			   Int_t nContributors,const Char_t *vtxName) :
  AliVertex(),
  fCovXX(0.005*0.005),
  fCovXY(0),
  fCovYY(0.005*0.005),
  fCovXZ(0),
  fCovYZ(0),
  fCovZZ(sigmaZ*sigmaZ),
  fChi2(0),
  fID(-1),   // ID=-1 means the vertex with the biggest number of contributors 
  fBCID(AliVTrack::kTOFBCNA)
{
  //
  // Constructor for vertex Z from pixels
  //

  SetToZero();

  fPosition[2]   = positionZ;
  SetName(vtxName);
  SetNContributors(nContributors);

}

//------------------------------------------------------------------------- 
AliESDVertex::AliESDVertex(const Double_t position[3],
			   const Double_t covmatrix[6],
			   Double_t chi2,Int_t nContributors,
			   const Char_t *vtxName) :
  AliVertex(position,0.,nContributors),
  fCovXX(covmatrix[0]),
  fCovXY(covmatrix[1]),
  fCovYY(covmatrix[2]),
  fCovXZ(covmatrix[3]),
  fCovYZ(covmatrix[4]),
  fCovZZ(covmatrix[5]),
  fChi2(chi2),
  fID(-1),   // ID=-1 means the vertex with the biggest number of contributors 
  fBCID(AliVTrack::kTOFBCNA)
{
  //
  // Constructor for vertex in 3D from tracks
  //

  SetToZero();
  SetName(vtxName);

}
//--------------------------------------------------------------------------
AliESDVertex::AliESDVertex(Double_t position[3],Double_t sigma[3],
			   const Char_t *vtxName) :
  AliVertex(position,0.,0),
  fCovXX(sigma[0]*sigma[0]),
  fCovXY(0),
  fCovYY(sigma[1]*sigma[1]),
  fCovXZ(0),
  fCovYZ(0),
  fCovZZ(sigma[2]*sigma[2]),
  fChi2(0),
  fID(-1),   // ID=-1 means the vertex with the biggest number of contributors 
  fBCID(AliVTrack::kTOFBCNA)
{
  //
  // Constructor for smearing of true position
  //

  SetToZero();
  SetName(vtxName);

}
//--------------------------------------------------------------------------
AliESDVertex::AliESDVertex(Double_t position[3],Double_t sigma[3],
			   Double_t snr[3], const Char_t *vtxName) :
  AliVertex(position,0.,0),
  fCovXX(sigma[0]*sigma[0]),
  fCovXY(0),
  fCovYY(sigma[1]*sigma[1]),
  fCovXZ(0),
  fCovYZ(0),
  fCovZZ(sigma[2]*sigma[2]),
  fChi2(0),
  fID(-1),   // ID=-1 means the vertex with the biggest number of contributors 
  fBCID(AliVTrack::kTOFBCNA)
{
  //
  // Constructor for Pb-Pb
  //

  SetToZero();
  SetName(vtxName);

  fSNR[0]        = snr[0];
  fSNR[1]        = snr[1];
  fSNR[2]        = snr[2];

}
//--------------------------------------------------------------------------
AliESDVertex::AliESDVertex(const AliESDVertex &source):
  AliVertex(source),
  fCovXX(source.fCovXX),
  fCovXY(source.fCovXY),
  fCovYY(source.fCovYY),
  fCovXZ(source.fCovXZ),
  fCovYZ(source.fCovYZ),
  fCovZZ(source.fCovZZ),
  fChi2(source.fChi2),
  fID(source.fID),
  fBCID(source.fBCID)
{
  //
  // Copy constructor
  //
  for(Int_t i=0;i<3;i++) {
    fSNR[i] = source.fSNR[i];
  }
}
//--------------------------------------------------------------------------
AliESDVertex &AliESDVertex::operator=(const AliESDVertex &source){
  //
  // assignment operator
  //
  if(&source != this){
    AliVertex::operator=(source);
    for(Int_t i=0;i<3;++i)fSNR[i] = source.fSNR[i];
    fCovXX = source.fCovXX;
    fCovXY = source.fCovXY;
    fCovYY = source.fCovYY;
    fCovXZ = source.fCovXZ;
    fCovYZ = source.fCovYZ;
    fCovZZ = source.fCovZZ;
    fChi2 = source.fChi2;
    fID = source.fID;
    fBCID = source.fBCID;
  }
  return *this;
}
//--------------------------------------------------------------------------
void AliESDVertex::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDVertex *robj = dynamic_cast<AliESDVertex*>(&obj);
  if(!robj)return; // not an AliESDVertex
  *robj = *this;

}
//--------------------------------------------------------------------------
void AliESDVertex::SetToZero() {
  //
  // Set the content of arrays to 0. Used by constructors
  //
  for(Int_t i=0; i<3; i++){
    fSNR[i] = 0.;
  }
}
//--------------------------------------------------------------------------
void AliESDVertex::GetSigmaXYZ(Double_t sigma[3]) const {
  //
  // Return errors on vertex position in thrust frame
  //
  sigma[0] = TMath::Sqrt(fCovXX);
  sigma[1] = TMath::Sqrt(fCovYY);
  sigma[2] = TMath::Sqrt(fCovZZ);

  return;
}
//--------------------------------------------------------------------------
void AliESDVertex::GetCovMatrix(Double_t covmatrix[6]) const {
  //
  // Return covariance matrix of the vertex
  //
  covmatrix[0] = fCovXX;
  covmatrix[1] = fCovXY;
  covmatrix[2] = fCovYY;
  covmatrix[3] = fCovXZ;
  covmatrix[4] = fCovYZ;
  covmatrix[5] = fCovZZ;

  return;
}

//--------------------------------------------------------------------------
void AliESDVertex::SetCovarianceMatrix(const Double_t *covmatrix) {
  //
  // Return covariance matrix of the vertex
  //
  fCovXX = covmatrix[0];
  fCovXY = covmatrix[1];
  fCovYY = covmatrix[2];
  fCovXZ = covmatrix[3];
  fCovYZ = covmatrix[4];
  fCovZZ = covmatrix[5];

  return;
}

//--------------------------------------------------------------------------
void AliESDVertex::GetSNR(Double_t snr[3]) const {
  //
  // Return S/N ratios
  //
  for(Int_t i=0;i<3;i++) snr[i] = fSNR[i];

  return;
}
//--------------------------------------------------------------------------
void AliESDVertex::Print(Option_t* /*option*/) const {
  //
  // Print out information on all data members
  //
  printf("ESD vertex position:\n");
  printf("   x = %f +- %f\n",fPosition[0], fCovXX>0 ? TMath::Sqrt(fCovXX) : 0.);
  printf("   y = %f +- %f\n",fPosition[1], fCovYY>0 ? TMath::Sqrt(fCovYY) : 0.);
  printf("   z = %f +- %f\n",fPosition[2], fCovZZ>0 ? TMath::Sqrt(fCovZZ) : 0.);
  printf(" Covariance matrix:\n");
  printf(" %12.10f  %12.10f  %12.10f\n %12.10f  %12.10f  %12.10f\n %12.10f  %12.10f  %12.10f\n",fCovXX,fCovXY,fCovXZ,fCovXY,fCovYY,fCovYZ,fCovXZ,fCovYZ,fCovZZ);
  printf(" S/N = (%f, %f, %f)\n",fSNR[0],fSNR[1],fSNR[2]);
  printf(" chi2 = %f\n",fChi2);
  printf(" # tracks (or tracklets) = %d BCID=%d\n",fNContributors,int(fBCID));
  //
  if (fCovXX<0 || fCovYY<0 || fCovZZ<0) {AliError("Attention: negative diagonal element");}
  //
  return;
}


//____________________________________________________________
Double_t AliESDVertex::GetWDist(const AliESDVertex* v) const
{
  // calculate sqrt of weighted distance to other vertex
  static TMatrixDSym vVb(3);
  double dist = -1;
  double dx = fPosition[0]-v->fPosition[0], dy = fPosition[1]-v->fPosition[1], dz = fPosition[2]-v->fPosition[2];
  vVb(0,0) = fCovXX + v->fCovXX;
  vVb(1,1) = fCovYY + v->fCovYY;
  vVb(2,2) = fCovZZ + v->fCovZZ;;
  vVb(1,0) = vVb(0,1) = fCovXY + v->fCovXY;
  vVb(0,2) = vVb(1,2) = vVb(2,0) = vVb(2,1) = 0.;
  vVb.InvertFast();
  if (!vVb.IsValid()) {AliError("Singular Matrix"); return dist;}
  dist = vVb(0,0)*dx*dx + vVb(1,1)*dy*dy + vVb(2,2)*dz*dz
    +    2*vVb(0,1)*dx*dy + 2*vVb(0,2)*dx*dz + 2*vVb(1,2)*dy*dz;
  return dist>0 ? TMath::Sqrt(dist) : -1; 
}
