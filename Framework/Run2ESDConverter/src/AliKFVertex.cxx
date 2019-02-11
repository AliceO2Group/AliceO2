//----------------------------------------------------------------------------
// Implementation of the AliKFVertex class
// .
// @author  S.Gorbunov, I.Kisel
// @version 1.0
// @since   13.05.07
// 
// Class to reconstruct and store primary and secondary vertices
// The method is described in CBM-SOFT note 2007-003, 
// ``Reconstruction of decayed particles based on the Kalman filter'', 
// http://www.gsi.de/documents/DOC-2007-May-14-1.pdf
//
// This class is ALICE interface to general mathematics in AliKFParticleCore
// 
//  -= Copyright &copy ALICE HLT Group =-
//____________________________________________________________________________


#include "AliKFVertex.h"

ClassImp(AliKFVertex)


AliKFVertex::AliKFVertex( const AliVVertex &vertex ): fIsConstrained(0)
{
  // Constructor from ALICE VVertex

  vertex.GetXYZ( fP );
  vertex.GetCovarianceMatrix( fC );  
  fChi2 = vertex.GetChi2();  
  fNDF = 2*vertex.GetNContributors() - 3;
  fQ = 0;
  fAtProductionVertex = 0;
  fIsLinearized = 0;
  fSFromDecay = 0;
}

/*
void     AliKFVertex::Print(Option_t* ) const
{  
  cout<<"AliKFVertex position:    "<<GetX()<<" "<<GetY()<<" "<<GetZ()<<endl;
  cout<<"AliKFVertex cov. matrix: "<<GetCovariance(0)<<endl;
  cout<<"                         "<<GetCovariance(1)<<" "<<GetCovariance(2)<<endl;
  cout<<"                         "<<GetCovariance(3)<<" "<<GetCovariance(4)<<" "<<GetCovariance(5)<<endl;
}
  */

void AliKFVertex::SetBeamConstraint( Double_t x, Double_t y, Double_t z, 
				     Double_t errX, Double_t errY, Double_t errZ )
{
  // Set beam constraint to the vertex
  fP[0] = x;
  fP[1] = y;
  fP[2] = z;
  fC[0] = errX*errX;
  fC[1] = 0;
  fC[2] = errY*errY;
  fC[3] = 0;
  fC[4] = 0;
  fC[5] = errZ*errZ;
  fIsConstrained = 1;
}

void AliKFVertex::SetBeamConstraintOff()
{
  fIsConstrained = 0;
}

void AliKFVertex::ConstructPrimaryVertex( const AliKFParticle *vDaughters[], 
					  int NDaughters, Bool_t vtxFlag[],
					  Double_t ChiCut  )
{
  //* Primary vertex finder with simple rejection of outliers

  if( NDaughters<2 ) return;
  double constrP[3]={fP[0], fP[1], fP[2]};
  double constrC[6]={fC[0], fC[1], fC[2], fC[3], fC[4], fC[5]};

  Construct( vDaughters, NDaughters, 0, -1, fIsConstrained );

  SetVtxGuess( fVtxGuess[0], fVtxGuess[1], fVtxGuess[2] );

  for( int i=0; i<NDaughters; i++ ) vtxFlag[i] = 1;

  Int_t nRest = NDaughters;
  while( nRest>2 )
    {    
      Double_t worstChi = 0.;
      Int_t worstDaughter = 0;
      for( Int_t it=0; it<NDaughters; it++ ){
	if( !vtxFlag[it] ) continue;	
	const AliKFParticle &p = *(vDaughters[it]);
	AliKFVertex tmp = *this - p;
	Double_t chi = p.GetDeviationFromVertex( tmp );      
	if( worstChi < chi ){
	  worstChi = chi;
	  worstDaughter = it;
	}
      }
      if( worstChi < ChiCut ) break;
      
      vtxFlag[worstDaughter] = 0;    
      *this -= *(vDaughters[worstDaughter]);
      nRest--;
    } 

  if( nRest>=2 ){// final refit     
    SetVtxGuess( fP[0], fP[1], fP[2] );
    if( fIsConstrained ){
      fP[0] = constrP[0];
      fP[1] = constrP[1];
      fP[2] = constrP[2];
      for( int i=0; i<6; i++ ) fC[i] = constrC[i];
    }
    int nDaughtersNew=0;
    const AliKFParticle **vDaughtersNew=new const AliKFParticle *[NDaughters];
    for( int i=0; i<NDaughters; i++ ){
      if( vtxFlag[i] )  vDaughtersNew[nDaughtersNew++] = vDaughters[i];
    }
    Construct( vDaughtersNew, nDaughtersNew, 0, -1, fIsConstrained );
    delete[] vDaughtersNew;
  }

  if( nRest<=2 && GetChi2()>ChiCut*ChiCut*GetNDF() ){
    for( int i=0; i<NDaughters; i++ ) vtxFlag[i] = 0;
    fNDF = -3;
    fChi2 = 0;
  }
}
