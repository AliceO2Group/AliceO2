//----------------------------------------------------------------------------
// Implementation of the AliKFParticle class
// .
// @author  S.Gorbunov, I.Kisel
// @version 1.0
// @since   13.05.07
// 
// Class to reconstruct and store the decayed particle parameters.
// The method is described in CBM-SOFT note 2007-003, 
// ``Reconstruction of decayed particles based on the Kalman filter'', 
// http://www.gsi.de/documents/DOC-2007-May-14-1.pdf
//
// This class is ALICE interface to general mathematics in AliKFParticleCore
// 
//  -= Copyright &copy ALICE HLT Group =-
//____________________________________________________________________________


#include "AliKFParticle.h"
#include "TDatabasePDG.h"
#include "TParticlePDG.h"
#include "AliVTrack.h"
#include "AliVVertex.h"

#include "AliExternalTrackParam.h"

ClassImp(AliKFParticle)

Double_t AliKFParticle::fgBz = -5.;  //* Bz compoment of the magnetic field

AliKFParticle::AliKFParticle( const AliKFParticle &d1, const AliKFParticle &d2, Bool_t gamma )
{
  if (!gamma) {
    AliKFParticle mother;
    mother+= d1;
    mother+= d2;
    *this = mother;
  } else
    ConstructGamma(d1, d2);
}

void AliKFParticle::Create( const Double_t Param[], const Double_t Cov[], Int_t Charge, Int_t PID )
{
  // Constructor from "cartesian" track, PID hypothesis should be provided
  //
  // Param[6] = { X, Y, Z, Px, Py, Pz } - position and momentum
  // Cov [21] = lower-triangular part of the covariance matrix:
  //
  //                (  0  .  .  .  .  . )
  //                (  1  2  .  .  .  . )
  //  Cov. matrix = (  3  4  5  .  .  . ) - numbering of covariance elements in Cov[]
  //                (  6  7  8  9  .  . )
  //                ( 10 11 12 13 14  . )
  //                ( 15 16 17 18 19 20 )
  Double_t C[21];
  for( int i=0; i<21; i++ ) C[i] = Cov[i];
  
  TParticlePDG* particlePDG = TDatabasePDG::Instance()->GetParticle(PID);
  Double_t mass = (particlePDG) ? particlePDG->Mass() :0.13957;
  
  AliKFParticleBase::Initialize( Param, C, Charge, mass );
}

void AliKFParticle::Create( const Double_t Param[], const Double_t Cov[], Int_t Charge, Double_t Mass )
{
  // Constructor from "cartesian" track, PID hypothesis should be provided
  //
  // Param[6] = { X, Y, Z, Px, Py, Pz } - position and momentum
  // Cov [21] = lower-triangular part of the covariance matrix:
  //
  //                (  0  .  .  .  .  . )
  //                (  1  2  .  .  .  . )
  //  Cov. matrix = (  3  4  5  .  .  . ) - numbering of covariance elements in Cov[]
  //                (  6  7  8  9  .  . )
  //                ( 10 11 12 13 14  . )
  //                ( 15 16 17 18 19 20 )
  Double_t C[21];
  for( int i=0; i<21; i++ ) C[i] = Cov[i];

  AliKFParticleBase::Initialize( Param, C, Charge, Mass );
}

AliKFParticle::AliKFParticle( const AliVTrack &track, Int_t PID )
{
  // Constructor from ALICE track, PID hypothesis should be provided

  track.GetXYZ(fP);
  track.PxPyPz(fP+3);
  fQ = track.Charge();
  track.GetCovarianceXYZPxPyPz( fC );
  Create(fP,fC,fQ,PID);
}

AliKFParticle::AliKFParticle( const AliExternalTrackParam &track, Double_t Mass, Int_t Charge )
{
  // Constructor from ALICE track, Mass and Charge hypothesis should be provided

  track.GetXYZ(fP);
  track.PxPyPz(fP+3);
  fQ = track.Charge()*TMath::Abs(Charge);
  fP[3] *= TMath::Abs(Charge);
  fP[4] *= TMath::Abs(Charge);
  fP[5] *= TMath::Abs(Charge);

  Double_t pt=1./TMath::Abs(track.GetParameter()[4]) * TMath::Abs(Charge);
  Double_t cs=TMath::Cos(track.GetAlpha()), sn=TMath::Sin(track.GetAlpha());
  Double_t r=TMath::Sqrt((1.-track.GetParameter()[2])*(1.+track.GetParameter()[2]));

  Double_t m00=-sn, m10=cs;
  Double_t m23=-pt*(sn + track.GetParameter()[2]*cs/r), m43=-pt*pt*(r*cs - track.GetParameter()[2]*sn);
  Double_t m24= pt*(cs - track.GetParameter()[2]*sn/r), m44=-pt*pt*(r*sn + track.GetParameter()[2]*cs);
  Double_t m35=pt, m45=-pt*pt*track.GetParameter()[3];

  m43*=track.GetSign();
  m44*=track.GetSign();
  m45*=track.GetSign();

  const Double_t *cTr = track.GetCovariance();

  fC[0 ] = cTr[0]*m00*m00;
  fC[1 ] = cTr[0]*m00*m10; 
  fC[2 ] = cTr[0]*m10*m10;
  fC[3 ] = cTr[1]*m00; 
  fC[4 ] = cTr[1]*m10; 
  fC[5 ] = cTr[2];
  fC[6 ] = m00*(cTr[3]*m23 + cTr[10]*m43); 
  fC[7 ] = m10*(cTr[3]*m23 + cTr[10]*m43); 
  fC[8 ] = cTr[4]*m23 + cTr[11]*m43; 
  fC[9 ] = m23*(cTr[5]*m23 + cTr[12]*m43)  +  m43*(cTr[12]*m23 + cTr[14]*m43);
  fC[10] = m00*(cTr[3]*m24 + cTr[10]*m44); 
  fC[11] = m10*(cTr[3]*m24 + cTr[10]*m44); 
  fC[12] = cTr[4]*m24 + cTr[11]*m44; 
  fC[13] = m23*(cTr[5]*m24 + cTr[12]*m44)  +  m43*(cTr[12]*m24 + cTr[14]*m44);
  fC[14] = m24*(cTr[5]*m24 + cTr[12]*m44)  +  m44*(cTr[12]*m24 + cTr[14]*m44);
  fC[15] = m00*(cTr[6]*m35 + cTr[10]*m45); 
  fC[16] = m10*(cTr[6]*m35 + cTr[10]*m45); 
  fC[17] = cTr[7]*m35 + cTr[11]*m45; 
  fC[18] = m23*(cTr[8]*m35 + cTr[12]*m45)  +  m43*(cTr[13]*m35 + cTr[14]*m45);
  fC[19] = m24*(cTr[8]*m35 + cTr[12]*m45)  +  m44*(cTr[13]*m35 + cTr[14]*m45); 
  fC[20] = m35*(cTr[9]*m35 + cTr[13]*m45)  +  m45*(cTr[13]*m35 + cTr[14]*m45);

  Create(fP,fC,fQ,Mass);
}

AliKFParticle::AliKFParticle( const AliVVertex &vertex )
{
  // Constructor from ALICE vertex

  vertex.GetXYZ( fP );
  vertex.GetCovarianceMatrix( fC );  
  fChi2 = vertex.GetChi2();
  fNDF = 2*vertex.GetNContributors() - 3;
  fQ = 0;
  fAtProductionVertex = 0;
  fIsLinearized = 0;
  fSFromDecay = 0;
}

void AliKFParticle::GetExternalTrackParam( const AliKFParticleBase &p, Double_t &X, Double_t &Alpha, Double_t P[5] ) 
{
  // Conversion to AliExternalTrackParam parameterization

  Double_t cosA = p.GetPx(), sinA = p.GetPy(); 
  Double_t pt = TMath::Sqrt(cosA*cosA + sinA*sinA);
  Double_t pti = 0;
  if( pt<1.e-4 ){
    cosA = 1;
    sinA = 0;
  } else {
    pti = 1./pt;
    cosA*=pti;
    sinA*=pti;
  }
  Alpha = TMath::ATan2(sinA,cosA);  
  X   = p.GetX()*cosA + p.GetY()*sinA;   
  P[0]= p.GetY()*cosA - p.GetX()*sinA;
  P[1]= p.GetZ();
  P[2]= 0;
  P[3]= p.GetPz()*pti;
  P[4]= p.GetQ()*pti;
}

Bool_t AliKFParticle::GetDistanceFromVertexXY( const Double_t vtx[], const Double_t Cv[], Double_t &val, Double_t &err ) const
{
  //* Calculate DCA distance from vertex (transverse impact parameter) in XY
  //* v = [xy], Cv=[Cxx,Cxy,Cyy ]-covariance matrix

  Bool_t ret = 0;
  
  Double_t mP[8];
  Double_t mC[36];
  
  Transport( GetDStoPoint(vtx), mP, mC );  

  Double_t dx = mP[0] - vtx[0];
  Double_t dy = mP[1] - vtx[1];
  Double_t px = mP[3];
  Double_t py = mP[4];
  Double_t pt = TMath::Sqrt(px*px + py*py);
  Double_t ex=0, ey=0;
  if( pt<1.e-4 ){
    ret = 1;
    pt = 1.;
    val = 1.e4;
  } else{
    ex = px/pt;
    ey = py/pt;
    val = dy*ex - dx*ey;
  }

  Double_t h0 = -ey;
  Double_t h1 = ex;
  Double_t h3 = (dy*ey + dx*ex)*ey/pt;
  Double_t h4 = -(dy*ey + dx*ex)*ex/pt;
  
  err = 
    h0*(h0*GetCovariance(0,0) + h1*GetCovariance(0,1) + h3*GetCovariance(0,3) + h4*GetCovariance(0,4) ) +
    h1*(h0*GetCovariance(1,0) + h1*GetCovariance(1,1) + h3*GetCovariance(1,3) + h4*GetCovariance(1,4) ) +
    h3*(h0*GetCovariance(3,0) + h1*GetCovariance(3,1) + h3*GetCovariance(3,3) + h4*GetCovariance(3,4) ) +
    h4*(h0*GetCovariance(4,0) + h1*GetCovariance(4,1) + h3*GetCovariance(4,3) + h4*GetCovariance(4,4) );

  if( Cv ){
    err+= h0*(h0*Cv[0] + h1*Cv[1] ) + h1*(h0*Cv[1] + h1*Cv[2] ); 
  }

  err = TMath::Sqrt(TMath::Abs(err));

  return ret;
}

Bool_t AliKFParticle::GetDistanceFromVertexXY( const Double_t vtx[], Double_t &val, Double_t &err ) const
{
  return GetDistanceFromVertexXY( vtx, 0, val, err );
}


Bool_t AliKFParticle::GetDistanceFromVertexXY( const AliKFParticle &Vtx, Double_t &val, Double_t &err ) const 
{
  //* Calculate distance from vertex [cm] in XY-plane

  return GetDistanceFromVertexXY( Vtx.fP, Vtx.fC, val, err );
}

Bool_t AliKFParticle::GetDistanceFromVertexXY( const AliVVertex &Vtx, Double_t &val, Double_t &err ) const 
{
  //* Calculate distance from vertex [cm] in XY-plane

  return GetDistanceFromVertexXY( AliKFParticle(Vtx), val, err );
}

Double_t AliKFParticle::GetDistanceFromVertexXY( const Double_t vtx[] ) const
{
  //* Calculate distance from vertex [cm] in XY-plane
  Double_t val, err;
  GetDistanceFromVertexXY( vtx, 0, val, err );
  return val;
}

Double_t AliKFParticle::GetDistanceFromVertexXY( const AliKFParticle &Vtx ) const 
{
  //* Calculate distance from vertex [cm] in XY-plane

  return GetDistanceFromVertexXY( Vtx.fP );
}

Double_t AliKFParticle::GetDistanceFromVertexXY( const AliVVertex &Vtx ) const 
{
  //* Calculate distance from vertex [cm] in XY-plane

  return GetDistanceFromVertexXY( AliKFParticle(Vtx).fP );
}

Double_t AliKFParticle::GetDistanceFromParticleXY( const AliKFParticle &p ) const 
{
  //* Calculate distance to other particle [cm]

  Double_t dS, dS1;
  GetDStoParticleXY( p, dS, dS1 );   
  Double_t mP[8], mC[36], mP1[8], mC1[36];
  Transport( dS, mP, mC ); 
  p.Transport( dS1, mP1, mC1 ); 
  Double_t dx = mP[0]-mP1[0]; 
  Double_t dy = mP[1]-mP1[1]; 
  return TMath::Sqrt(dx*dx+dy*dy);
}

Double_t AliKFParticle::GetDeviationFromParticleXY( const AliKFParticle &p ) const 
{
  //* Calculate sqrt(Chi2/ndf) deviation from other particle

  Double_t dS, dS1;
  GetDStoParticleXY( p, dS, dS1 );   
  Double_t mP1[8], mC1[36];
  p.Transport( dS1, mP1, mC1 ); 

  Double_t d[2]={ fP[0]-mP1[0], fP[1]-mP1[1] };

  Double_t sigmaS = .1+10.*TMath::Sqrt( (d[0]*d[0]+d[1]*d[1] )/
					(mP1[3]*mP1[3]+mP1[4]*mP1[4] )  );

  Double_t h[2] = { mP1[3]*sigmaS, mP1[4]*sigmaS };       
  
  mC1[0] +=h[0]*h[0];
  mC1[1] +=h[1]*h[0]; 
  mC1[2] +=h[1]*h[1]; 

  return GetDeviationFromVertexXY( mP1, mC1 )*TMath::Sqrt(2./1.);
}


Double_t AliKFParticle::GetDeviationFromVertexXY( const Double_t vtx[], const Double_t Cv[] ) const 
{
  //* Calculate sqrt(Chi2/ndf) deviation from vertex
  //* v = [xyz], Cv=[Cxx,Cxy,Cyy,Cxz,Cyz,Czz]-covariance matrix

  Double_t val, err;
  Bool_t problem = GetDistanceFromVertexXY( vtx, Cv, val, err );
  if( problem || err<1.e-20 ) return 1.e4;
  else return val/err;
}


Double_t AliKFParticle::GetDeviationFromVertexXY( const AliKFParticle &Vtx ) const  
{
  //* Calculate sqrt(Chi2/ndf) deviation from vertex
  //* v = [xyz], Cv=[Cxx,Cxy,Cyy,Cxz,Cyz,Czz]-covariance matrix

  return GetDeviationFromVertexXY( Vtx.fP, Vtx.fC );
}

Double_t AliKFParticle::GetDeviationFromVertexXY( const AliVVertex &Vtx ) const 
{
  //* Calculate sqrt(Chi2/ndf) deviation from vertex
  //* v = [xyz], Cv=[Cxx,Cxy,Cyy,Cxz,Cyz,Czz]-covariance matrix

  AliKFParticle v(Vtx);
  return GetDeviationFromVertexXY( v.fP, v.fC );
}

Double_t AliKFParticle::GetAngle  ( const AliKFParticle &p ) const 
{
  //* Calculate the opening angle between two particles

  Double_t dS, dS1;
  GetDStoParticle( p, dS, dS1 );   
  Double_t mP[8], mC[36], mP1[8], mC1[36];
  Transport( dS, mP, mC ); 
  p.Transport( dS1, mP1, mC1 ); 
  Double_t n = TMath::Sqrt( mP[3]*mP[3] + mP[4]*mP[4] + mP[5]*mP[5] );
  Double_t n1= TMath::Sqrt( mP1[3]*mP1[3] + mP1[4]*mP1[4] + mP1[5]*mP1[5] );
  n*=n1;
  Double_t a = 0;
  if( n>1.e-8 ) a = ( mP[3]*mP1[3] + mP[4]*mP1[4] + mP[5]*mP1[5] )/n;
  if (TMath::Abs(a)<1.) a = TMath::ACos(a);
  else a = (a>=0) ?0 :TMath::Pi();
  return a;
}

Double_t AliKFParticle::GetAngleXY( const AliKFParticle &p ) const 
{
  //* Calculate the opening angle between two particles in XY plane

  Double_t dS, dS1;
  GetDStoParticleXY( p, dS, dS1 );   
  Double_t mP[8], mC[36], mP1[8], mC1[36];
  Transport( dS, mP, mC ); 
  p.Transport( dS1, mP1, mC1 ); 
  Double_t n = TMath::Sqrt( mP[3]*mP[3] + mP[4]*mP[4] );
  Double_t n1= TMath::Sqrt( mP1[3]*mP1[3] + mP1[4]*mP1[4] );
  n*=n1;
  Double_t a = 0;
  if( n>1.e-8 ) a = ( mP[3]*mP1[3] + mP[4]*mP1[4] )/n;
  if (TMath::Abs(a)<1.) a = TMath::ACos(a);
  else a = (a>=0) ?0 :TMath::Pi();
  return a;
}

Double_t AliKFParticle::GetAngleRZ( const AliKFParticle &p ) const 
{
  //* Calculate the opening angle between two particles in RZ plane

  Double_t dS, dS1;
  GetDStoParticle( p, dS, dS1 );   
  Double_t mP[8], mC[36], mP1[8], mC1[36];
  Transport( dS, mP, mC ); 
  p.Transport( dS1, mP1, mC1 ); 
  Double_t nr = TMath::Sqrt( mP[3]*mP[3] + mP[4]*mP[4] );
  Double_t n1r= TMath::Sqrt( mP1[3]*mP1[3] + mP1[4]*mP1[4]  );
  Double_t n = TMath::Sqrt( nr*nr + mP[5]*mP[5] );
  Double_t n1= TMath::Sqrt( n1r*n1r + mP1[5]*mP1[5] );
  n*=n1;
  Double_t a = 0;
  if( n>1.e-8 ) a = ( nr*n1r +mP[5]*mP1[5])/n; 
  if (TMath::Abs(a)<1.) a = TMath::ACos(a);
  else a = (a>=0) ?0 :TMath::Pi();
  return a;
}


/*

#include "AliExternalTrackParam.h"

void AliKFParticle::GetDStoParticleALICE( const AliKFParticleBase &p, 
					   Double_t &DS, Double_t &DS1 ) 
  const
{ 
  DS = DS1 = 0;   
  Double_t x1, a1, x2, a2;
  Double_t par1[5], par2[5], cov[15];
  for(int i=0; i<15; i++) cov[i] = 0;
  cov[0] = cov[2] = cov[5] = cov[9] = cov[14] = .001;

  GetExternalTrackParam( *this, x1, a1, par1 );
  GetExternalTrackParam( p, x2, a2, par2 );

  AliExternalTrackParam t1(x1,a1, par1, cov);
  AliExternalTrackParam t2(x2,a2, par2, cov);

  Double_t xe1=0, xe2=0;
  t1.GetDCA( &t2, -GetFieldAlice(), xe1, xe2 );
  t1.PropagateTo( xe1, -GetFieldAlice() );
  t2.PropagateTo( xe2, -GetFieldAlice() );

  Double_t xyz1[3], xyz2[3];
  t1.GetXYZ( xyz1 );
  t2.GetXYZ( xyz2 );
  
  DS = GetDStoPoint( xyz1 );
  DS1 = p.GetDStoPoint( xyz2 );

  return;
}
*/

  // * Pseudo Proper Time of decay = (r*pt) / |pt| * M/|pt|
Double_t AliKFParticle::GetPseudoProperDecayTime( const AliKFParticle &pV, const Double_t& mass, Double_t* timeErr2 ) const
{ // TODO optimize with respect to time and stability
  const Double_t ipt2 = 1/( Px()*Px() + Py()*Py() );
  const Double_t mipt2 = mass*ipt2;
  const Double_t dx = X() - pV.X();
  const Double_t dy = Y() - pV.Y();

  if ( timeErr2 ) {
      // -- calculate error = sigma(f(r)) = f'Cf'
      // r = {x,y,px,py,x_pV,y_pV}
      // df/dr = { px*m/pt^2,
      //     py*m/pt^2,
      //    ( x - x_pV )*m*(1/pt^2 - 2(px/pt^2)^2),
      //    ( y - y_pV )*m*(1/pt^2 - 2(py/pt^2)^2),
      //     -px*m/pt^2,
      //     -py*m/pt^2 }
    const Double_t f0 = Px()*mipt2;
    const Double_t f1 = Py()*mipt2;
    const Double_t mipt2derivative = mipt2*(1-2*Px()*Px()*ipt2);
    const Double_t f2 = dx*mipt2derivative;
    const Double_t f3 = -dy*mipt2derivative;
    const Double_t f4 = -f0;
    const Double_t f5 = -f1;

    const Double_t& mC00 =    GetCovariance(0,0);
    const Double_t& mC10 =    GetCovariance(0,1);
    const Double_t& mC11 =    GetCovariance(1,1);
    const Double_t& mC20 =    GetCovariance(3,0);
    const Double_t& mC21 =    GetCovariance(3,1);
    const Double_t& mC22 =    GetCovariance(3,3);
    const Double_t& mC30 =    GetCovariance(4,0);
    const Double_t& mC31 =    GetCovariance(4,1);
    const Double_t& mC32 =    GetCovariance(4,3);
    const Double_t& mC33 =    GetCovariance(4,4);
    const Double_t& mC44 = pV.GetCovariance(0,0);
    const Double_t& mC54 = pV.GetCovariance(1,0);
    const Double_t& mC55 = pV.GetCovariance(1,1);

    *timeErr2 =
      f5*mC55*f5 +
      f5*mC54*f4 +
      f4*mC44*f4 +
      f3*mC33*f3 +
      f3*mC32*f2 +
      f3*mC31*f1 +
      f3*mC30*f0 +
      f2*mC22*f2 +
      f2*mC21*f1 +
      f2*mC20*f0 +
      f1*mC11*f1 +
      f1*mC10*f0 +
      f0*mC00*f0;
  }
  return ( dx*Px() + dy*Py() )*mipt2;
}
