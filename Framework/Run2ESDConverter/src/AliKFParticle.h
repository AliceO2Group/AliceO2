//---------------------------------------------------------------------------------
// The AliKFParticle class
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
// This class is ALICE interface to general mathematics in AliKFParticleBase
// 
//  -= Copyright &copy ALICE HLT Group =-
//_________________________________________________________________________________

#ifndef ALIKFPARTICLE_H
#define ALIKFPARTICLE_H

#include "AliKFParticleBase.h"
#include "TMath.h"

class AliVTrack;
class AliVVertex;
class AliExternalTrackParam;

class AliKFParticle :public AliKFParticleBase
{
  
 public:

  //*
  //*  INITIALIZATION
  //*

  //* Set magnetic field for all particles
  
  static void SetField( Double_t Bz );

  //* Constructor (empty)

  AliKFParticle():AliKFParticleBase(){ ; }

  //* Destructor (empty)

  ~AliKFParticle(){ ; }

  //* Construction of mother particle by its 2-3-4 daughters

  AliKFParticle( const AliKFParticle &d1, const AliKFParticle &d2, Bool_t gamma = kFALSE );

  AliKFParticle( const AliKFParticle &d1, const AliKFParticle &d2, 
		 const AliKFParticle &d3 );

  AliKFParticle( const AliKFParticle &d1, const AliKFParticle &d2, 
		 const AliKFParticle &d3, const AliKFParticle &d4 );
 
 //* Initialisation from "cartesian" coordinates ( X Y Z Px Py Pz )
 //* Parameters, covariance matrix, charge and PID hypothesis should be provided 

  void Create( const Double_t Param[], const Double_t Cov[], Int_t Charge, Int_t PID );
  void Create( const Double_t Param[], const Double_t Cov[], Int_t Charge, Double_t Mass );

 //* Initialisation from ALICE track, PID hypothesis shoould be provided 

  AliKFParticle( const AliVTrack &track, Int_t PID );
  AliKFParticle( const AliExternalTrackParam &track, Double_t Mass, Int_t Charge );

  //* Initialisation from VVertex 

  AliKFParticle( const AliVVertex &vertex );

  //* Initialise covariance matrix and set current parameters to 0.0 

  void Initialize();

  //* Set decay vertex parameters for linearisation 

  void SetVtxGuess( Double_t x, Double_t y, Double_t z );

  //*
  //*  ACCESSORS
  //*

  //* Simple accessors 

  Double_t GetX    () const ; //* x of current position
  Double_t GetY    () const ; //* y of current position
  Double_t GetZ    () const ; //* z of current position
  Double_t GetPx   () const ; //* x-compoment of 3-momentum
  Double_t GetPy   () const ; //* y-compoment of 3-momentum
  Double_t GetPz   () const ; //* z-compoment of 3-momentum
  Double_t GetE    () const ; //* energy
  Double_t GetS    () const ; //* decay length / momentum
  Int_t    GetQ    () const ; //* charge
  Double_t GetChi2 () const ; //* chi^2
  Int_t    GetNDF  () const ; //* Number of Degrees of Freedom

  const Double_t& X    () const { return fP[0]; }
  const Double_t& Y    () const { return fP[1]; }
  const Double_t& Z    () const { return fP[2]; }
  const Double_t& Px   () const { return fP[3]; }
  const Double_t& Py   () const { return fP[4]; }
  const Double_t& Pz   () const { return fP[5]; }
  const Double_t& E    () const { return fP[6]; }
  const Double_t& S    () const { return fP[7]; }
  const Int_t   & Q    () const { return fQ;    }
  const Double_t& Chi2 () const { return fChi2; }
  const Int_t   & NDF  () const { return fNDF;  }
  
  Double_t GetParameter ( int i ) const ;
  Double_t GetCovariance( int i ) const ;
  Double_t GetCovariance( int i, int j ) const ;

  //* Accessors with calculations, value returned w/o error flag
  
  Double_t GetP           () const; //* momentum
  Double_t GetPt          () const; //* transverse momentum
  Double_t GetEta         () const; //* pseudorapidity
  Double_t GetPhi         () const; //* phi
  Double_t GetMomentum    () const; //* momentum (same as GetP() )
  Double_t GetMass        () const; //* mass
  Double_t GetDecayLength () const; //* decay length
  Double_t GetDecayLengthXY () const; //* decay length in XY
  Double_t GetLifeTime    () const; //* life time
  Double_t GetR           () const; //* distance to the origin

  //* Accessors to estimated errors

  Double_t GetErrX           () const ; //* x of current position 
  Double_t GetErrY           () const ; //* y of current position
  Double_t GetErrZ           () const ; //* z of current position
  Double_t GetErrPx          () const ; //* x-compoment of 3-momentum
  Double_t GetErrPy          () const ; //* y-compoment of 3-momentum
  Double_t GetErrPz          () const ; //* z-compoment of 3-momentum
  Double_t GetErrE           () const ; //* energy
  Double_t GetErrS           () const ; //* decay length / momentum
  Double_t GetErrP           () const ; //* momentum
  Double_t GetErrPt          () const ; //* transverse momentum
  Double_t GetErrEta         () const ; //* pseudorapidity
  Double_t GetErrPhi         () const ; //* phi
  Double_t GetErrMomentum    () const ; //* momentum
  Double_t GetErrMass        () const ; //* mass
  Double_t GetErrDecayLength () const ; //* decay length
  Double_t GetErrDecayLengthXY () const ; //* decay length in XY
  Double_t GetErrLifeTime    () const ; //* life time
  Double_t GetErrR           () const ; //* distance to the origin

  //* Accessors with calculations( &value, &estimated sigma )
  //* error flag returned (0 means no error during calculations) 

  int GetP           ( Double_t &P, Double_t &SigmaP ) const ;   //* momentum
  int GetPt          ( Double_t &Pt, Double_t &SigmaPt ) const ; //* transverse momentum
  int GetEta         ( Double_t &Eta, Double_t &SigmaEta ) const ; //* pseudorapidity
  int GetPhi         ( Double_t &Phi, Double_t &SigmaPhi ) const ; //* phi
  int GetMomentum    ( Double_t &P, Double_t &SigmaP ) const ;   //* momentum
  int GetMass        ( Double_t &M, Double_t &SigmaM ) const ;   //* mass
  int GetDecayLength ( Double_t &L, Double_t &SigmaL ) const ;   //* decay length
  int GetDecayLengthXY ( Double_t &L, Double_t &SigmaL ) const ;   //* decay length in XY
   int GetLifeTime    ( Double_t &T, Double_t &SigmaT ) const ;   //* life time
  int GetR           ( Double_t &R, Double_t &SigmaR ) const ; //* R


  //*
  //*  MODIFIERS
  //*
  
  Double_t & X    () ;
  Double_t & Y    () ;
  Double_t & Z    () ;
  Double_t & Px   () ;
  Double_t & Py   () ;
  Double_t & Pz   () ;
  Double_t & E    () ;
  Double_t & S    () ;
  Int_t    & Q    () ;
  Double_t & Chi2 () ;
  Int_t    & NDF  () ;

  Double_t & Parameter ( int i ) ;
  Double_t & Covariance( int i ) ;
  Double_t & Covariance( int i, int j ) ;
  Double_t * Parameters () ;
  Double_t * CovarianceMatrix() ;

  //* 
  //* CONSTRUCTION OF THE PARTICLE BY ITS DAUGHTERS AND MOTHER
  //* USING THE KALMAN FILTER METHOD
  //*


  //* Add daughter to the particle 

  void AddDaughter( const AliKFParticle &Daughter );

  //* Add daughter via += operator: ex.{ D0; D0+=Pion; D0+= Kaon; }

  void operator +=( const AliKFParticle &Daughter );  

  //* Set production vertex 

  void SetProductionVertex( const AliKFParticle &Vtx );

  //* Set mass constraint 

  void SetMassConstraint( Double_t Mass, Double_t SigmaMass = 0  );
  
  //* Set no decay length for resonances

  void SetNoDecayLength();

  //* Everything in one go  

  void Construct( const AliKFParticle *vDaughters[], int NDaughters, 
		  const AliKFParticle *ProdVtx=0,   Double_t Mass=-1, Bool_t IsConstrained=0  );

  //*
  //*                   TRANSPORT
  //* 
  //*  ( main transportation parameter is S = SignedPath/Momentum )
  //*  ( parameters of decay & production vertices are stored locally )
  //*

  //* Transport the particle to its decay vertex 

  void TransportToDecayVertex();

  //* Transport the particle to its production vertex 

  void TransportToProductionVertex();

  //* Transport the particle close to xyz[] point 

  void TransportToPoint( const Double_t xyz[] );

  //* Transport the particle close to VVertex  

  void TransportToVertex( const AliVVertex &v );

  //* Transport the particle close to another particle p 

  void TransportToParticle( const AliKFParticle &p );

  //* Transport the particle on dS parameter (SignedPath/Momentum) 

  void TransportToDS( Double_t dS );

  //* Get dS to a certain space point 

  Double_t GetDStoPoint( const Double_t xyz[] ) const ;
  
  //* Get dS to other particle p (dSp for particle p also returned) 

  void GetDStoParticle( const AliKFParticle &p, 
			Double_t &DS, Double_t &DSp ) const ;
  
  //* Get dS to other particle p in XY-plane

  void GetDStoParticleXY( const AliKFParticleBase &p, 
			  Double_t &DS, Double_t &DSp ) const ;
  
  //* 
  //* OTHER UTILITIES
  //*


  //* Calculate distance from another object [cm]

  Double_t GetDistanceFromVertex( const Double_t vtx[] ) const ;
  Double_t GetDistanceFromVertex( const AliKFParticle &Vtx ) const ;
  Double_t GetDistanceFromVertex( const AliVVertex &Vtx ) const ;
  Double_t GetDistanceFromParticle( const AliKFParticle &p ) const ;

  //* Calculate sqrt(Chi2/ndf) deviation from another object
  //* ( v = [xyz]-vertex, Cv=[Cxx,Cxy,Cyy,Cxz,Cyz,Czz]-covariance matrix )

  Double_t GetDeviationFromVertex( const Double_t v[], const Double_t Cv[]=0 ) const ;
  Double_t GetDeviationFromVertex( const AliKFParticle &Vtx ) const ;
  Double_t GetDeviationFromVertex( const AliVVertex &Vtx ) const ;
  Double_t GetDeviationFromParticle( const AliKFParticle &p ) const ;
 
  //* Calculate distance from another object [cm] in XY-plane

  Bool_t GetDistanceFromVertexXY( const Double_t vtx[], Double_t &val, Double_t &err ) const ;
  Bool_t GetDistanceFromVertexXY( const Double_t vtx[], const Double_t Cv[], Double_t &val, Double_t &err ) const ;
  Bool_t GetDistanceFromVertexXY( const AliKFParticle &Vtx, Double_t &val, Double_t &err ) const ;
  Bool_t GetDistanceFromVertexXY( const AliVVertex &Vtx, Double_t &val, Double_t &err ) const ;

  Double_t GetDistanceFromVertexXY( const Double_t vtx[] ) const ;
  Double_t GetDistanceFromVertexXY( const AliKFParticle &Vtx ) const ;
  Double_t GetDistanceFromVertexXY( const AliVVertex &Vtx ) const ;
  Double_t GetDistanceFromParticleXY( const AliKFParticle &p ) const ;

  //* Calculate sqrt(Chi2/ndf) deviation from another object in XY plane
  //* ( v = [xyz]-vertex, Cv=[Cxx,Cxy,Cyy,Cxz,Cyz,Czz]-covariance matrix )

  Double_t GetDeviationFromVertexXY( const Double_t v[], const Double_t Cv[]=0 ) const ;
  Double_t GetDeviationFromVertexXY( const AliKFParticle &Vtx ) const ;
  Double_t GetDeviationFromVertexXY( const AliVVertex &Vtx ) const ;
  Double_t GetDeviationFromParticleXY( const AliKFParticle &p ) const ;

  //* Calculate opennig angle between two particles

  Double_t GetAngle  ( const AliKFParticle &p ) const ;
  Double_t GetAngleXY( const AliKFParticle &p ) const ;
  Double_t GetAngleRZ( const AliKFParticle &p ) const ;

  //* Subtract the particle from the vertex  

  void SubtractFromVertex( AliKFParticle &v ) const ;

  //* Special method for creating gammas

  void ConstructGamma( const AliKFParticle &daughter1,
		       const AliKFParticle &daughter2  );
  
    // * Pseudo Proper Time of decay = (r*pt) / |pt| * M/|pt|
    // @primVertex - primary vertex
    // @mass - mass of the mother particle (in the case of "Hb -> JPsi" it would be JPsi mass)
    // @*timeErr2 - squared error of the decay time. If timeErr2 = 0 it isn't calculated
  Double_t GetPseudoProperDecayTime( const AliKFParticle &primVertex, const Double_t& mass, Double_t* timeErr2 = 0 ) const;
  
 protected: 
  
  //*
  //*  INTERNAL STUFF
  //* 

  //* Method to access ALICE field 
 
  static Double_t GetFieldAlice() { return fgBz;}

  
  //* Other methods required by the abstract AliKFParticleBase class 
  
  void GetFieldValue( const Double_t xyz[], Double_t B[] ) const ;
  void GetDStoParticle( const AliKFParticleBase &p, Double_t &DS, Double_t &DSp )const ;
  void Transport( Double_t dS, Double_t P[], Double_t C[] ) const ;
  static void GetExternalTrackParam( const AliKFParticleBase &p, Double_t &X, Double_t &Alpha, Double_t P[5]  ) ;

  //void GetDStoParticleALICE( const AliKFParticleBase &p, Double_t &DS, Double_t &DS1 ) const;


 private:

  static Double_t fgBz;  //* Bz compoment of the magnetic field

  ClassDef( AliKFParticle, 1 );

};



//---------------------------------------------------------------------
//
//     Inline implementation of the AliKFParticle methods
//
//---------------------------------------------------------------------


inline void AliKFParticle::SetField( Double_t Bz )
{ 
  fgBz = Bz;
}

inline AliKFParticle::AliKFParticle( const AliKFParticle &d1, 
				     const AliKFParticle &d2, 
				     const AliKFParticle &d3 )
{
  AliKFParticle mother;
  mother+= d1;
  mother+= d2;
  mother+= d3;
  *this = mother;
}

inline AliKFParticle::AliKFParticle( const AliKFParticle &d1, 
				     const AliKFParticle &d2, 
				     const AliKFParticle &d3, 
				     const AliKFParticle &d4 )
{
  AliKFParticle mother;
  mother+= d1;
  mother+= d2;
  mother+= d3;
  mother+= d4;
  *this = mother;
}


inline void AliKFParticle::Initialize()
{ 
  AliKFParticleBase::Initialize(); 
}

inline void AliKFParticle::SetVtxGuess( Double_t x, Double_t y, Double_t z )
{
  AliKFParticleBase::SetVtxGuess(x,y,z);
}  

inline Double_t AliKFParticle::GetX    () const 
{ 
  return AliKFParticleBase::GetX();    
}

inline Double_t AliKFParticle::GetY    () const 
{ 
  return AliKFParticleBase::GetY();    
}

inline Double_t AliKFParticle::GetZ    () const 
{ 
  return AliKFParticleBase::GetZ();    
}

inline Double_t AliKFParticle::GetPx   () const 
{ 
  return AliKFParticleBase::GetPx();   
}

inline Double_t AliKFParticle::GetPy   () const 
{ 
  return AliKFParticleBase::GetPy();   
}

inline Double_t AliKFParticle::GetPz   () const 
{ 
  return AliKFParticleBase::GetPz();   
}

inline Double_t AliKFParticle::GetE    () const 
{ 
  return AliKFParticleBase::GetE();    
}

inline Double_t AliKFParticle::GetS    () const 
{ 
  return AliKFParticleBase::GetS();    
}

inline Int_t    AliKFParticle::GetQ    () const 
{ 
  return AliKFParticleBase::GetQ();    
}

inline Double_t AliKFParticle::GetChi2 () const 
{ 
  return AliKFParticleBase::GetChi2(); 
}

inline Int_t    AliKFParticle::GetNDF  () const 
{ 
  return AliKFParticleBase::GetNDF();  
}

inline Double_t AliKFParticle::GetParameter ( int i ) const 
{ 
  return AliKFParticleBase::GetParameter(i);  
}

inline Double_t AliKFParticle::GetCovariance( int i ) const 
{ 
  return AliKFParticleBase::GetCovariance(i); 
}

inline Double_t AliKFParticle::GetCovariance( int i, int j ) const 
{ 
  return AliKFParticleBase::GetCovariance(i,j);
}


inline Double_t AliKFParticle::GetP    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetMomentum( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetPt   () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetPt( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetEta   () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetEta( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetPhi   () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetPhi( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetMomentum    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetMomentum( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetMass        () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetMass( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetDecayLength () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetDecayLength( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetDecayLengthXY () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetDecayLengthXY( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetLifeTime    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetLifeTime( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetR   () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetR( par, err ) ) return 0;
  else return par;
}

inline Double_t AliKFParticle::GetErrX           () const 
{
  return TMath::Sqrt(TMath::Abs( GetCovariance(0,0) ));
}

inline Double_t AliKFParticle::GetErrY           () const 
{
  return TMath::Sqrt(TMath::Abs( GetCovariance(1,1) ));
}

inline Double_t AliKFParticle::GetErrZ           () const 
{
  return TMath::Sqrt(TMath::Abs( GetCovariance(2,2) ));
}

inline Double_t AliKFParticle::GetErrPx          () const 
{
  return TMath::Sqrt(TMath::Abs( GetCovariance(3,3) ));
}

inline Double_t AliKFParticle::GetErrPy          () const 
{
  return TMath::Sqrt(TMath::Abs( GetCovariance(4,4) ));
}

inline Double_t AliKFParticle::GetErrPz          () const 
{
  return TMath::Sqrt(TMath::Abs( GetCovariance(5,5) ));
}

inline Double_t AliKFParticle::GetErrE           () const 
{
  return TMath::Sqrt(TMath::Abs( GetCovariance(6,6) ));
}

inline Double_t AliKFParticle::GetErrS           () const 
{
  return TMath::Sqrt(TMath::Abs( GetCovariance(7,7) ));
}

inline Double_t AliKFParticle::GetErrP    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetMomentum( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrPt    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetPt( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrEta    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetEta( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrPhi    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetPhi( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrMomentum    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetMomentum( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrMass        () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetMass( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrDecayLength () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetDecayLength( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrDecayLengthXY () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetDecayLengthXY( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrLifeTime    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetLifeTime( par, err ) ) return 1.e10;
  else return err;
}

inline Double_t AliKFParticle::GetErrR    () const
{
  Double_t par, err;
  if( AliKFParticleBase::GetR( par, err ) ) return 1.e10;
  else return err;
}


inline int AliKFParticle::GetP( Double_t &P, Double_t &SigmaP ) const 
{
  return AliKFParticleBase::GetMomentum( P, SigmaP );
}

inline int AliKFParticle::GetPt( Double_t &Pt, Double_t &SigmaPt ) const 
{
  return AliKFParticleBase::GetPt( Pt, SigmaPt );
}

inline int AliKFParticle::GetEta( Double_t &Eta, Double_t &SigmaEta ) const 
{
  return AliKFParticleBase::GetEta( Eta, SigmaEta );
}

inline int AliKFParticle::GetPhi( Double_t &Phi, Double_t &SigmaPhi ) const 
{
  return AliKFParticleBase::GetPhi( Phi, SigmaPhi );
}

inline int AliKFParticle::GetMomentum( Double_t &P, Double_t &SigmaP ) const 
{
  return AliKFParticleBase::GetMomentum( P, SigmaP );
}

inline int AliKFParticle::GetMass( Double_t &M, Double_t &SigmaM ) const 
{
  return AliKFParticleBase::GetMass( M, SigmaM );
}

inline int AliKFParticle::GetDecayLength( Double_t &L, Double_t &SigmaL ) const 
{
  return AliKFParticleBase::GetDecayLength( L, SigmaL );
}

inline int AliKFParticle::GetDecayLengthXY( Double_t &L, Double_t &SigmaL ) const 
{
  return AliKFParticleBase::GetDecayLengthXY( L, SigmaL );
}

inline int AliKFParticle::GetLifeTime( Double_t &T, Double_t &SigmaT ) const 
{
  return AliKFParticleBase::GetLifeTime( T, SigmaT );
}

inline int AliKFParticle::GetR( Double_t &R, Double_t &SigmaR ) const 
{
  return AliKFParticleBase::GetR( R, SigmaR );
}

inline Double_t & AliKFParticle::X() 
{ 
  return AliKFParticleBase::X();    
}

inline Double_t & AliKFParticle::Y()
{ 
  return AliKFParticleBase::Y();    
}

inline Double_t & AliKFParticle::Z() 
{ 
  return AliKFParticleBase::Z();    
}

inline Double_t & AliKFParticle::Px() 
{ 
  return AliKFParticleBase::Px();   
}

inline Double_t & AliKFParticle::Py() 
{ 
  return AliKFParticleBase::Py();   
}

inline Double_t & AliKFParticle::Pz() 
{ 
  return AliKFParticleBase::Pz();   
}

inline Double_t & AliKFParticle::E() 
{ 
  return AliKFParticleBase::E();    
}

inline Double_t & AliKFParticle::S() 
{ 
  return AliKFParticleBase::S();    
}

inline Int_t    & AliKFParticle::Q() 
{ 
  return AliKFParticleBase::Q();    
}

inline Double_t & AliKFParticle::Chi2() 
{ 
  return AliKFParticleBase::Chi2(); 
}

inline Int_t    & AliKFParticle::NDF() 
{ 
  return AliKFParticleBase::NDF();  
}

inline Double_t & AliKFParticle::Parameter ( int i )        
{ 
  return AliKFParticleBase::Parameter(i);
}

inline Double_t & AliKFParticle::Covariance( int i )        
{ 
  return AliKFParticleBase::Covariance(i);
}

inline Double_t & AliKFParticle::Covariance( int i, int j ) 
{ 
  return AliKFParticleBase::Covariance(i,j); 
}

inline Double_t * AliKFParticle::Parameters ()
{
  return fP;
}

inline Double_t * AliKFParticle::CovarianceMatrix()
{
  return fC;
}


inline void AliKFParticle::operator +=( const AliKFParticle &Daughter )
{
  AliKFParticleBase::operator +=( Daughter );
}
  

inline void AliKFParticle::AddDaughter( const AliKFParticle &Daughter )
{
  AliKFParticleBase::AddDaughter( Daughter );
}

inline void AliKFParticle::SetProductionVertex( const AliKFParticle &Vtx )
{
  AliKFParticleBase::SetProductionVertex( Vtx );
}

inline void AliKFParticle::SetMassConstraint( Double_t Mass, Double_t SigmaMass )
{
  AliKFParticleBase::SetMassConstraint( Mass, SigmaMass );
}
    
inline void AliKFParticle::SetNoDecayLength()
{
  AliKFParticleBase::SetNoDecayLength();
}

inline void AliKFParticle::Construct( const AliKFParticle *vDaughters[], int NDaughters, 
			       const AliKFParticle *ProdVtx,   Double_t Mass, Bool_t IsConstrained  )
{    
  AliKFParticleBase::Construct( ( const AliKFParticleBase**)vDaughters, NDaughters, 
			 ( const AliKFParticleBase*)ProdVtx, Mass, IsConstrained );
}

inline void AliKFParticle::TransportToDecayVertex()
{ 
  AliKFParticleBase::TransportToDecayVertex(); 
}

inline void AliKFParticle::TransportToProductionVertex()
{
  AliKFParticleBase::TransportToProductionVertex();
}

inline void AliKFParticle::TransportToPoint( const Double_t xyz[] )
{ 
  TransportToDS( GetDStoPoint(xyz) );
}

inline void AliKFParticle::TransportToVertex( const AliVVertex &v )
{       
  TransportToPoint( AliKFParticle(v).fP );
}

inline void AliKFParticle::TransportToParticle( const AliKFParticle &p )
{ 
  Double_t dS, dSp;
  GetDStoParticle( p, dS, dSp );
  TransportToDS( dS );
}

inline void AliKFParticle::TransportToDS( Double_t dS )
{
  AliKFParticleBase::TransportToDS( dS );
} 

inline Double_t AliKFParticle::GetDStoPoint( const Double_t xyz[] ) const 
{
  return AliKFParticleBase::GetDStoPointBz( GetFieldAlice(), xyz );
}

  
inline void AliKFParticle::GetDStoParticle( const AliKFParticle &p, 
					    Double_t &DS, Double_t &DSp ) const 
{
  GetDStoParticleXY( p, DS, DSp );
}


inline Double_t AliKFParticle::GetDistanceFromVertex( const Double_t vtx[] ) const
{
  return AliKFParticleBase::GetDistanceFromVertex( vtx );
}

inline Double_t AliKFParticle::GetDeviationFromVertex( const Double_t v[], 
						       const Double_t Cv[] ) const
{
  return AliKFParticleBase::GetDeviationFromVertex( v, Cv);
}

inline Double_t AliKFParticle::GetDistanceFromVertex( const AliKFParticle &Vtx ) const
{
  return AliKFParticleBase::GetDistanceFromVertex( Vtx );
}

inline Double_t AliKFParticle::GetDeviationFromVertex( const AliKFParticle &Vtx ) const
{
  return AliKFParticleBase::GetDeviationFromVertex( Vtx );
}

inline Double_t AliKFParticle::GetDistanceFromVertex( const AliVVertex &Vtx ) const
{
  return GetDistanceFromVertex( AliKFParticle(Vtx) );
}

inline Double_t AliKFParticle::GetDeviationFromVertex( const AliVVertex &Vtx ) const
{
  return GetDeviationFromVertex( AliKFParticle(Vtx) );
}
 
inline Double_t AliKFParticle::GetDistanceFromParticle( const AliKFParticle &p ) const 
{
  return AliKFParticleBase::GetDistanceFromParticle( p );
}

inline Double_t AliKFParticle::GetDeviationFromParticle( const AliKFParticle &p ) const 
{
  return AliKFParticleBase::GetDeviationFromParticle( p );
}

inline void AliKFParticle::SubtractFromVertex( AliKFParticle &v ) const 
{
  AliKFParticleBase::SubtractFromVertex( v );
}

inline void AliKFParticle::GetFieldValue( const Double_t * /*xyz*/, Double_t B[] ) const 
{    
  B[0] = B[1] = 0;
  B[2] = GetFieldAlice();
}

inline void AliKFParticle::GetDStoParticle( const AliKFParticleBase &p, 
					    Double_t &DS, Double_t &DSp )const
{
  GetDStoParticleXY( p, DS, DSp );
}

inline void AliKFParticle::GetDStoParticleXY( const AliKFParticleBase &p, 
				       Double_t &DS, Double_t &DSp ) const
{ 
  AliKFParticleBase::GetDStoParticleBz( GetFieldAlice(), p, DS, DSp ) ;
  //GetDStoParticleALICE( p, DS, DSp ) ;
}

inline void AliKFParticle::Transport( Double_t dS, Double_t P[], Double_t C[] ) const 
{
  AliKFParticleBase::TransportBz( GetFieldAlice(), dS, P, C );
}

inline void AliKFParticle::ConstructGamma( const AliKFParticle &daughter1,
					   const AliKFParticle &daughter2  )
{
  AliKFParticleBase::ConstructGammaBz( daughter1, daughter2, GetFieldAlice() );
}

#endif 
