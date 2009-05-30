//-*- Mode: C++ -*-
// @(#) $Id: AliHLT3DTrackParam.h 31935 2009-04-13 20:57:12Z sgorbuno $
// ***************************************************************************
// This file is property of and copyright by the ALICE HLT Project           *
// ALICE Experiment at CERN, All rights reserved.                            *
// See cxx source for full Copyright notice                                  *
//                                                                           *
// AliHLT3DTrackParam class is under development and currently not in use    *
//                                                                           *
//****************************************************************************

#ifndef ALIHLT3DTRACKPARAM_H
#define ALIHLT3DTRACKPARAM_H

#include "TObject.h"

/**
 * @class AliHLT3DTrackParam
 */
class AliHLT3DTrackParam : public TObject
{
  public:

    //*
    //*  INITIALIZATION
    //*

    //* Constructor

    AliHLT3DTrackParam(): fChi2( 0 ), fNDF( 0 ), fSignQ( 0 ) {}

    //* Destructor (empty)

    virtual ~AliHLT3DTrackParam() {}

    //*
    //*  ACCESSORS
    //*


    //* Simple accessors

    double GetX()      const { return fParam[0]; }
    double GetY()      const { return fParam[1]; }
    double GetZ()      const { return fParam[2]; }
    double GetPx()     const { return fParam[3]; }
    double GetPy()     const { return fParam[4]; }
    double GetPz()     const { return fParam[5]; }
    double GetChi2()   const { return fChi2;  }
    int    GetNDF()    const { return fNDF;   }
    int    GetCharge() const { return fSignQ; }

    double GetParameter ( int i ) const { return fParam[i]; }
    double GetCovariance( int i ) const { return fCov[i]; }
    double GetCovariance( int i, int j ) const { return fCov[( j<=i ) ? i*( i+1 )/2+j :j*( j+1 )/2+i]; }

    //*
    //* Accessors
    //*

    const double *Param()  const { return fParam; }
    const double *Cov() const   { return fCov;   }
    double X()      const  { return fParam[0]; }
    double Y()      const  { return fParam[1]; }
    double Z()      const  { return fParam[2]; }
    double Px()     const  { return fParam[3]; }
    double Py()     const  { return fParam[4]; }
    double Pz()     const  { return fParam[5]; }
    double Chi2()   const  { return fChi2;  }
    int    NDF()    const  { return fNDF;   }
    int    Charge()  const { return fSignQ; }

    //* Accessors with calculations( &value, &estimated sigma )
    //* error flag returned (0 means no error during calculations)


    //*
    //*  MODIFIERS
    //*

    void SetParam( int i, double v )  { fParam[i] = v; }
    void SetCov( int i, double v ) { fCov[i] = v;   }
    void SetX( double v )      { fParam[0] = v; }
    void SetY( double v )      { fParam[1] = v; }
    void SetZ( double v )      { fParam[2] = v; }
    void SetPx( double v )     { fParam[3] = v; }
    void SetPy( double v )     { fParam[4] = v; }
    void SetPz( double v )     { fParam[5] = v; }
    void SetChi2( double v )   { fChi2 = v;  }
    void SetNDF( int v )       { fNDF = v;   }
    void SetCharge( int v )    { fSignQ = v; }


    //*
    //*  UTILITIES
    //*

    //* Transport utilities

    double GetDStoPoint( double Bz, const double xyz[3], const double *T0 = 0 ) const;

    void TransportToDS( double Bz, double DS, double *T0 = 0 );

    void TransportToPoint( double Bz, const double xyz[3], double *T0 = 0 ) {
      TransportToDS( Bz, GetDStoPoint( Bz, xyz, T0 ), T0 ) ;
    }

    void TransportToPoint( double Bz, double x, double y, double z, const double *T0 = 0 ) {
      double xyz[3] = {x, y, z};
      TransportToPoint( Bz, xyz, T0 );
    }

    //* Fit utilities

    void InitializeCovarianceMatrix();

    void GetGlueMatrix( const double p[3], double G[6], const double *T0 = 0  ) const ;

    void Filter( const double m[3], const double V[6], const double G[6] );

    //* Other utilities

    void SetDirection( double Direction[3] );

    void RotateCoordinateSystem( double alpha );

    void Get5Parameters( double alpha, double T[6], double C[15] ) const;

  protected:

    double fParam[6]; // Parameters ( x, y, z, px, py, pz ): 3-position and 3-momentum
    double fCov[21];  // Covariance matrix
    double fChi2;     // Chi^2
    int    fNDF;      // Number of Degrees of Freedom
    int    fSignQ;    // Charge

    ClassDef( AliHLT3DTrackParam, 1 );

};


#endif
