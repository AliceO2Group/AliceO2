//-*- Mode: C++ -*-
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliGPUTPCGMPolynomialField_H
#define AliGPUTPCGMPolynomialField_H

#include "AliGPUTPCDef.h"

/**
 * @class AliGPUTPCGMPolynomialField
 *
 */

class AliGPUTPCGMPolynomialField
{
  public:
	AliGPUTPCGMPolynomialField() : fNominalBz(0.f)
	{
		Reset();
	}

	void Reset();

        void SetFieldNominal( float nominalBz );

        void SetFieldTpc( const float *Bx, const float *By, const float *Bz );
        void SetFieldTrd( const float *Bx, const float *By, const float *Bz );
        void SetFieldIts( const float *Bx, const float *By, const float *Bz );

	GPUdi() float GetNominalBz() const { return fNominalBz; }

	GPUd() void GetField(float x, float y, float z, float B[3]) const;
	GPUd() float GetFieldBz(float x, float y, float z) const;

	GPUd() void GetFieldTrd(float x, float y, float z, float B[3]) const;
	GPUd() float GetFieldTrdBz(float x, float y, float z) const;

        GPUd() void GetFieldIts(float x, float y, float z, float B[3]) const;
	GPUd() float GetFieldItsBz(float x, float y, float z) const;

	void Print() const;

	static CONSTEXPR int fkTpcM = 10;    // number of coefficients
	static CONSTEXPR int fkTrdM = 20; // number of coefficients for the TRD field
	static CONSTEXPR int fkItsM = 10; // number of coefficients for the ITS field

	GPUd() static void GetPolynomsTpc(float x, float y, float z, float f[fkTpcM]);
	GPUd() static void GetPolynomsTrd(float x, float y, float z, float f[fkTrdM]);
	GPUd() static void GetPolynomsIts(float x, float y, float z, float f[fkItsM]);

	const float *GetCoeffTpcBx() const { return fTpcBx; }
	const float *GetCoeffTpcBy() const { return fTpcBy; }
	const float *GetCoeffTpcBz() const { return fTpcBz; }

	const float *GetCoeffTrdBx() const { return fTrdBx; }
	const float *GetCoeffTrdBy() const { return fTrdBy; }
	const float *GetCoeffTrdBz() const { return fTrdBz; }

	const float *GetCoeffItsBx() const { return fItsBx; }
	const float *GetCoeffItsBy() const { return fItsBy; }
	const float *GetCoeffItsBz() const { return fItsBz; }

  private:
	float fNominalBz; // nominal constant field value in [kG * 2.99792458E-4 GeV/c/cm]
	float fTpcBx[fkTpcM];   // polynomial coefficients
	float fTpcBy[fkTpcM];
	float fTpcBz[fkTpcM];
	float fTrdBx[fkTrdM]; // polynomial coefficients
	float fTrdBy[fkTrdM];
	float fTrdBz[fkTrdM];
	float fItsBx[fkItsM]; // polynomial coefficients
	float fItsBy[fkItsM];
	float fItsBz[fkItsM];
};

inline void AliGPUTPCGMPolynomialField::Reset()
{
	fNominalBz = 0.f;
	for (int i = 0; i < fkTpcM; i++)
	{
		fTpcBx[i] = 0.f;
		fTpcBy[i] = 0.f;
		fTpcBz[i] = 0.f;
	}
	for (int i = 0; i < fkTrdM; i++)
	{
		fTrdBx[i] = 0.f;
		fTrdBy[i] = 0.f;
		fTrdBz[i] = 0.f;
	}
	for (int i = 0; i < fkItsM; i++)
	{
		fItsBx[i] = 0.f;
		fItsBy[i] = 0.f;
		fItsBz[i] = 0.f;
	}
}


inline void AliGPUTPCGMPolynomialField::SetFieldNominal( float nominalBz )
{
	fNominalBz = nominalBz;
}

inline void AliGPUTPCGMPolynomialField::SetFieldTpc( const float *Bx, const float *By, const float *Bz )
{
	if ( Bx && By && Bz )
	{
	  for (int i = 0; i < fkTpcM; i++)
	    {
	      fTpcBx[i] = Bx[i];
	      fTpcBy[i] = By[i];
	      fTpcBz[i] = Bz[i];
	    }
	}
}

inline void AliGPUTPCGMPolynomialField::SetFieldTrd( const float *Bx, const float *By, const float *Bz )
{
	if ( Bx && By && Bz ){
	  for (int i = 0; i < fkTrdM; i++)
	    {
	      fTrdBx[i] = Bx[i];
	      fTrdBy[i] = By[i];
	      fTrdBz[i] = Bz[i];
	    }
	}
}

inline void AliGPUTPCGMPolynomialField::SetFieldIts( const float *Bx, const float *By, const float *Bz )
{
	if ( Bx && By && Bz ){
	  for (int i = 0; i < fkItsM; i++)
	    {
	      fItsBx[i] = Bx[i];
	      fItsBy[i] = By[i];
	      fItsBz[i] = Bz[i];
	    }
	}
}


GPUdi() void AliGPUTPCGMPolynomialField::GetPolynomsTpc( float x, float y, float z, float f[fkTpcM] )
{
	f[0]=1.f;
	f[1]=x;   f[2]=y;   f[3]=z;
	f[4]=x*x; f[5]=x*y; f[6]=x*z; f[7]=y*y; f[8]=y*z; f[9]=z*z;
}

GPUdi() void AliGPUTPCGMPolynomialField::GetField( float x, float y, float z, float B[3] ) const
{
        const float *fBxS = &fTpcBx[1];
        const float *fByS = &fTpcBy[1];
        const float *fBzS = &fTpcBz[1];

	const float f[fkTpcM-1] = { x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
	float bx = fTpcBx[0], by = fTpcBy[0], bz = fTpcBz[0];
	for( int i=fkTpcM-1; i--;) {
	  //for (int i=0;i<fkTpcM-1; i++){
		bx += fBxS[i]*f[i];
		by += fByS[i]*f[i];
		bz += fBzS[i]*f[i];
	}
	B[0] = bx;
	B[1] = by;
	B[2] = bz;
}

GPUdi() float AliGPUTPCGMPolynomialField::GetFieldBz( float x, float y, float z ) const
{
        const float *fBzS = &fTpcBz[1];

	const float f[fkTpcM-1] = { x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
	float bz = fTpcBz[0];
	for( int i=fkTpcM-1; i--;) {
		bz += fBzS[i]*f[i];
	}
	return bz;
}

GPUdi() void AliGPUTPCGMPolynomialField::GetPolynomsTrd( float x, float y, float z, float f[fkTrdM] )
{
	float xx=x*x, xy=x*y, xz=x*z, yy=y*y, yz=y*z, zz=z*z;
	f[ 0]=1.f;
	f[ 1]=x;    f[ 2]=y;    f[ 3]=z;
	f[ 4]=xx;   f[ 5]=xy;   f[ 6]=xz;   f[ 7]=yy;   f[ 8]=yz;   f[ 9]=zz;
	f[10]=x*xx; f[11]=x*xy; f[12]=x*xz; f[13]=x*yy; f[14]=x*yz; f[15]=x*zz;
	f[16]=y*yy; f[17]=y*yz; f[18]=y*zz;
	f[19]=z*zz;
}

GPUdi() void AliGPUTPCGMPolynomialField::GetFieldTrd( float x, float y, float z, float B[3] ) const
{
	float f[fkTrdM];
	GetPolynomsTrd(x,y,z,f);
	float bx = 0.f, by = 0.f, bz = 0.f;
	for( int i=0; i<fkTrdM; i++){
		bx += fTrdBx[i]*f[i];
		by += fTrdBy[i]*f[i];
		bz += fTrdBz[i]*f[i];
	}
	B[0] = bx;
	B[1] = by;
	B[2] = bz;
}

GPUdi() float AliGPUTPCGMPolynomialField::GetFieldTrdBz( float x, float y, float z ) const
{
	float f[fkTrdM];
	GetPolynomsTrd(x,y,z,f);
	float bz = 0.f;
	for( int i=0; i<fkTrdM; i++){
		bz += fTrdBz[i]*f[i];
	}
	return bz;
}

GPUdi() void AliGPUTPCGMPolynomialField::GetPolynomsIts( float x, float y, float z, float f[fkItsM] )
{
	float xx=x*x, xy=x*y, xz=x*z, yy=y*y, yz=y*z, zz=z*z;
	f[ 0]=1.f;
	f[ 1]=x;    f[ 2]=y;    f[ 3]=z;
	f[ 4]=xx;   f[ 5]=xy;   f[ 6]=xz;   f[ 7]=yy;   f[ 8]=yz;   f[ 9]=zz;
	/*
	f[10]=x*xx; f[11]=x*xy; f[12]=x*xz; f[13]=x*yy; f[14]=x*yz; f[15]=x*zz;
	f[16]=y*yy; f[17]=y*yz; f[18]=y*zz;
	f[19]=z*zz;
	*/
}

GPUdi() void AliGPUTPCGMPolynomialField::GetFieldIts( float x, float y, float z, float B[3] ) const
{
        const float *fBxS = &fItsBx[1];
        const float *fByS = &fItsBy[1];
        const float *fBzS = &fItsBz[1];

	const float f[fkItsM-1] = { x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
	float bx = fItsBx[0], by = fItsBy[0], bz = fItsBz[0];
	for( int i=fkItsM-1; i--;) {	  
		bx += fBxS[i]*f[i];
		by += fByS[i]*f[i];
		bz += fBzS[i]*f[i];
	}
	B[0] = bx;
	B[1] = by;
	B[2] = bz;
}

GPUdi() float AliGPUTPCGMPolynomialField::GetFieldItsBz( float x, float y, float z ) const
{
        const float *fBzS = &fItsBz[1];

	const float f[fkItsM-1] = { x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
	float bz = fItsBz[0];
	for( int i=fkItsM-1; i--;) {	  
		bz += fBzS[i]*f[i];
	}
	return bz;
}

#endif
