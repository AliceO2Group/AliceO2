//-*- Mode: C++ -*-
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMPolynomialField_H
#define AliHLTTPCGMPolynomialField_H

#include "AliHLTTPCCADef.h"

/**
 * @class AliHLTTPCGMPolynomialField
 *
 */

class AliHLTTPCGMPolynomialField
{
  public:
	AliHLTTPCGMPolynomialField() : fNominalBz(0.)
	{
		Reset();
	}

	void Reset();

	void Set(float nominalBz,
	         const float *Bx, const float *By, const float *Bz,
	         const float *TrdBx, const float *TrdBy, const float *TrdBz);

	GPUdi() float GetNominalBz() const { return fNominalBz; }

	GPUd() void GetField(float x, float y, float z, float B[3]) const;
	GPUd() float GetFieldBz(float x, float y, float z) const;

	GPUd() void GetFieldTrd(float x, float y, float z, float B[3]) const;
	GPUd() float GetFieldTrdBz(float x, float y, float z) const;

	void Print() const;

	static const int fkM = 10;    // number of coefficients
	static const int fkTrdM = 20; // number of coefficients for TRD field

	GPUd() static void GetPolynoms(float x, float y, float z, float f[fkM]);
	GPUd() static void GetPolynomsTrd(float x, float y, float z, float f[fkTrdM]);

	const float *GetCoeffBx() const { return fBx; }
	const float *GetCoeffBy() const { return fBy; }
	const float *GetCoeffBz() const { return fBz; }

	const float *GetCoeffTrdBx() const { return fTrdBx; }
	const float *GetCoeffTrdBy() const { return fTrdBy; }
	const float *GetCoeffTrdBz() const { return fTrdBz; }

  private:
	float fNominalBz; // nominal constant field value in [kG * 2.99792458E-4 GeV/c/cm]
	float fBx[fkM];   // polynomial coefficients
	float fBy[fkM];
	float fBz[fkM];
	float fTrdBx[fkTrdM]; // polynomial coefficients
	float fTrdBy[fkTrdM];
	float fTrdBz[fkTrdM];
};

inline void AliHLTTPCGMPolynomialField::Reset()
{
	fNominalBz = 0.f;
	for (int i = 0; i < fkM; i++)
	{
		fBx[i] = 0.f;
		fBy[i] = 0.f;
		fBz[i] = 0.f;
	}
	for (int i = 0; i < fkTrdM; i++)
	{
		fTrdBx[i] = 0.f;
		fTrdBy[i] = 0.f;
		fTrdBz[i] = 0.f;
	}
}

inline void AliHLTTPCGMPolynomialField::Set(float nominalBz, const float *Bx, const float *By, const float *Bz, const float *TrdBx, const float *TrdBy, const float *TrdBz)
{
	if (!Bx || !By || !Bz || !TrdBx || !TrdBy || !TrdBz)
	{
		Reset();
		return;
	}
	fNominalBz = nominalBz;
	for (int i = 0; i < fkM; i++)
	{
		fBx[i] = Bx[i];
		fBy[i] = By[i];
		fBz[i] = Bz[i];
	}
	for (int i = 0; i < fkTrdM; i++)
	{
		fTrdBx[i] = TrdBx[i];
		fTrdBy[i] = TrdBy[i];
		fTrdBz[i] = TrdBz[i];
	}
}


GPUdi() void AliHLTTPCGMPolynomialField::GetPolynoms( float x, float y, float z, float f[fkM] )
{
	f[0]=1.f;
	f[1]=x;   f[2]=y;   f[3]=z;
	f[4]=x*x; f[5]=x*y; f[6]=x*z; f[7]=y*y; f[8]=y*z; f[9]=z*z;
}

GPUdi() void AliHLTTPCGMPolynomialField::GetField( float x, float y, float z, float B[3] ) const
{
	const float f[fkM] = { 1.f, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
	float bx = 0.f, by = 0.f, bz = 0.f;
	for( int i=0; i<fkM; i++){
		bx += fBx[i]*f[i];
		by += fBy[i]*f[i];
		bz += fBz[i]*f[i];
	}
	B[0] = bx;
	B[1] = by;
	B[2] = bz;
}

GPUdi() float AliHLTTPCGMPolynomialField::GetFieldBz( float x, float y, float z ) const
{
	const float f[fkM] = { 1.f, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
	float bz = 0.f;
	for( int i=0; i<fkM; i++){
		bz += fBz[i]*f[i];
	}
	return bz;
}

GPUdi() void AliHLTTPCGMPolynomialField::GetPolynomsTrd( float x, float y, float z, float f[fkTrdM] )
{
	float xx=x*x, xy=x*y, xz=x*z, yy=y*y, yz=y*z, zz=z*z;
	f[ 0]=1.f;
	f[ 1]=x;    f[ 2]=y;    f[ 3]=z;
	f[ 4]=xx;   f[ 5]=xy;   f[ 6]=xz;   f[ 7]=yy;   f[ 8]=yz;   f[ 9]=zz;
	f[10]=x*xx; f[11]=x*xy; f[12]=x*xz; f[13]=x*yy; f[14]=x*yz; f[15]=x*zz;
	f[16]=y*yy; f[17]=y*yz; f[18]=y*zz;
	f[19]=z*zz;
}

GPUdi() void AliHLTTPCGMPolynomialField::GetFieldTrd( float x, float y, float z, float B[3] ) const
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

GPUdi() float AliHLTTPCGMPolynomialField::GetFieldTrdBz( float x, float y, float z ) const
{
	float f[fkTrdM];
	GetPolynomsTrd(x,y,z,f);
	float bz = 0.f;
	for( int i=0; i<fkTrdM; i++){
		bz += fTrdBz[i]*f[i];
	}
	return bz;
}

#endif
