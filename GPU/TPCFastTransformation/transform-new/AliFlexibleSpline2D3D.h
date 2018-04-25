#ifndef AliFlexibleSpline2D3D_H
#define AliFlexibleSpline2D3D_H

/********************************************************************* 
 * This file is property of and copyright by the ALICE HLT Project.  * 
 * ALICE Experiment at CERN, All rights reserved.                    *
 * See cxx source for full Copyright notice                          *
 * Authors: sergey.gorbunov@cern.ch                                  *
 *                                                                   *
 *********************************************************************/

/*
  The class presents spline interpolation for 2D->3D function (u,v)->(x,y,z) 
*/

#include "AliFlexibleSpline1D.h"

class AliFlexibleSpline2D3D{
 public:

  AliFlexibleSpline2D3D( int nKnotsU, const float knotsU[], int nAxisTicksU,
			 int nKnotsV, const float knotsV[], int nAxisTicksV );

  ~AliFlexibleSpline2D3D() {}

  int Init( int nKnotsU, const float knotsU[], int nAxisTicksU,
	    int nKnotsV, const float knotsV[], int nAxisTicksV  );


  void GetSpline( const float *data, float u, float v, float &x, float &y, float &z ) const;
  void GetSplineVec( const float *data, float u, float v, float &x, float &y, float &z ) const;

  const AliFlexibleSpline1D& GetGridU() const { return *(reinterpret_cast<const AliFlexibleSpline1D *>( fContent )); }

  const AliFlexibleSpline1D& GetGridV() const { return *(reinterpret_cast<const AliFlexibleSpline1D *>( fContent + fGridVPointer)); }

  // --------------------------------------------------------------------------------
  // -- Size methods

  static unsigned int EstimateSize( int nKnotsU, int nAxisTicksU,
				    int nKnotsV, int nAxisTicksV ); 
  
  unsigned int GetSize() const { 
    return fContent -  reinterpret_cast<const unsigned char *>(this) + fContentSize; 
  }

 private:

  /** copy constructor prohibited */
  AliFlexibleSpline2D3D(const AliFlexibleSpline2D3D&);
  /** assignment operator prohibited */
  AliFlexibleSpline2D3D& operator=(const AliFlexibleSpline2D3D&);
    
  AliFlexibleSpline1D& GridU(){ return *(reinterpret_cast<AliFlexibleSpline1D *>( fContent )); }
  AliFlexibleSpline1D& GridV(){ return *(reinterpret_cast<AliFlexibleSpline1D *>( fContent + fGridVPointer)); }

  // --------------------------------------------------------------------------------
  unsigned int fContentSize;                      // size of fContent array
  // --------------------------------------------------------------------------------
  
  unsigned int fGridVPointer;  // pointer to V axis in fContent array. U axis is at position 0

  unsigned char fMemAlignment[8]; // gap in memory in order to align fContent to 128 bits, 
                                  // assuming the structure itself is 128-bit aligned

  unsigned char fContent[1];      // variale size array which contains all data
};



inline AliFlexibleSpline2D3D::AliFlexibleSpline2D3D( int nKnotsU, const float knotsU[], int nAxisTicksU,
				     int nKnotsV, const float knotsV[], int nAxisTicksV )
 :		       		       
 fContentSize(0),
 fGridVPointer(0)
{ 
  Init( nKnotsU, knotsU, nAxisTicksU, nKnotsV, knotsV, nAxisTicksV );
}


inline int AliFlexibleSpline2D3D::Init( int nKnotsU, const float knotsU[], int nAxisTicksU,
					int nKnotsV, const float knotsV[], int nAxisTicksV )
{

  int ret = 0;
  const float knotsDef[4] = { 0.f, 0.33f, 0.66f, 1.f };

  if( nKnotsU < 4 ){ nKnotsU = 4; knotsU = knotsDef; ret = -1; }
  if( nKnotsV < 4 ){ nKnotsV = 4; knotsV = knotsDef; ret = -1; }
  if( nAxisTicksU < 1 ){ nAxisTicksU = 1; ret = -1; }
  if( nAxisTicksV < 1 ){ nAxisTicksV = 1; ret = -1; }

  AliFlexibleSpline1D &gridU = GridU();
  gridU.Init( nKnotsU, knotsU, nAxisTicksU );
  gridU.ReleaseBorderRegions();

  fGridVPointer = (unsigned int) ( gridU.GetSize() );

  if( fGridVPointer % 16 ) fGridVPointer = (fGridVPointer/16 + 1)*16; // 128 bit alignment
 
  AliFlexibleSpline1D &gridV = GridV();
  gridV.Init( nKnotsU, knotsU, nAxisTicksU );

  fContentSize = fGridVPointer + gridV.GetSize();
  return ret;
}


inline unsigned int AliFlexibleSpline2D3D::EstimateSize( int nKnotsU, int nAxisTicksU,
							 int nKnotsV, int nAxisTicksV )
{
  if( nKnotsU < 4 ) nKnotsU = 4;
  if( nKnotsV < 4 ) nKnotsV = 4;
  if( nAxisTicksU < 1 ) nAxisTicksU = 1;
  if( nAxisTicksV < 1 ) nAxisTicksV = 1;
  return sizeof(AliFlexibleSpline2D3D) +
    + AliFlexibleSpline1D::EstimateSize( nKnotsU+2, nAxisTicksU ) // +2 knots in case 0.&1. will be  missing in the knots array
    + 16 // alignment
    + AliFlexibleSpline1D::EstimateSize( nKnotsV+2, nAxisTicksV ); // +2 knots in case 0.&1. will be missing in the knots array
}

/*
inline void AliFlexibleSpline2D3D::GetSpline( const float *data, float u, float v, float &x, float &y, float &z ) const
{
  const AliFlexibleSpline1D &gridU = GetGridU();
  const AliFlexibleSpline1D &gridV = GetGridV();
  int nu = gridU.GetNKnots();
  int iu = gridU.GetKnotIndex( u );
  int iv = gridV.GetKnotIndex( v );
  const AliFlexibleSpline1D::TKnot &knotU =  gridU.GetKnot( iu );
  const AliFlexibleSpline1D::TKnot &knotV =  gridV.GetKnot( iv );
  
  float f[4][3];
  const float *dt = data + (nu*(iv-1)+iu-1)*3;
  for( int j=0; j<4; j++){
    f[j][0] = gridU.GetSpline( u, knotU, dt[0], dt[3+0], dt[6+0], dt[9+0]);
    f[j][1] = gridU.GetSpline( u, knotU, dt[1], dt[3+1], dt[6+1], dt[9+1]);
    f[j][2] = gridU.GetSpline( u, knotU, dt[2], dt[3+2], dt[6+2], dt[9+2]);
    dt+=nu*3;
  }
  x = gridV.GetSpline( v, knotV, f[0][0], f[1][0], f[2][0], f[3][0] );
  y = gridV.GetSpline( v, knotV, f[0][1], f[1][1], f[2][1], f[3][1] );
  z = gridV.GetSpline( v, knotV, f[0][2], f[1][2], f[2][2], f[3][2] ); 
}
*/

inline void AliFlexibleSpline2D3D::GetSpline( const float *data, float u, float v, float &x, float &y, float &z ) const
{
  const AliFlexibleSpline1D &gridU = GetGridU();
  const AliFlexibleSpline1D &gridV = GetGridV();
  int nu = gridU.GetNKnots();
  int iu = gridU.GetKnotIndex( u );
  int iv = gridV.GetKnotIndex( v );
  const AliFlexibleSpline1D::TKnot &knotU =  gridU.GetKnot( iu );
  const AliFlexibleSpline1D::TKnot &knotV =  gridV.GetKnot( iv );
  
  const float *dataV0 = data + (nu*(iv-1)+iu-1)*3;
  const float *dataV1 = dataV0 + 3*nu;
  const float *dataV2 = dataV0 + 6*nu;
  const float *dataV3 = dataV0 + 9*nu;

  float dataV[12];
  for( int i=0; i<12; i++){
    dataV[i] = gridV.GetSpline( v, knotV, dataV0[i], dataV1[i], dataV2[i], dataV3[i]);
  }

  float *dataU0 = dataV + 0;
  float *dataU1 = dataV + 3;
  float *dataU2 = dataV + 6;
  float *dataU3 = dataV + 9;

  float res[3];
  for( int i=0; i<3; i++ ){
    res[i] = gridU.GetSpline( u, knotU, dataU0[i], dataU1[i], dataU2[i], dataU3[i] );
  }
  x = res[0];
  y = res[1];
  z = res[2];
}



#endif
