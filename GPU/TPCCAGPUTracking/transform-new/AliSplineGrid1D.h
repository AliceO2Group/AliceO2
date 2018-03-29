#ifndef AliSplineGrid1D_H
#define AliSplineGrid1D_H

/********************************************************************* 
 * This file is property of and copyright by the ALICE HLT Project.  * 
 * ALICE Experiment at CERN, All rights reserved.                    *
 * See cxx source for full Copyright notice                          *
 * Authors: sergey.gorbunov@cern.ch                                  *
 *                                                                   *
 *********************************************************************/

/*
 The class represents 1-dimensional non-uniform grid with variable distance between its knots.
 To be used by a spline parameterisation of TPC transformation.

 Local coordinate U on the interval [0., 1.] is used.

 The class provides fast search of a grid segment: (float U ) -> (int iSegment ).
 Duing the search, U coordinate is rounded to closest i*1./nAxisTicks value.

 The class is flat C structure. No virtual methods, no ROOT types are used.
 All arrays are placed in memory right after the structure and accessed via fContent[].


 To create a grid, do:

 byte *memory = new byte [ AliSplineGrid1D::EstimateSize( .. )];

   then 

 AliSplineGrid1D *grid = new( memory ) AliSplineGrid1D( .. )];

    or

 AliSplineGrid1D *grid = reinterpret_cast<AliSplineGrid1D*>( memory );
 grid->Init( .. );

*/



class AliSplineGrid1D
{
 public:
  
  struct AliSplineGrid1DKnot{
    float fU; // u coordinate of the knot
    float fLength2Inv; // inverse length of the segment [this knote, right knote]
    float fLength4Inv; // inverse length of 3 segments: [left + this + rigth + next]
  };

  AliSplineGrid1D( int nKnots, float knots[], int nAxisTicks );

  ~AliSplineGrid1D(){} // do nothing

  int Init( int nKnots, float knots[], int nAxisTicks );

  void ReleaseBorderRegions();

  int GetKnotNumber( float u ) const;  

  const AliSplineGrid1DKnot &GetKnot( int i) const { 
    return GetKnots()[i]; 
  }


  const AliSplineGrid1DKnot* GetKnots() const { 
    return reinterpret_cast<const  AliSplineGrid1DKnot*>( fContent ); 
  }

  // --------------------------------------------------------------------------------
  // -- Size methods

  static unsigned int EstimateSize( int nKnots, int nAxisTicks ); 

  unsigned int GetSize() const { 
    return fContent -  reinterpret_cast<const unsigned char *>(this) + fContentSize; 
  }

  // technical stuff

  const int* GetTick2KnotMap() const  { 
    return reinterpret_cast<const int*>( fContent + fTick2KnotMapPointer); 
  }

 private: 

  /** copy constructor & assignment operator prohibited */
  AliSplineGrid1D(const AliSplineGrid1D&);
  AliSplineGrid1D& operator=(const AliSplineGrid1D&);

  //
  AliSplineGrid1DKnot* Knots() { 
    return reinterpret_cast<AliSplineGrid1DKnot*>( fContent ); 
  }
  int* Tick2KnotMap(){ 
    return reinterpret_cast<int*>( fContent + fTick2KnotMapPointer); 
  }

  // --------------------------------------------------------------------------------
  unsigned int fContentSize;                      // size of fContent array
  // --------------------------------------------------------------------------------
  
  int fNKnots; // n knots on the grid
  int fNAxisTicks;
  float fNAxisTicksFloat;

  unsigned int fTick2KnotMapPointer;  // pointer to tick -> Segment map in fContent array

  unsigned char fContent[1];                  // variale size array, which contains all data
};

  
inline AliSplineGrid1D::AliSplineGrid1D( int nKnots, float knots[], int nAxisTicks )
 :		       		       
 fContentSize(0),
 fNKnots(0),
 fNAxisTicks(0),
 fNAxisTicksFloat(0.f),
 fTick2KnotMapPointer(0)
{ 
  Init( nKnots, knots, nAxisTicks ); 
}

inline unsigned int AliSplineGrid1D::EstimateSize( int  nKnots, int nAxisTicks )
{
  if( nKnots<2 || nAxisTicks<1 ){ // something wrong, initialise a small grid and return an error
    nKnots = 2;
    nAxisTicks = 1;
  }
  return sizeof(AliSplineGrid1D) + nKnots*sizeof(AliSplineGrid1DKnot)+ (nAxisTicks+1)*sizeof(int);
}


inline int AliSplineGrid1D::Init( int nKnots, float knots[], int nAxisTicks )
{ 
  // initialise the grid with nKnots knots in the interval [0,1]
  // knots on the borders u==0. & u==1. are obligatory
  // array knots[] has nKnots entries
  //

  int ret = 0;
  if( nKnots<2 || nAxisTicks<1 ){ // something wrong, initialise a small grid and return an error
    ret = -1;
    nKnots = 2;    
    nAxisTicks = 1;
  }
  fNAxisTicks = nAxisTicks;
  fNAxisTicksFloat = fNAxisTicks;

  fNKnots = 1;
  AliSplineGrid1DKnot *s = Knots();
  s[0].fU = 0.;  
  for( int i=0; i<nKnots; i++){
    if( (knots[i] >= 1.) || (knots[i] <= s[fNKnots-1].fU) ) continue;
    s[fNKnots++].fU = knots[i];
  }
  s[fNKnots++].fU = 1.;

  s[0].fLength2Inv = 1./(s[1].fU - s[0].fU);
  s[0].fLength4Inv = 0; // undefined
  s[fNKnots-1].fLength2Inv = 0; // undefined
  s[fNKnots-1].fLength4Inv = 0; // undefined
  s[fNKnots-2].fLength2Inv = 1./(s[fNKnots-1].fU - s[fNKnots-2].fU);
  s[fNKnots-2].fLength4Inv = 0; // undefined
  for( int i=1; i<fNKnots-2; i++){
    s[i].fLength2Inv = 1./(s[i+1].fU - s[i].fU);
    s[i].fLength4Inv = 1./(s[i+2].fU - s[i-1].fU);
  }  
  
  fTick2KnotMapPointer = (unsigned int) ( fNKnots*sizeof(AliSplineGrid1DKnot) );
  int *map = Tick2KnotMap();
  float u = 0; // first tick
  float step = 1./fNAxisTicks;
  for( int i=0, iKnot=0; i<fNAxisTicks; i++){
    while( s[iKnot+1].fU <= u ){
      iKnot++;
    }
    map[i] = iKnot;    
    u += step;
  }
  map[fNAxisTicks] = map[fNAxisTicks-1]; // this is map for just single value u == 1.f

  fContentSize = fTick2KnotMapPointer + (fNAxisTicks+1)*sizeof(int);

  return ret;
}


inline void AliSplineGrid1D::ReleaseBorderRegions()
{
  //
  // Map U coordinates from the first segment [knot0,knot1] to the second segment [knot1,knot2].
  // Map coordinates from segment [knot{n-2},knot{n-1}] to the previous segment [knot{n-3},knot{n-4}]
  //
  // This trick allows one to use splines without special conditions for border cases. 
  // Any U from [0,1] is mapped to some knote i, where i-1, i, i+1, and i+2 knots are always exist
  // Note that the grid should have at least 4 knots.
  //

  if( fNKnots >=4 ){
    int *map = Tick2KnotMap();
    for( int i=0; i<fNAxisTicks; i++){
      if( map[i] < 1 ) map[i]=1;
      if( map[i] > fNKnots-3 ) map[i]=fNKnots-3;
    }
  }
}


inline int AliSplineGrid1D::GetKnotNumber( float u ) const 
{
  // most frequently called method
  int itick = (int) ( u * fNAxisTicksFloat );
  return GetTick2KnotMap()[ itick ];
}  





#endif
