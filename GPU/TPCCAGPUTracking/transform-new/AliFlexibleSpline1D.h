#ifndef AliFlexibleSpline1D_H
#define AliFlexibleSpline1D_H

/********************************************************************* 
 * This file is property of and copyright by the ALICE HLT Project.  * 
 * ALICE Experiment at CERN, All rights reserved.                    *
 * See cxx source for full Copyright notice                          *
 * Authors: sergey.gorbunov@cern.ch                                  *
 *                                                                   *
 *********************************************************************/

/*
 The class represents 1-dimensional non-uniform (let's call it flexible) spline 
 with variable distance between knots.
 
 To be used by a spline parameterisation of TPC transformation.

 Local coordinate U on the interval [0., 1.] is used.

 The class provides fast search of a spline segment: (float U ) -> (int iSegment ).
 Duing the search, U coordinate is rounded to closest i*1./nAxisTicks value.

 The class is flat C structure. No virtual methods, no ROOT types are used.
 All arrays are placed in memory right after the structure and accessed via fContent[].

 To create a spline, do:

 byte *memory = new byte [ AliFlexibleSpline1D::EstimateSize( .. )];

   then 

 AliFlexibleSpline1D *spline = new( memory ) AliFlexibleSpline1D( .. )];

    or

 AliFlexibleSpline1D *spline = reinterpret_cast<AliFlexibleSpline1D*>( memory );
 spline->Init( .. );

*/



class AliFlexibleSpline1D
{
 public:
  
  struct TKnot{
    float fU; // u coordinate of the knot i

    // some useful values for spline calculation:

    float fScale; // scale for u = inverse length of a  segment [i, i+1]:  1./L[i,i+1]
    float fScaleL; // scale for u derivative at knot i: L[i,i+1]/L[i-1,i+1]
    float fScaleR; //  scale for u derivative at knot i+1:  L[i,i+1] / L[i,i+2]
  };

  AliFlexibleSpline1D( int nKnots, float knots[], int nAxisTicks );

  ~AliFlexibleSpline1D(){} // do nothing

  int Init( int nKnots, float knots[], int nAxisTicks );

  template <typename T>
    T GetSpline( float u, T f0, T f1, T f2, T f3 );

  void ReleaseBorderRegions();

  int GetNKnots() const { return fNKnots; }

  int GetKnotIndex( float u ) const;  
 
  const AliFlexibleSpline1D::TKnot& GetKnot( int i ) const { 
    return GetKnots()[i]; 
  }

  const AliFlexibleSpline1D::TKnot& GetAssociatedKnot( float u ) const { 
    return GetKnot( GetKnotIndex( u ) ); 
  }

  const AliFlexibleSpline1D::TKnot* GetKnots() const { 
    return reinterpret_cast<const  AliFlexibleSpline1D::TKnot*>( fContent ); 
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

  int GetNAxisTicks() const { return (int) fNAxisTicksFloat; }

 private: 

  /** copy constructor & assignment operator prohibited */
  AliFlexibleSpline1D(const AliFlexibleSpline1D&);
  AliFlexibleSpline1D& operator=(const AliFlexibleSpline1D&);

  //

  AliFlexibleSpline1D::TKnot* Knots() { 
    return reinterpret_cast<AliFlexibleSpline1D::TKnot*>( fContent ); 
  }

  int* Tick2KnotMap(){ 
    return reinterpret_cast<int*>( fContent + fTick2KnotMapPointer); 
  }

  // --------------------------------------------------------------------------------
  unsigned int fContentSize;                      // size of fContent array
  // --------------------------------------------------------------------------------
  
  int fNKnots; // n knots on the grid
  float fNAxisTicksFloat; // keep this integer number in float format for faster calculations in GetKnotIndex()

  unsigned int fTick2KnotMapPointer;  // pointer to tick -> Segment map in fContent array
  
  //unsigned char fMemAlignment[0]; // gap in memory in order to align fContent to 128 bits, 
                                    // assuming the structure itself is also 128-bit aligned

  unsigned char fContent[1];                  // variale size array, which contains all data
};

  
inline AliFlexibleSpline1D::AliFlexibleSpline1D( int nKnots, float knots[], int nAxisTicks )
 :		       		       
 fContentSize(0),
 fNKnots(0), 
 fNAxisTicksFloat(0.f),
 fTick2KnotMapPointer(0)
{ 
  Init( nKnots, knots, nAxisTicks ); 
}

inline unsigned int AliFlexibleSpline1D::EstimateSize( int  nKnots, int nAxisTicks )
{
  if( nKnots<4 || nAxisTicks<1 ){ // something wrong, initialise a small grid and return an error
    nKnots = 4;
    nAxisTicks = 1;
  }
  return sizeof(AliFlexibleSpline1D) + nKnots*sizeof(AliFlexibleSpline1D::TKnot)+ (nAxisTicks+1)*sizeof(int);
}


inline int AliFlexibleSpline1D::Init( int nKnots, float knots[], int nAxisTicks )
{ 
  // initialise the grid with nKnots knots in the interval [0,1]
  // knots on the borders u==0. & u==1. are obligatory
  // array knots[] has nKnots entries
  //
  // number of knots can change during this initialisation 
  //

  int ret = 0;
  if( nAxisTicks<1 ){ // something wrong
    ret = -1;
    nAxisTicks = 1;
  }
  fNAxisTicksFloat = nAxisTicks;

  
  float knotsDef[4] = { 0.f, 0.33f, 0.66f, 1.f };

  if( nKnots<4  ){ // something wrong, initialise a small grid and return an error
    ret = -1;
    nKnots = 4;    
    knots = knotsDef;
  }
 
  fNKnots = 1;
  AliFlexibleSpline1D::TKnot *s = Knots();
  const float minDist = 1.e-8; // safety first
  s[0].fU = 0.; // we expect border knots 0. & 1. to be present in knots[] array, but to be sure we set them explicitly
  for( int i=0; i<nKnots; i++){
    if( (knots[i] >= 1. - minDist) || (knots[i] <= s[fNKnots-1].fU + minDist ) ) continue;
    s[fNKnots++].fU = knots[i];
  }
  s[fNKnots++].fU = 1.;

  {
    double du = (s[1].fU - s[0].fU);
    s[0].fScale = 1./du;
    s[0].fScaleL = 0.; // undefined
    s[0].fScaleR = du/(s[2].fU - s[0].fU);
  } 

  for( int i=1; i<fNKnots-2; i++){
    double du = (s[i+1].fU - s[i].fU);
    s[i].fScale  = 1./du;
    s[i].fScaleR = du/(s[i+2].fU - s[i].fU);
    s[i].fScaleL = du/(s[i+1].fU - s[i-1].fU);
  } 
 
  {
    double du = (s[fNKnots-1].fU - s[fNKnots-2].fU);
    s[fNKnots-2].fScale  = 1./du;
    s[fNKnots-2].fScaleL = du / (s[fNKnots-1].fU - s[fNKnots-3].fU);
    s[fNKnots-2].fScaleR = 0; // undefined
  }

  {
    s[fNKnots-1].fScale  = 0; // undefined
    s[fNKnots-1].fScaleL = 0; // undefined
    s[fNKnots-1].fScaleR = 0; // undefined
  }

  fTick2KnotMapPointer = (unsigned int) ( fNKnots*sizeof(AliFlexibleSpline1D::TKnot) );
  int *map = Tick2KnotMap();
  float u = 0; // first tick
  float step = 1./fNAxisTicksFloat;
  for( int i=0, iKnot=0; i<nAxisTicks; i++){
    while( s[iKnot+1].fU <= u ){
      iKnot++;
    }
    map[i] = iKnot;    
    u += step;
  }
  map[nAxisTicks] = map[nAxisTicks-1]; // this is map for just single value of u == 1.f

  fContentSize = fTick2KnotMapPointer + (nAxisTicks+1)*sizeof(int);

  return ret;
}


inline void AliFlexibleSpline1D::ReleaseBorderRegions()
{
  //
  // Map U coordinates from the first segment [knot0,knot1] to the second segment [knot1,knot2].
  // Map coordinates from segment [knot{n-2},knot{n-1}] to the previous segment [knot{n-3},knot{n-4}]
  //
  // This trick allows one to use splines without special conditions for border cases. 
  // Any U from [0,1] is mapped to some knote i, where i-1, i, i+1, and i+2 knots are always exist
  // Note that the grid should have at least 4 knots.
  //
  int nAxisTicks = GetNAxisTicks();
  if( fNKnots >=4 ){
    int *map = Tick2KnotMap();
    for( int i=0; i<nAxisTicks+1; i++){
      if( map[i] < 1 ) map[i]=1;
      if( map[i] > fNKnots-3 ) map[i]=fNKnots-3;
    }
  }
}


inline int AliFlexibleSpline1D::GetKnotIndex( float u ) const 
{
  //
  // most frequently called method
  // no out-of-range check performed
  //
  int itick = (int) ( u * fNAxisTicksFloat );
  return GetTick2KnotMap()[ itick ];
}  

template <typename T>
inline T AliFlexibleSpline1D::GetSpline( float u, T f0, T f1, T f2, T f3 )
{  
  const AliFlexibleSpline1D::TKnot &knot = GetAssociatedKnot( u );
 
  T x = (u-knot.fU)*knot.fScale; // scaled u
  T z1 = (f2-f0)*knot.fScaleL; // scaled u derivative at the knot 1 
  T z2 = (f3-f1)*knot.fScaleR; // scaled u derivative at the knot 2 
  
  T x2 = x*x;
  T df = f2-f1;
 
  // f(x) = ax^3 + bx^2 + cx + d
  //
  // f(0) = f1
  // f(1) = f2
  // f'(0) = z1
  // f'(1) = z2
  //

  // T d = f1;
  // T c = z1;
  T a = -df -df + z1 + z2;
  T b =  df - z1 - a;
  return a*x*x2 + b*x2 + z1*x + f1;
}

#endif
