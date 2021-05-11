// TODO:
// - throw exception on error see other mechanisms
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <float.h>
# include <stdexcept>
#
# include "MCHClustering/dataStructure.h"
# include "MCHClustering/mathUtil.h"
# include "MCHClustering/padProcessing.h"

# define CHECK 1
# define VERBOSE 0

// Intersection matrix
static PadIdx_t* IInterJ=0;
static PadIdx_t* JInterI=0;
static PadIdx_t* intersectionMatrix=0;

// Maps
static MapKToIJ_t *mapKToIJ=0;
static PadIdx_t   *mapIJToK=0;

// Neighbors
static PadIdx_t *neighbors=0;

// Projected Pads
static int maxNbrOfProjPads=0;
static int nbrOfProjPads=0;
static double* projected_xyDxy=0;
static double* projX;
static double* projDX;
static double* projY;
static double* projDY;
// Charge on the projected pads
static double* projCh0=0;
static double* projCh1=0;
// cathodes group
static short* cath0ToGrpFromProj=0;
static short* cath1ToGrpFromProj=0;
//
static short* cath0ToTGrp=0;
static short* cath1ToTGrp=0;
//
int getNbrProjectedPads() { return nbrOfProjPads; };

void setNbrProjectedPads( int n) { nbrOfProjPads = n; maxNbrOfProjPads= n; };

void storeProjectedPads ( const double *xyDxyProj, const double *z, int nPads) {
  projected_xyDxy = new double[nPads*4];
  projCh0 = new double[nPads];
  projCh1 = new double[nPads];

  vectorCopy( xyDxyProj, 4*nPads, projected_xyDxy);
  vectorCopy( z, nPads, projCh0 );
  vectorCopy( z, nPads, projCh1 );
}

void copyProjectedPads(double *xyDxy, double *chA, double *chB) {

  for ( int i=0; i < 4; i++) {  
    for( int k=0; k < nbrOfProjPads; k++) xyDxy[i*nbrOfProjPads + k] = projected_xyDxy[i*maxNbrOfProjPads + k];
  }
  for( int k=0; k < nbrOfProjPads; k++) chA[k] = projCh0[k];
  for( int k=0; k < nbrOfProjPads; k++) chB[k] = projCh1[k];
  
}

void printMatrixInt( const char *str, const int *matrix, int N, int M) {
  printf("%s\n", str);
  for ( int i=0; i < N; i++) {
    for ( int j=0; j < M; j++) {
      printf(" %2d", matrix[i*M+j]);
    }
    printf("\n");
  }
  printf("\n");
}    

void printMatrixShort( const char *str, const short *matrix, int N, int M) {
  printf("%s\n", str);
  for ( int i=0; i < N; i++) {
    for ( int j=0; j < M; j++) {
      printf(" %2d", matrix[i*M+j]);
    }
    printf("\n");
  }
  printf("\n");
}  

void printInterMap( const char *str, const PadIdx_t *inter, int N) {
  const PadIdx_t *ij_ptr = inter;
  printf("%s\n", str);
  for( PadIdx_t i=0; i < N; i++) {
    printf("row/col %d:", i);  
    for( int k = 0; *ij_ptr != -1; k++, ij_ptr++) {
      printf(" %2d", *ij_ptr);
    }
    // skip -1, row/col change
    ij_ptr++;
    printf("\n");
  }
  printf("\n");
}

void printNeighbors( ) {
  int  N = nbrOfProjPads;
  PadIdx_t *neigh = neighbors;
  printf("Neighbors %d\n", N);
  for ( int i=0; i < N; i++) {
    printf("  neigh of i=%2d: ", i);
    for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t j = *neigh_ptr;
      printf("%d, ", j);
    }
    printf("\n");
  }
}

int checkConsistencyMapKToIJ( const PadIdx_t *aloneIPads, const PadIdx_t *aloneJPads, int N0, int N1) {
  MapKToIJ_t ij;
  int n=0;
  int rc=0;
  // Consistency with intersectionMatrix
  // and aloneI/JPads
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    ij = mapKToIJ[k];    
    if ((ij.i >= 0) && (ij.j >= 0)) { 
      if ( intersectionMatrix[ij.i*N1 + ij.j] != 1) {
        printf("ERROR: no intersection %d %d %d\n", ij.i , ij.j, intersectionMatrix[ij.i*N1 + ij.j]);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      } else {
          n++;
      }
    } else if (ij.i < 0) {
      if ( aloneJPads[ij.j] != k) { 
        printf("ERROR: j-pad should be alone %d %d %d %d\n", ij.i , ij.j, aloneIPads[ij.j], k);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      }
    } else if (ij.j < 0) {
        if ( aloneIPads[ij.i] != k) {
          printf("ERROR: i-pad should be alone %d %d %d %d\n", ij.i , ij.j, aloneJPads[ij.j], k);      
          throw std::overflow_error("Divide by zero exception");
          rc = -1;
        }
    }
  }
  // TODO : Make a test with alone pads ??? 
  int sum = vectorSumInt( intersectionMatrix, N0*N1 );
  if( sum != n ) {
    printf("ERROR: nbr of intersection differs %d %d \n", n, sum);      
    throw std::overflow_error("Divide by zero exception");
    rc = -1;
  }
  
  for ( int i=0; i<N0; i++ ) {
    for ( int j=0; j<N1; j++ ) {
      int k = mapIJToK[i*N1+ j];
      if (k >= 0) {
        ij = mapKToIJ[k];
        if ( ( ij.i != i) || ( ij.j != j) ) throw std::overflow_error("checkConsistencyMapKToIJ: MapIJToK/MapKToIJ");
      } 
    }      
  }
  // Check mapKToIJ / mapIJToK 
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    ij = mapKToIJ[k];
    if ( ij.i < 0) {
      if (aloneJPads[ij.j] != k ) {
        printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
        throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK aloneJPads");
      }
    } else if( ij.j < 0 ) {
      if (aloneIPads[ij.i] != k ) {
        printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
        throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK aloneIPads");
      }
    } else if( mapIJToK[ij.i*N1+ ij.j] != k ) {
      printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
      throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK");
    }
  }
  
  return rc;
}

int getIndexByRow( PadIdx_t *matrix, PadIdx_t N, PadIdx_t M, PadIdx_t *IIdx) {
  int k = 0;
  // printf("N=%d, M=%d\n", N, M);
  for( PadIdx_t i=0; i < N; i++) {
    for( PadIdx_t j=0; j < M; j++) {    
      if (matrix[i*M+j] == 1) {
        IIdx[k]= j; k++;
        // printf("k=%d,", k);
      }
    }
    // end of row/columns
    IIdx[k]= -1; k++;
  }
  // printf("\n final k=%d \n", k);
  return k;
}

int getIndexByColumns( PadIdx_t *matrix, PadIdx_t N, PadIdx_t M, PadIdx_t *JIdx) {
  int k = 0;
  for( PadIdx_t j=0; j < M; j++) {    
    for( PadIdx_t i=0; i < N; i++) {
      if (matrix[i*M+j] == 1) {
        JIdx[k]= i; k++;
      }
    }
    // end of row/columns
    JIdx[k]= -1; k++;
  }
  return k;
}

// Build the neighbor list
PadIdx_t *getFirstNeighbors( const double *xyDxy, int N, int allocatedN) {
  const double relEps = (1.0 + 1.0e-7);
  const double *X = getConstX( xyDxy, allocatedN);
  const double *Y = getConstY( xyDxy, allocatedN);
  const double *DX = getConstDX( xyDxy, allocatedN);
  const double *DY = getConstDY( xyDxy, allocatedN); 
  // 8 neigbours + the center pad itself + separator (-1)
  neighbors = new PadIdx_t[MaxNeighbors*N];
  for( PadIdx_t i=0; i<N; i++) {
    PadIdx_t *i_neigh = getNeighborsOf(neighbors, i);
    // Search neighbors of i
    for( PadIdx_t j=0; j<N; j++) {
      int xMask0 = ( fabs( X[i] - X[j]) < relEps * (DX[i] + DX[j]) );
      int yMask0 = ( fabs( Y[i] - Y[j]) < relEps * (DY[i]) );
      int xMask1 = ( fabs( X[i] - X[j]) < relEps * (DX[i]) );
      int yMask1 = ( fabs( Y[i] - Y[j]) < relEps * (DY[i] + DY[j]) );
      if ( (xMask0 && yMask0) || (xMask1 && yMask1) ) {
        *i_neigh = j;
        i_neigh++;
        // Check
        // printf( "pad %d neighbor %d xMask=%d yMask=%d\n", i, j, (xMask0 && yMask0), (xMask1 && yMask1));
      }
    }
    *i_neigh = -1;
    if (  CHECK && (fabs( i_neigh - getNeighborsOf(neighbors, i) ) > MaxNeighbors) ) {
      printf("Pad %d : nbr of neighbours %ld greater than the limit %d \n", 
              i, i_neigh - getNeighborsOf(neighbors, i), MaxNeighbors );
      throw std::overflow_error("Not enough allocation");
    }
  }
  if (VERBOSE) printNeighbors();
  return neighbors;
}

/// Used ???
void computeAndStoreFirstNeighbors( const double *xyDxy, int N, int allocatedN) {            
  neighbors = getFirstNeighbors( xyDxy, N, allocatedN);
}

int buildProjectedPads( 
            const double *xy0InfSup, const double *xy1InfSup, 
            PadIdx_t N0, PadIdx_t N1, 
            PadIdx_t *aloneIPads, PadIdx_t *aloneJPads, int includeAlonePads) {
  // Use positive values of the intersectionMatrix
  // negative ones are isolated pads
  // Compute the new location of the projected pads (projected_xyDxy)
  // and the mapping mapKToIJ which maps k (projected pads) 
  // to i, j (cathode pads)
  const double *x0Inf = getConstXInf( xy0InfSup, N0);
  const double *y0Inf = getConstYInf( xy0InfSup, N0);
  const double *x0Sup = getConstXSup( xy0InfSup, N0);
  const double *y0Sup = getConstYSup( xy0InfSup, N0);
  const double *x1Inf = getConstXInf( xy1InfSup, N1);
  const double *y1Inf = getConstYInf( xy1InfSup, N1);
  const double *x1Sup = getConstXSup( xy1InfSup, N1);
  const double *y1Sup = getConstYSup( xy1InfSup, N1);
  
  double l, r, b, t;
  int k=0; PadIdx_t *ij_ptr=IInterJ;
  double countIInterJ, countJInterI;
  PadIdx_t i, j;
  for( i=0; i < N0; i++) {
    // Nbr of j-pads intersepting  i-pad
    for( countIInterJ = 0; *ij_ptr != -1; countIInterJ++, ij_ptr++) {
      j = *ij_ptr;
      // Debug
      // printf("X[0/1]inf/sup %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j,  x0Inf[i], x0Sup[i], x1Inf[j], x1Sup[j]);
      l = fmax( x0Inf[i], x1Inf[j]);
      r = fmin( x0Sup[i], x1Sup[j]);
      b = fmax( y0Inf[i], y1Inf[j]);     
      t = fmin( y0Sup[i], y1Sup[j]);
      projX[k] = (l+r)*0.5;
      projY[k] = (b+t)*0.5;
      projDX[k] = (r-l)*0.5;
      projDY[k] = (t-b)*0.5;
      mapKToIJ[k].i = i;
      mapKToIJ[k].j = j;          
      mapIJToK[i*N1+ j] = k;
      // Debug
      // printf("newpad %d %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j, k, projX[k], projY[k], projDX[k], projDY[k]);
      k++;
    }
    // Test if there is no intercepting pads with i-pad
    if( (countIInterJ == 0) && includeAlonePads ) {
      l = x0Inf[i];
      r = x0Sup[i];
      b = y0Inf[i];      
      t = y0Sup[i];
      projX[k] = (l+r)*0.5;
      projY[k] = (b+t)*0.5;
      projDX[k] = (r-l)*0.5;
      projDY[k] = (t-b)*0.5;
      // printf("newpad alone cath0 %d %d %9.3g %9.3g %9.3g %9.3g\n", i, k, projX[k], projY[k], projDX[k], projDY[k]);
      // Not used ??? 
      mapKToIJ[k].i =i ; mapKToIJ[k].j=-1;
      aloneIPads[i] = k;
      k++;
    }
    // Row change
    ij_ptr++;
  }
  // Just add alone j-pads of cathode 1
  if ( includeAlonePads ) {
    ij_ptr = JInterI;
    for( PadIdx_t j=0; j < N1; j++) {
      for( countJInterI = 0; *ij_ptr != -1; countJInterI++, ij_ptr++);
      if( countJInterI == 0) {
        l = x1Inf[j];
        r = x1Sup[j];
        b = y1Inf[j];      
        t = y1Sup[j];
        projX[k] = (l+r)*0.5;
        projY[k] = (b+t)*0.5;
        projDX[k] = (r-l)*0.5;
        projDY[k] = (t-b)*0.5;
        // Debug
        // printf("newpad alone cath1 %d %d %9.3g %9.3g %9.3g %9.3g\n", j, k, projX[k], projY[k], projDX[k], projDY[k]);
        // newCh0[k] = ch0[i];
        // Not used ??? 
        mapKToIJ[k].i=-1; mapKToIJ[k].j=j;
        aloneJPads[j] = k;
        k++;
      }
      // Skip -1, row/ col change
      ij_ptr++;
    }
  }
  return k;
}

void buildProjectedSaturatedPads( const Mask_t *saturated0, const Mask_t *saturated1, Mask_t *saturatedProj) {
  for( int k=0; k < nbrOfProjPads; k++) {
    MapKToIJ_t ij = mapKToIJ[k];
    saturatedProj[k] = saturated0[ij.i] || saturated1[ij.j];
  }
}

int projectChargeOnOnePlane( 
        const double *xy0InfSup, const double *ch0, 
        const double *xy1InfSup, const double *ch1, 
        PadIdx_t N0, PadIdx_t N1, int includeAlonePads) {
  // xy0InfSup, ch0, N0 : cathode 0
  // xy1InfSup, ch1, N1 : cathode 1
  // i: describes the cathode-0 pads
  // j: describes the cathode-1 pads
  // double maxFloat = DBL_MAX;
  double epsilon = 1.0e-4;
  const double *x0Inf = getConstXInf( xy0InfSup, N0);
  const double *y0Inf = getConstYInf( xy0InfSup, N0);
  const double *x0Sup = getConstXSup( xy0InfSup, N0);
  const double *y0Sup = getConstYSup( xy0InfSup, N0);
  const double *x1Inf = getConstXInf( xy1InfSup, N1);
  const double *y1Inf = getConstYInf( xy1InfSup, N1);
  const double *x1Sup = getConstXSup( xy1InfSup, N1);
  const double *y1Sup = getConstYSup( xy1InfSup, N1);
  // alocate in heap    
  intersectionMatrix = new PadIdx_t[N0*N1];
  // mapIJToK = (PadIdx_t *) malloc( N0*N1 * sizeof(PadIdx_t))
  mapIJToK = new PadIdx_t[N0*N1];
  PadIdx_t aloneIPads[N0], aloneJPads[N1];
  vectorSetZeroInt( intersectionMatrix, N0*N1);
  vectorSetInt( mapIJToK, -1, N0*N1);
  vectorSetInt( aloneIPads, -1, N0);
  vectorSetInt( aloneJPads, -1, N1);
  //
  // Looking for j pads, intercepting pad i
  // Build the intersection matrix
  //
  double xmin, xmax, ymin, ymax;
  PadIdx_t xInter, yInter;
  for( PadIdx_t i=0; i < N0; i++) {
    for( PadIdx_t j=0; j < N1; j++) {
      xmin = fmax( x0Inf[i], x1Inf[j] );
      xmax = fmin( x0Sup[i], x1Sup[j] );
      xInter = ( xmin <= (xmax - epsilon) );
      ymin = fmax( y0Inf[i], y1Inf[j] );
      ymax = fmin( y0Sup[i], y1Sup[j] );
      yInter = ( ymin <= (ymax - epsilon));
      intersectionMatrix[i*N1+j] =  (xInter & yInter);
    }
  }
  //  
  if (VERBOSE) printMatrixInt( "  Intersection Matrix", intersectionMatrix, N0, N1);
  //
  // Compute the max number of projected pads to make 
  // memory allocations 
  //
  maxNbrOfProjPads = vectorSumInt( intersectionMatrix, N0*N1 );
  int nbrOfAlonePads = 0;
  if (includeAlonePads) {
    // Add alone cath0-pads 
    for( PadIdx_t i=0; i < N0; i++) {
      if ( vectorSumRowInt( &intersectionMatrix[i*N1], N0, N1 ) == 0) nbrOfAlonePads++; 
    }
    // Add alone cath1-pads 
    for( PadIdx_t j=0; j < N1; j++) {
      if ( vectorSumColumnInt( &intersectionMatrix[j], N0, N1) == 0) nbrOfAlonePads++; 
    }  
  }
  // Add alone pas and row/column separators
  maxNbrOfProjPads += nbrOfAlonePads + fmax( N0, N1);
  if (VERBOSE) printf("  maxNbrOfProjPads %d\n", maxNbrOfProjPads);
  //
  //
  // Projected pad allocation
  // The limit maxNbrOfProjPads is alocated
  //
  projected_xyDxy = new double [4*maxNbrOfProjPads];
  projX  = getX ( projected_xyDxy, maxNbrOfProjPads);
  projY  = getY ( projected_xyDxy, maxNbrOfProjPads);
  projDX = getDX( projected_xyDxy, maxNbrOfProjPads);
  projDY = getDY( projected_xyDxy, maxNbrOfProjPads);
  //
  // Intersection Matrix Sparse representation
  //
  IInterJ = new PadIdx_t[maxNbrOfProjPads]; 
  JInterI = new PadIdx_t[maxNbrOfProjPads]; 
  int checkr = getIndexByRow( intersectionMatrix, N0, N1, IInterJ);

  int checkc = getIndexByColumns( intersectionMatrix, N0, N1, JInterI);
  if (CHECK) { 
    if (( checkr > maxNbrOfProjPads) || (checkc > maxNbrOfProjPads)) {
      printf("Allocation pb for  IInterJ or JInterI: allocated=%d, needed for row=%d, for col=%d \n", 
          maxNbrOfProjPads, checkr, checkc);
      throw std::overflow_error(
         "Allocation pb for  IInterJ or JInterI" );
    }
  }
  if (VERBOSE) {
    printInterMap("  IInterJ", IInterJ, N0 );
    printInterMap("  JInterI", JInterI, N1 );
  }
  mapKToIJ = new MapKToIJ_t[maxNbrOfProjPads];

  //
  // Build the new pads
  //
  nbrOfProjPads = buildProjectedPads( xy0InfSup, xy1InfSup, N0, N1, 
                                      aloneIPads, aloneJPads, includeAlonePads );
  
  if (CHECK == 1) checkConsistencyMapKToIJ( aloneIPads, aloneJPads, N0, N1);
  
  //
  // Get the isolated new pads
  // (they have no neighborhing)
  //
  int thereAreIsolatedPads = 0;
  PadIdx_t *neigh = getFirstNeighbors( projected_xyDxy, nbrOfProjPads, maxNbrOfProjPads ); 
  // printNeighbors();
  MapKToIJ_t ij;
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    if (getTheFirstNeighborOf(neighbors, k) == -1) {
      // pad k is isolated     
      thereAreIsolatedPads = 1;
      ij = mapKToIJ[k];
      if( (ij.i >= 0) && (ij.j >= 0)) { 
        if (VERBOSE) printf(" Isolated pad: nul intersection i,j = %d %d\n", ij.i, ij.j);
        intersectionMatrix[ij.i*N1 + ij.j] = 0; 
      } else {
        throw std::overflow_error("I/j negative (alone pad)");
      }
    }
  }
  if (VERBOSE && thereAreIsolatedPads) printf("There are isolated pads %d\n", thereAreIsolatedPads);
  //
  if( thereAreIsolatedPads == 1) {
    // Recompute all
    getIndexByRow( intersectionMatrix, N0, N1, IInterJ);
    getIndexByColumns( intersectionMatrix, N0, N1, JInterI);
    //
    // Build the new pads
    //
    nbrOfProjPads = buildProjectedPads( xy0InfSup, xy1InfSup, N0, N1, 
                                      aloneIPads, aloneJPads, includeAlonePads );
    getFirstNeighbors( projected_xyDxy, nbrOfProjPads, maxNbrOfProjPads );
  }

  //
  // Computing charges of the projected pads
  // Ch0 part
  //
  projCh0 = new double[nbrOfProjPads];
  projCh1 = new double[nbrOfProjPads];
  PadIdx_t k=0;
  double sumCh1ByRow;
  PadIdx_t *ij_ptr = IInterJ;
  PadIdx_t *rowStart;
  for( PadIdx_t i=0; i < N0; i++) {
    // Save the starting index of the beginnig of the row
    rowStart = ij_ptr;
    // sum of charge intercepting j-pad  
    for( sumCh1ByRow = 0.0; *ij_ptr != -1; ij_ptr++) sumCh1ByRow += ch1[ *ij_ptr ];
    if ( sumCh1ByRow != 0.0 ) { 
      double cst = ch0[i] / sumCh1ByRow;
      for( ij_ptr = rowStart; *ij_ptr != -1; ij_ptr++) {
        projCh0[k] = ch1[ *ij_ptr ] * cst;
        // Debug
        // printf(" i=%d, j=%d, k=%d, sumCh0ByCol = %g, projCh1[k]= %g \n", i, *ij_ptr, k, sumCh1ByRow, projCh0[k]);
        k++;
      }
    } else if (includeAlonePads){
      // Alone i-pad
      projCh0[k] = ch0[i];
      k++;      
    }
    // Move on a new row
    ij_ptr++;
  }
  // Just add alone pads of cathode 1
  if ( includeAlonePads ) {
    for( PadIdx_t j=0; j < N1; j++) {
      k = aloneJPads[j];
      if ( k >= 0 ){
        projCh0[k] = ch1[j];
      }
    }
  }
  //
  // Computing charges of the projected pads
  // Ch1 part
  //
  k=0; 
  double sumCh0ByCol;
  ij_ptr = JInterI;
  PadIdx_t *colStart;
  for( PadIdx_t j=0; j < N1; j++) {
    // Save the starting index of the beginnig of the column
    colStart = ij_ptr;
    // sum of charge intercepting i-pad  
    for( sumCh0ByCol = 0.0; *ij_ptr != -1; ij_ptr++) sumCh0ByCol += ch0[ *ij_ptr ];
    if ( sumCh0ByCol != 0.0 ) { 
      double cst = ch1[j] / sumCh0ByCol;
      for( ij_ptr = colStart; *ij_ptr != -1; ij_ptr++) {
        PadIdx_t i = *ij_ptr;
        k = mapIJToK[i * N1 + j];
        projCh1[k] = ch0[i] * cst;
        // Debug
        // printf(" j=%d, i=%d, k=%d, sumCh0ByCol = %g, projCh1[k]= %g \n", j, i, k, sumCh0ByCol, projCh1[k]);
      }
    } else if (includeAlonePads){
      // Alone j-pad
      k = aloneJPads[j];
      if ( CHECK && (k < 0) ) printf("ERROR: Alone j-pad with negative index j=%d\n", j);
      // printf("Alone i-pad  i=%d, k=%d\n", i, k);
      projCh1[k] = ch1[j];
    }
    ij_ptr++;
  }

  // Just add alone pads of cathode 0
  if ( includeAlonePads ) {
    ij_ptr = IInterJ;
    for( PadIdx_t i=0; i < N0; i++) {
      k = aloneIPads[i];
      if ( k >= 0) {
        // printf("Alone i-pad  i=%d, k=%d\n", i, k);
        projCh1[k] = ch0[i];
      } 
    }
  }
  
  if (VERBOSE) printXYdXY( "Projection", projected_xyDxy, maxNbrOfProjPads, nbrOfProjPads, projCh0, projCh1);
  return nbrOfProjPads;
}

int getConnectedComponentsOfProjPads( short *padGrp ) { 
  // Class from neighbors list of Projected pads, the pads in groups (connected components) 
  // padGrp is set to the group Id of the pad. 
  // If the group Id is zero, the the pad is unclassified
  // Return the number of groups
  int  N = nbrOfProjPads;
  PadIdx_t *neigh = neighbors;
  PadIdx_t neighToDo[N]; 
  vectorSetZeroShort( padGrp, N);  
  int nbrOfPadSetInGrp = 0;
  // Last padGrp to process
  short *curPadGrp = padGrp;
  short currentGrpId = 0;
  //
  int i, j, k;
  // printNeighbors();
  while (nbrOfPadSetInGrp < N) {
    // Seeking the first unclassed pad (padGrp[k]=0)
    for( ; (curPadGrp < &padGrp[N]) && *curPadGrp != 0; curPadGrp++ );
    k = curPadGrp - padGrp;
    // printf( "\nnbrOfPadSetInGrp %d\n", nbrOfPadSetInGrp);
    //
    // New group for k - then search all neighbours of k
    currentGrpId++;
    padGrp[k] = currentGrpId;
    nbrOfPadSetInGrp++;
    PadIdx_t startIdx = 0, endIdx = 1;
    neighToDo[ startIdx ] = k;
    // Labels k neighbors
    for( ; startIdx < endIdx; startIdx++) {
      i = neighToDo[startIdx];
      // printf("i %d neigh[i] %d: ", i, neigh[i] );
      //
      // Scan i neighbors
      for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh,i); *neigh_ptr != -1; neigh_ptr++) {
        j = *neigh_ptr;
        // printf("j %d\n, ", j);
        if (padGrp[j] == 0) {
          // Add the neighbors in the currentgroup  
          //
          // printf("found %d\n", j);
          padGrp[ j ] = currentGrpId;
          nbrOfPadSetInGrp++;
          // Append in the neighbor lis to search
          neighToDo[ endIdx] = j;
          endIdx++;
        }
      }
    }
  }
  // return tne number of Grp
  return currentGrpId;
}

int findLocalMaxWithLaplacian( const double *xyDxy, const double *z,
        Group_t *padToGrp,
        int nGroups,
        int N, int K, double *laplacian,  double *theta, PadIdx_t *thetaIndexes, Group_t *thetaToGrp) {
  //
  // ??? WARNING : theta must be allocated to N components (# of pads) 
  // ??? WARNING : use neigh of proj-pad
  //
  // Theta's
  double *varX  = getVarX(theta, K);
  double *varY  = getVarY(theta, K);
  double *muX   = getMuX(theta, K);
  double *muY   = getMuY(theta, K);
  double *w     = getW(theta, K);

  const double *X  = getConstX( xyDxy, N);
  const double *Y  = getConstY( xyDxy, N);
  const double *DX = getConstDX( xyDxy, N);
  const double *DY = getConstDY( xyDxy, N);
  //
  PadIdx_t *neigh = neighbors;
  // k is the # of seeds founds
  int k=0;
  double zi;
  int nSupi, nInfi;
  int nNeighi;
  // Group sum & max
  double sumW[nGroups+1];
  double chargeMax[nGroups+1];
  double chargeMin[nGroups+1];
  double maxLapl[nGroups+1];
  int kLocalMax[nGroups+1];
  for( int g=0; g < nGroups+1; g++) { sumW[g]=0; maxLapl[g]=0; kLocalMax[g]=0; chargeMin[g]=DBL_MAX; chargeMax[g]= 0.0;}
  //
  // Min/max
  for ( int i=0; i< N; i++) {
    Group_t g = padToGrp[i];
    if ( z[i] > chargeMax[g] ) chargeMax[g] = z[i];
    if ( z[i] < chargeMin[g] ) chargeMin[g] = z[i];
  }
  double zMin = vectorMin( chargeMin,  nGroups+1); 
  double zMax = vectorMax( chargeMax,  nGroups+1); 
  // Min charge wich can be a maximum
  double chargeMinForAMax = zMin + 0.05 * (zMax - zMin);
  //
  //
  int j;
  for ( int i=0; i< N; i++) {
    zi = z[i];
    nSupi = 0;
    nInfi = 0;
    nNeighi = 0;
    for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
      j = *neigh_ptr;
      if( zi >= z[j] ) nSupi++; else  nInfi++;
    }
    nNeighi = nSupi + nInfi;
    if ( (nNeighi <= 2) && (N > 2) ) {
      // Low nbr of neighbors but the # Pads must be > 2
      laplacian[i] = 0;
    } else {
      // TODO Test ((double) nSup -Inf) / nNeigh;
      laplacian[i] = ((double) nSupi) / nNeighi;
    }
    Group_t g = padToGrp[i];
    if (laplacian[i] > maxLapl[g] ) {
      maxLapl[g] = laplacian[i];
    }
    /// printf("???? lapl=%g %d %d %d group=%d maxLapl[g]=%g\n", laplacian[i], nSupi, nInfi, nNeighi, g, maxLapl[g]);
    // Strong max
    if ( (laplacian[i] >= 0.99) && (z[i] > chargeMinForAMax) ) {
        // save the seed
        w[k] = z[i];
        muX[k] = X[i];
        muY[k] = Y[i];
        thetaToGrp[k] = g;
        sumW[g]  += w[k];
        kLocalMax[g] += 1;
        thetaIndexes[k] = i;
        k++;
    }
  }
  // If there is no max in a group
  for (int g=1; g <= nGroups; g++) {
    if (VERBOSE) printf("findLocalMaxWithLaplacian: group=%d nLocalMax=%d, maxLaplacian=%7.3g\n", g,  kLocalMax[g], maxLapl[g]);
    if ( kLocalMax[g] == 0 ) {
      double lMax = maxLapl[g];
      for ( int i=0; i< N; i++) {
        if ( (padToGrp[i] == g) && (laplacian[i] == lMax) )  {
          // save the seed
          w[k]   = z[i];
          muX[k] = X[i];
          muY[k] = Y[i];
          thetaToGrp[k] = g;
          sumW[g] += w[k];
          kLocalMax[g] += 1;
          thetaIndexes[k] = i;
          k++;
        }
      }
    }
  }
  //
  // w normalization
  double cst[nGroups];
  for (int g=1; g <= nGroups; g++) {
    if ( kLocalMax[g] != 0 ) {
      cst[g] = 1.0 / sumW[g];
    } else {
      cst[g] = 1.0;
    }
  }
  // 
  for( int l=0; l < k; l++) w[l] = w[l] * cst[ thetaToGrp[l]];
  //
  return k;
}

// With Groups. Not Used ???
int findLocalMaxWithLaplacianV0( const double *xyDxy, const double *z, const PadIdx_t *grpIdxToProjIdx, int N, int xyDxyAllocated, double *laplacian,  double *theta) {
  //
  // ??? WARNING : theta must be allocated to N components (# of pads) 
  // ??? WARNING : use neigh of proj-pad
  //
  // Theta's
  double *varX  = getVarX(theta, N);
  double *varY  = getVarY(theta, N);
  double *muX   = getMuX(theta, N);
  double *muY   = getMuY(theta, N);
  double *w     = getW(theta, N);

  // ??? allocated
  const double *X  = getConstX( xyDxy, xyDxyAllocated);
  const double *Y  = getConstY( xyDxy, xyDxyAllocated);
  const double *DX = getConstDX( xyDxy, xyDxyAllocated);
  const double *DY = getConstDY( xyDxy, xyDxyAllocated);
  //
  PadIdx_t *neigh = neighbors;
  // k is the # of seeds founds
  int k=0;
  double zi;
  int nSupi, nInfi;
  int nNeighi;
  double sumW=0;
  int j;
  for ( int i=0; i< N; i++) {
    zi = z[i];
    nSupi = 0;
    nInfi = 0;
    nNeighi = 0;
    int iProj = ( grpIdxToProjIdx != 0) ? grpIdxToProjIdx[i] : i;
    for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, iProj); *neigh_ptr != -1; neigh_ptr++) {
      j = *neigh_ptr;
      if (CHECK && ( j >= N )) { 
        printf("findLocalMaxWithLaplacian error: i=%d has the j=%d neighbor but j > N=%d \n", iProj, j, N);
      }
      if( zi >= z[j] ) nSupi++; else  nInfi++;
    }
    nNeighi = nSupi + nInfi;
    if ( nNeighi <= 2) {
      laplacian[i] = 0;
    } else {
      // TODO Test ((double) nSup -Inf) / nNeigh;
      laplacian[i] = ((double) nSupi) / nNeighi;
    }
    // Strong max
    if ( (laplacian[i] >= 0.99) && (z[i] > 5.0) ) {
        // save the seed
        w[k] = z[i];
        muX[k] = X[i];
        muY[k] = Y[i];
        sumW  += w[k];
        k++;
    }
  }
  if (k==0) {
    // No founded seed
    // Weak max/seed
    double lMax = vectorMax(laplacian, N); 
    for ( int i=0; i< N; i++) {
      if ( laplacian[i] == lMax ) {
        // save the seed
        w[k]   = z[i];
        muX[k] = X[i];
        muY[k] = Y[i];
        sumW  += w[k];
        k++;
      }      
    }
  }
  // w normalization
  // ??? should be zero
  if ( k != 0 ) {
    double cst = 1.0 / sumW;
    for( int l=0; l < k; l++) w[l] = w[l] * cst;
  }
  //
  return k;
}

void assignOneCathPadsToGroup( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup) {
  cath0ToGrpFromProj = 0;
  cath1ToGrpFromProj = 0;
  if ( nCath0 != 0) {
    cath0ToGrpFromProj = new short[nCath0];
    vectorCopyShort( padGroup, nCath0, cath0ToGrpFromProj);
  } else {
    cath1ToGrpFromProj = new short[nCath1];
    vectorCopyShort( padGroup, nCath1, cath1ToGrpFromProj);   
  }
  vectorSetShort( wellSplitGroup, 1, nGrp+1);
}

/* ???
void forceSplitCathodes( double *newCath0, double *newcath1) {
    
  // need i/j intersection 
  // Cath0
  for ( int c=0; c < nCath0; c++ ) {
    if ( cath0ToGrp[c] <= 0 ) {
      // find conflicting pads with cath1 (j index)
      // .. i.e intersecting  j with an other group g
      // then separated charge from pads with different groups
      // To do that: project compute coef for group g
      //  S = sum z[u, g], u in same grp g and intercep i
      // For all g :
      //   zi' = zi/ S(g,i)
      // Matrix chargeRatio[ group, cathodes ]
    } else {
      // copy to new cath
         
    }   
  }
}
*/

void assignCathPadsToGroupFromProj( short *projPadGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup, short *matGrpGrp) {
  cath0ToGrpFromProj = new short[nCath0];
  cath1ToGrpFromProj = new short[nCath1];
  vectorSetZeroShort( cath0ToGrpFromProj, nCath0);
  vectorSetZeroShort( cath1ToGrpFromProj, nCath1);
  vectorSetShort( wellSplitGroup, 1, nGrp+1);
  vectorSetZeroShort( matGrpGrp, (nGrp+1)*(nGrp+1) );
  //
  PadIdx_t i, j; 
  short g, prevGroup0, prevGroup1;
  for( int k=0; k < nPads; k++) {
    g = projPadGroup[k];
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    //
    // Cathode 0
    //
    if ( i >= 0 ) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      prevGroup0 = cath0ToGrpFromProj[i];
      if ( (prevGroup0 == 0) || (prevGroup0 == g) ) {
        // No group before or same group
        cath0ToGrpFromProj[i] = g;
        matGrpGrp[ g*(nGrp+1) +  g ] = 1; 
      } else  {
        // Already a Grp which differs
        if ( prevGroup0 > 0) {
          // Invalid Old group
          wellSplitGroup[ prevGroup0 ] = 0;
          matGrpGrp[ g*(nGrp+1) +  prevGroup0 ] = 1; 
          matGrpGrp[ prevGroup0*(nGrp+1) +  g ] = 1; 
        }
        cath0ToGrpFromProj[i] = -g;
        // Invalid current group
        wellSplitGroup[ g ] = 0;
      }
    }
    //
    // Cathode 1
    //
    if ( j >= 0) {
      // Remark: if j is an alone pad (i<0)
      // j is processed as well
      prevGroup1 = cath1ToGrpFromProj[j];
      if ( (prevGroup1 == 0) || (prevGroup1 == g) ){
         // No group before
         cath1ToGrpFromProj[j] = g;
         matGrpGrp[ g*(nGrp+1) +  g ] = 1; 
      } else {
        // Already a Group 
        if ( prevGroup1 > 0) {
          // Invalid Old group
          wellSplitGroup[ prevGroup1 ] = 0;
          matGrpGrp[ g*(nGrp+1) +  prevGroup1 ] = 1; 
          matGrpGrp[ prevGroup1*(nGrp+1) +  g ] = 1; 
        }
        cath1ToGrpFromProj[j] = -g;
        // Invalid current group
        wellSplitGroup[ g ] = 0;
      }
    }
  }
  if (VERBOSE) printMatrixShort("Group/Group matrix", matGrpGrp, nGrp+1, nGrp+1);
}

int assignCathPadsToGroup( short *matGrpGrp, int nGrp, int nCath0, int nCath1, short *grpToGrp ) {
  cath0ToTGrp = new short[nCath0];
  cath1ToTGrp = new short[nCath1];
  vectorSetZeroShort(grpToGrp, nGrp+1);
  int newGroupID = 0;
  //
  int iNewGroup = 1;
  while ( iNewGroup < (nGrp+1)) {
    newGroupID++; 
    grpToGrp[iNewGroup] = newGroupID; 
    // printf("new Group idx=%d, newGroupID=%d\n", iNewGroup, newGroupID);
    for (int i=iNewGroup; i < (nGrp+1); i++) {
      // New Group
      int ishift = i*(nGrp+1);
      for (int j=i+1; j < (nGrp+1); j++) {
        if ( matGrpGrp[ishift+j] ) {
          /*
          if (CHECK) {
            if ( (grpToGrp[j] != 0) && (grpToGrp[j] != i) ) {
              printf("The mapping grpTogrp can't have 2 group values (surjective) oldGrp=%d newGrp=%d\n", grpToGrp[j], i);
              throw std::overflow_error("The mapping grpTogrp can't have 2 group values");
            }
          }
          */
          grpToGrp[j] = newGroupID;        
        }
      }
    }
    // Get the next index which have not a group
    int k;
    for( k = iNewGroup; k < (nGrp+1) && (grpToGrp[k] > 0); k++);
    iNewGroup = k; 
  }
  // vectorPrintShort( "grpToGrp", grpToGrp, nGrp+1);
  for (int c=0; c < nCath0; c++) {
    cath0ToTGrp[c] = grpToGrp[ abs( cath0ToGrpFromProj[c] ) ];
  }
  for (int c=0; c < nCath1; c++) {
    cath1ToTGrp[c] = grpToGrp[ abs( cath1ToGrpFromProj[c] ) ];
  }
  // vectorPrintShort( "cath0ToTGrp", cath0ToTGrp, nCath0);
  // vectorPrintShort( "cath1ToTGrp", cath1ToTGrp, nCath1);
  return newGroupID;
}

void copyCathToGrpFromProj( short *cath0Grp, short *cath1Grp, int nCath0, int nCath1) {
  vectorCopyShort( cath0ToGrpFromProj, nCath0, cath0Grp);
  vectorCopyShort( cath1ToGrpFromProj, nCath1, cath1Grp);
}

void getMaskCathToGrpFromProj( short g, short* mask0, short *mask1, int nCath0, int nCath1) {
    vectorBuildMaskEqualShort( cath0ToGrpFromProj, g, nCath0, mask0);
    vectorBuildMaskEqualShort( cath1ToGrpFromProj, g, nCath1, mask1);
}

void freeMemoryPadProcessing() {
  //
  // Intersection matrix
  if( IInterJ != 0) {
    delete IInterJ;
    IInterJ = 0;
  }
  if( JInterI != 0) {
    delete JInterI;
    JInterI = 0;
  }
  if( intersectionMatrix != 0) {
    delete intersectionMatrix;
    intersectionMatrix = 0;
  }
  //
  // Maps
  if( mapKToIJ != 0) {
    delete mapKToIJ;
    mapKToIJ = 0;
  }
  if( mapIJToK != 0) {
    delete mapIJToK;
    mapIJToK = 0;
  }
  //
  // Neighbors
  if( neighbors != 0) {
    delete neighbors;
    neighbors = 0;
  }
  // Projected Pads
  if( projected_xyDxy != 0) {
    delete projected_xyDxy;
    projected_xyDxy = 0;
  }
  // Charge on the projected pads
  if( projCh0 != 0) {
    delete projCh0;
    projCh0 = 0;
  }
  if( projCh1 != 0) {
    delete projCh1;
    projCh1 = 0;
  }
  if( cath0ToGrpFromProj != 0) {
    delete cath0ToGrpFromProj;
    cath0ToGrpFromProj = 0;
  }
  if( cath1ToGrpFromProj != 0) {
    delete cath1ToGrpFromProj;
    cath1ToGrpFromProj = 0;
  }
  if( cath0ToTGrp != 0) {
    delete cath0ToTGrp;
    cath0ToTGrp = 0;
  }
  if( cath1ToTGrp != 0) {
    delete cath1ToTGrp;
    cath1ToTGrp = 0;
  }
}