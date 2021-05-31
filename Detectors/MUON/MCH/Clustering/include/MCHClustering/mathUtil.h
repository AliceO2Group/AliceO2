# ifndef  _MATHUTIL_H
# define  _MATHUTIL_H

# include <math.h>
# include <limits.h>
# include <float.h>
# include <stddef.h>

typedef short Mask_t;

inline static void vectorSetZero( double *u, int N) { 
  for (int i=0; i < N; i++) u[i] = 0; return;}

inline static void vectorSetZeroInt( int *u, int N) { 
  for (int i=0; i < N; i++) u[i] = 0; return;}

inline static void vectorSetZeroShort( short *u, int N) { 
  for (int i=0; i < N; i++) u[i] = 0; return;}

inline static void vectorSet( double *u, double value, int N) { 
  for (int i=0; i < N; i++) u[i] = value; return;}

inline static void vectorSetInt( int *u, int value, int N) { 
  for (int i=0; i < N; i++) u[i] = value; return;}

inline static void vectorSetShort( short *u, short value, int N) { 
  for (int i=0; i < N; i++) u[i] = value; return;}

inline static void vectorCopy( const double *src, int N, double *dest) { 
  for (int i=0; i < N; i++) dest[i] = src[i]; return;}

inline static void vectorCopyShort( const short *src, int N, short *dest) { 
  for (int i=0; i < N; i++) dest[i] = src[i]; return;}

inline static void vectorAddVector( const double *u, double cst, const double *v, int N, double *res) { 
  for (int i=0; i < N; i++) res[i] = u[i] + cst*v[i]; return;}
     
 inline static void vectorAddScalar( const double *u, double cst, int N, double *res) { 
  for (int i=0; i < N; i++) res[i] = u[i] + cst; return;}  
  
inline static void vectorMultVector( const double *u, const double *v, int N, double *res) {
  for (int i=0; i < N; i++) res[i] = u[i] * v[i]; return;}

inline static void vectorMultScalar( const double *u, double cst, int N, double *res) {
  for (int i=0; i < N; i++) res[i] = u[i] * cst; return;}

inline static double vectorSum( const double *u, int N) { 
  double res = 0; for (int i=0; i < N; i++) res += u[i]; return res; }

inline static int vectorSumInt( const int *u, int N) { 
  int res = 0; for (int i=0; i < N; i++) res += u[i]; return res; }

inline static int vectorSumShort( const short *u, int N) { 
  int res = 0; for (int i=0; i < N; i++) res += u[i]; return res; }

inline static int vectorSumRowInt( const int *matrix, int N, int M ) {
  int res = 0; for (int j=0; j < M; j++) res += matrix[j]; return res; }

inline static int vectorSumColumnInt( const int *matrix, int N, int M ) {
  int res = 0; for (int i=0; i < N; i++) res += matrix[i*M]; return res; }

inline static double vectorMin( const double *u, int N) { 
  double res = DBL_MAX; for (int i=0; i < N; i++) res = fmin(res, u[i]); return res; }

inline static double vectorMax( const double *u, int N) { 
  double res = -DBL_MAX; for (int i=0; i < N; i++) res = fmax(res, u[i]); return res; }

inline static short vectorMaxShort( const short *u, int N) { 
  short res = SHRT_MIN; for (int i=0; i < N; i++) res = std::max(res, u[i]); return res; }
//
// Logical operations
//
inline static void vectorNotShort( const short *src, int N, short *dest) { 
  for (int i=0; i < N; i++) dest[i] = ! src[i] ; return;}

//
// Compare oparations
//
inline static int vectorSumOfGreater( const double *src, double cmpValue, int N) { 
    int count=0; for(int i=0; i < N; i++) { count += (( src[i] > cmpValue ) ? 1: 0); } return count;}

//
// Mask operations
//
inline static int vectorBuildMaskEqualShort( short *src, short value, int N, short *mask) {
    int count=0; for(int i=0; i < N; i++) { mask[i] = (src[i] == value); count += ((src[i] == value));} return count;}

inline static int vectorGetIndexFromMaskInt( const Mask_t *mask, int N, int *indexVector ) { 
  int k=0;for (int i=0; i < N; i++) {  if ( mask[i] == 1) { indexVector[i] = k ; k++;} } return k;}
  // int k=0;for (int i=0; i < N; i++) { indexVector[i] = ( mask[i] == 1) ? k : -1; k++;} return k;}

inline static void vectorAppyMapIdxInt( const int *vect, const int *map, int N, int *mappedVect ) {
  for (int i=0; i < N; i++) { mappedVect[i] = map[ vect[ i ]]; } return; } 

inline static int vectorGather( const double *v, const Mask_t *mask, int N, double *gatherVector ) { 
  int k=0;for (int i=0; i < N; i++) { if ( mask[i] ) gatherVector[k++] = v[i];} return k;}

inline static int vectorScatter( const double *v, const Mask_t *mask, int N, double *scatterVec ) { 
  int k=0;for (int i=0; i < N; i++) { if ( mask[i] ) scatterVec[i] = v[k++];} return k;}

inline static int vectorGatherShort( const short *v, const Mask_t *mask, int N, short *gatherVector ) { 
  int k=0;for (int i=0; i < N; i++) { if ( mask[i] ) gatherVector[k++] = v[i];} return k;}

inline static int vectorGetIndexFromMaskShort( const Mask_t *mask, int N, short *index ) { 
  int k=0;for (int i=0; i < N; i++) { if ( mask[i] ) index[k++] = i;} return k;}

inline static int vectorGetIndexFromMask( const Mask_t *mask, int N, int *index ) { 
  int k=0;for (int i=0; i < N; i++) { if ( mask[i] ) index[k++] = i;} return k;}

inline static void vectorMaskedSet( const Mask_t *mask, const double *vTrue, const double *vFalse, int N, double *vR ) { 
  for (int i=0; i < N; i++) { vR[i] = ( mask[i] ) ? vTrue[i] : vFalse[i];}; return; }

inline static void vectorMaskedUpdate( const Mask_t *mask, const double *vTrue, int N, double *vR ) { 
  for (int i=0; i < N; i++) { vR[i] = ( mask[i] ) ? vTrue[i] : vR[i];}; return; }

inline static double vectorMaskedSum( const double *v, const Mask_t *mask, int N ) { 
  double sum=0; for (int i=0, k=0; i < N; i++) { sum += v[i]*mask[i];} return sum;}

inline static void vectorMaskedMult( const double *v, const Mask_t *mask, int N, double *res ) { 
  for (int i=0, k=0; i < N; i++) { res[i] = v[i]*mask[i];} return;} 

inline static void vectorMapShort( short *array, const short *map, int N) {
  for (int i=0; i < N; i++) { array[i] = map[ array[i] ];} }

void vectorPrint( const char *str, const double *x, int N);
void vectorPrintInt( const char *str, const int *x, int N);
void vectorPrintShort( const char *str, const short *x, int N);
void vectorPrint2Columns( const char *str, const double *x, const double *y, int N);

#endif // _MATHUTIL_H
