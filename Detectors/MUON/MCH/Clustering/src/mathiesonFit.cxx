# include <stdio.h>
# include <gsl/gsl_multifit_nlin.h>

# include "MCHClustering/dataStructure.h"
# include "MCHClustering/mathUtil.h"
# include "mathiesonFit.h"
# include "MCHClustering/mathieson.h"

int f_ChargeIntegral( const gsl_vector *gslParams, void *dataFit, gsl_vector *residuals) {
  funcDescription_t *dataPtr = (funcDescription_t *) dataFit; 
  int N = dataPtr->N;
  int K = dataPtr->K;
  double  *x = dataPtr->x_ptr;
  double  *y = dataPtr->y_ptr;
  double *dx = dataPtr->dx_ptr;
  double *dy = dataPtr->dy_ptr; 
  Mask_t  *cath = dataPtr->cath_ptr;
  double *zObs = dataPtr->zObs_ptr;
  Mask_t *notSaturated = dataPtr->notSaturated_ptr;
  int chamberId = dataPtr->chamberId;
  double *cathWeights      = dataPtr->cathWeights_ptr;
  double *zCathTotalCharge = dataPtr->zCathTotalCharge_ptr;
  int verbose = dataPtr->verbose;
  // Parameters
  const double *params = gsl_vector_const_ptr( gslParams, 0 );
  // Note:
  //  mux = mu[0:K-1]
  //  muy = mu[K:2K-1]
  const double *mu = &params[0]; 
  double *w = (double *) &params[2*K];
  
  // Set constrain: sum_(w_k) = 1
  // Rewriten ??? w[K-1] = 1.0 - vectorSum( w, K-1 ); 
  double lastW = 1.0 - vectorSum( w, K-1 );
  if (verbose > 1) {
    printf("  Function evaluation at:\n");
    for( int k=0; k < K; k++ )
       printf( "    mu_k[%d] = %g %g \n", k, mu[k], mu[K+k]);
    for( int k=0; k < K-1; k++ )
       printf( "    w_k[%d] = %g \n", k, w[k] );
    // Last W
    printf( "    w_k[%d] = %g \n", K-1, lastW );
  }
  int i;

  // Charge Integral on Pads
  double z[N];
  // Not used 
  double z_k[N];
  double zTmp[N];
  // TODO: optimize compute before
  double xyInfSup[4*N];
  //
  double *xInf = getXInf( xyInfSup, N);
  double *xSup = getXSup( xyInfSup, N);
  double *yInf = getYInf( xyInfSup, N);
  double *ySup = getYSup( xyInfSup, N);

  vectorSetZero( z, N);
  for (int k=0; k < K; k++) {
    // xInf[:] = x[:] - dx[:] - muX[k]
    vectorAddVector( x, -1.0, dx, N, xInf);
    vectorAddScalar( xInf, - mu[k], N, xInf);
    // xSup = xInf + 2.0 * dxy[0] 
    vectorAddVector( xInf, 2.0, dx, N, xSup);    
    // yInf = xy[1] - dxy[1] - mu[k,1]
    // ySup = yInf + 2.0 * dxy[1] 
    vectorAddVector( y, -1.0, dy, N, yInf);
    vectorAddScalar( yInf, - mu[K+k], N, yInf);
    // ySup = yInf + 2.0 * dxy[0] 
    vectorAddVector( yInf, 2.0, dy, N, ySup);
    //
    // z[:] +=  w[k] * cathWeight 
    //       * computeMathieson2DIntegral( xInf[:], xSup[:], yInf[:], ySup[:], N )  
    compute2DPadIntegrals( xyInfSup, N, chamberId, zTmp );
    vectorMultVector( zTmp, cathWeights, N, zTmp);
    // GG Note: if require compute the charge of k-th component
    // z_k[k] = vectorSum( zTmp, N )
    double wTmp = (k != K-1) ? w[k] : lastW;
    // saturated pads
    vectorMaskedMult( zTmp, notSaturated, N, zTmp);
    vectorAddVector( z, wTmp, zTmp, N, z );
  }

  // ??? why abs value for penalties
  double wPenal = abs( 1.0 - vectorSum( w, K-1 ) + lastW);
  // Not Used: zPenal
  // double zPenal = abs( 1.0 - vectorSum( z, N ) );
  //
  // Charge on cathodes penalisation
  double zCath0=0.0, zCath1=0.0;
  for (int i=0; i < N; i++) {
      if (cath[i] == 0) {
        zCath0 += z[i];
      } else {
        zCath1 += z[i];
      }
  } 
  double cathPenal = fabs(zCathTotalCharge[0] - zCath0) + fabs(zCathTotalCharge[1] - zCath1);
  // ??? vectorAdd( zObs, -1.0, residual );
  // TODO Optimize (elementwise not a good solution)
  for (i = 0; i < N; i++) {
    gsl_vector_set( residuals, i, (zObs[i] - z[i]) * (1.0+cathPenal) + wPenal );
    // gsl_vector_set( residuals, i, (zObs[i] - z[i])  + 0.*wPenal );
  }
  if (verbose > 1) {
    printf("  cathPenal=%5.4g wPenal=%5.4g \n", 1.0+cathPenal, wPenal);
    printf("  residual\n");
    printf("  %15s  %15s  %15s \n",  "zObs",  "z",  "residual");
    for (i = 0; i < N; i++) {
      printf("  %15.8f  %15.8f  %15.8f \n",  zObs[i], z[i], gsl_vector_get( residuals, i) );
    }
    printf("\n");
  }
  return GSL_SUCCESS;
}

void printState (int iter, gsl_multifit_fdfsolver * s, int K) { 
  printf("  Fitting iter=%3d |f(x)|=%g\n", iter, gsl_blas_dnrm2(s->f) );
  printf("    mu (x,y):");
  int k=0;
  for ( ; k < 2*K; k++)  
    printf (" % 7.3f", gsl_vector_get( s->x, k));
  printf("\n");
  double sumW = 0;
  printf("    w:");
  for ( ; k < 3*K-1; k++) {
    double w = gsl_vector_get( s->x, k);
    sumW += w;
    printf (" %7.3f", gsl_vector_get( s->x, k));
  }
  // Last w : 1.0 - sumW
  printf (" %7.3f", 1.0 - sumW);
  
  printf("\n");
  k=0;
  printf("    dx:");
  for ( ; k < 2*K; k++)  
    printf (" % 7.3f", gsl_vector_get( s->dx, k));
  printf("\n");
}

// Notes : 
//  - the intitialization of Mathieson module must be done before (initMathieson)
// zCathTotalCharge = sum excludind saturated pads
void fitMathieson( double *thetai, 
                   double *xyDxy, double *z, Mask_t *cath, Mask_t *notSaturated,
 
                   double *zCathTotalCharge,
                   int K, int N, int chamberId, int process,
                   double *thetaf,
                   double *khi2,
                   double *pError) {
  int status;

  // process
  int p = process; 
  int verbose = p & 0x3;
  p = p >> 2;
  int doJacobian = p & 0x1;
  p = p >> 1;
  int computeKhi2 = p & 0x1;
  p = p >> 1;
  int computeStdDev = p & 0x1;
  if (verbose) {
    printf( "Fitting \n");
    printf( "  mode: verbose, doJacobian, computeKhi2, computeStdDev %d %d %d %d\n", verbose, doJacobian, computeKhi2, computeStdDev);
  }
  //
  double *muAndWi = getMuAndW( thetai, K);
  //
  // Check if fitting is possible
  double *muAndWf = getMuAndW( thetaf, K);
  if (3*K-1 > N) {
      muAndWf[0] = NAN;
      muAndWf[K] = NAN;
      muAndWf[2*K] = NAN;
      return ;
  } 
  // Define Function, jacobian
  gsl_multifit_function_fdf f;
  f.f = &f_ChargeIntegral;
  f.df = 0;
  f.fdf = 0;
  f.n = N;
  f.p = 3*K-1;
  
  // Function description (extra data nor parameters)
  funcDescription_t mathiesonData;
  mathiesonData.N = N;
  mathiesonData.K = K;
  mathiesonData.x_ptr  = getX( xyDxy, N );
  mathiesonData.y_ptr  = getY( xyDxy, N );
  mathiesonData.dx_ptr = getDX( xyDxy, N );
  mathiesonData.dy_ptr = getDY( xyDxy, N );
  mathiesonData.cath_ptr = cath;
  mathiesonData.zObs_ptr = z;
  mathiesonData.notSaturated_ptr = notSaturated;
  
  // Init the weights
  double cathWeights[N];
  for (int i=0; i < N; i++) {
    cathWeights[i] = (cath[i] == 0) ? zCathTotalCharge[0] : zCathTotalCharge[1];
  } 
  mathiesonData.cathWeights_ptr = cathWeights;
  mathiesonData.chamberId = chamberId;
  mathiesonData.zCathTotalCharge_ptr = zCathTotalCharge;
  mathiesonData.verbose = verbose;
  //
  f.params = &mathiesonData; 
  
  // Set initial parameters
  gsl_vector_view params0 = gsl_vector_view_array (muAndWi, 3*K-1);
  
  // Fitting method
  gsl_multifit_fdfsolver *s = gsl_multifit_fdfsolver_alloc ( gsl_multifit_fdfsolver_lmsder, N, 3*K-1);
  // associate the fitting mode, the function, and the starting parameters
  gsl_multifit_fdfsolver_set (s, &f, &params0.vector);


  if (verbose) printState( -1, s, K);
  
  // Fitting iteration
  status = GSL_CONTINUE;
  for( int iter=0; (status == GSL_CONTINUE) && (iter < 500); iter++) {
    status = gsl_multifit_fdfsolver_iterate( s );
    if (verbose >= 2) printf ("  solver status = %s\n", gsl_strerror( status ));
    if (verbose) printState( iter, s, K);
    if (status) { printf("  End fitting \n"); break; };
    // GG TODO ???: adjust error in fct of charge 
    status = gsl_multifit_test_delta (s->dx, s->x, 1e-4, 1e-4);
    if (verbose >= 2) printf ("  status multifit_test_delta = %d %s\n", status, gsl_strerror( status ));
    iter++;
  }

   // Fitted parameters 
  for (int k=0; k < (3*K-1) ; k++) {
    muAndWf[k] = gsl_vector_get(s->x, k);
  }
  // Last w
  int k=0;
  // Mu part
  for ( ; k < 2*K; k++)
    muAndWf[k] = gsl_vector_get(s->x, k);
  // w part
  double sumW = 0;
  for ( ; k < 3*K-1; k++) {
    double w = gsl_vector_get( s->x, k);
    sumW += w;
    muAndWf[k] = w;
  }
  // Last w : 1.0 - sumW
  muAndWf[3*K-1] = 1.0 - sumW;
 
  // Khi2
  if ( computeKhi2 && (khi2 != 0) ) {
    // Khi2
    double chi = gsl_blas_dnrm2(s->f);
    double dof = N - (3*K -1);
    double c = fmax(1.0, chi / sqrt(dof)); 
    // printf("chisq/dof = %g\n",  chi*chi / dof);
    khi2[0] = chi*chi / dof;
  }

  // Parameter error
  if ( computeStdDev && (pError != 0) ) {    //
    // Covariance matrix an error
    gsl_matrix *covar = gsl_matrix_alloc (3*K-1, 3*K-1);  
    gsl_multifit_covar( s->J, 0.0, covar);      
    for (int k=0; k < (3*K-1) ; k++) {
      pError[k] = sqrt(gsl_matrix_get(covar,k,k));
    }
    gsl_matrix_free(covar);
  }   
  if (verbose >= 2) printf( "  status parameter error = %s\n", gsl_strerror (status));

  // Release memory
  gsl_multifit_fdfsolver_free(s);
  //
  return;
  }

