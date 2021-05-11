# ifndef  _MATHIESONFIT_H
# define  _MATHIESONFIT_H

# include <gsl/gsl_vector.h>
# include <gsl/gsl_blas.h>
# include <gsl/gsl_multifit_nlin.h>

# include "MCHClustering/mathUtil.h"

typedef struct dataFit {
  int N;
  int K;
  double *x_ptr;
  double *dx_ptr;
  double *y_ptr;
  double *dy_ptr;
  Mask_t *cath_ptr;
  double *zObs_ptr;
  Mask_t *notSaturated_ptr;
  double *cathWeights_ptr;
  int    chamberId;
  double *zCathTotalCharge_ptr;
  int    verbose;
} funcDescription_t;

// Notes : 
//  - the intitialization of Mathieson module must be done before (initMathieson)
extern "C" {
  void fitMathieson( double *muAndWi, 
                   double *xyAndDxy, double *z, Mask_t *cath, Mask_t *notSaturated,
                   double *zCathTotalCharge,
                   int K, int N, int chamberId, int jacobian,
                   double *muAndWf,
                   double *khi2,
                   double *pError);

  int f_ChargeIntegral( const gsl_vector *gslParams, void *data, 
        gsl_vector *residual );
}

# endif
