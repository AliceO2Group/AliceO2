#ifndef _MATHIESON_H
#define _MATHIESON_H

#include <stddef.h>
#include <math.h>

extern "C" {
void initMathieson();
void compute2DPadIntegrals(const double* xyInfSup, int N, int chamberI,
                           double Integrals[]);

void compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0, const double* theta,
                                           int N, int K, int chamberId,
                                           double Integrals[]);
}
#endif
