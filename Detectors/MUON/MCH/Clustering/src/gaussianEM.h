// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _GAUSSIANEM_H
#define _GAUSSIANEM_H

extern "C" {
void computeDiscretizedGaussian2D(const double* xyInfSup, const double* theta,
                                  int K, int N, int k, double* z);

void generateMixedGaussians2D(const double* xyInfSup, const double* theta,
                              int K, int N, double* z);

double computeWeightedLogLikelihood(const double* xyInfSup, const double* theta, const double* z,
                                    int K, int N);

double weightedEMLoop(const double* xyDxy, const Mask_t* saturated, const double* z,
                      const double* theta0, const Mask_t* thetaMask,
                      int K, int N,
                      int mode, double LConvergence, int verbose, double* theta);
}

#endif // _GAUSSIANEM_H