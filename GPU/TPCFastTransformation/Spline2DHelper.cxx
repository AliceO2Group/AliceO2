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

/// \file  Spline2DHelper.cxx
/// \brief Implementation of Spline2DHelper class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "Spline2DHelper.h"
#include "Spline1DHelper.h"

#include "SymMatrixSolver.h"
#include "GPUCommonDef.h"
#include "GPUCommonLogger.h"

#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TDecompBK.h"

#include <vector>
#include "TRandom.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TNtuple.h"
#include "TFile.h"
#include "GPUCommonMath.h"
#include <iostream>
#include <chrono>

using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT>
Spline2DHelper<DataT>::Spline2DHelper() : mError(), mFdimensions(0), mHelperU1(), mHelperU2()
{
}

template <typename DataT>
int Spline2DHelper<DataT>::storeError(int code, const char* msg)
{
  mError = msg;
  return code;
}

template <typename DataT>
void Spline2DHelper<DataT>::approximateFunction(
  DataT* Fparameters, double x1Min, double x1Max, double x2Min, double x2Max,
  std::function<void(double x1, double x2, double f[/*spline.getYdimensions()*/])> F) const
{
  /// Create best-fit spline parameters for a given input function F
  /// output in Fparameters

  std::vector<double> dataPointF(getNumberOfDataPoints() * mFdimensions);

  double scaleX1 = (x1Max - x1Min) / ((double)mHelperU1.getSpline().getUmax());
  double scaleX2 = (x2Max - x2Min) / ((double)mHelperU2.getSpline().getUmax());

  for (int iv = 0; iv < getNumberOfDataPointsU2(); iv++) {
    double x2 = x2Min + mHelperU2.getDataPoint(iv).u * scaleX2;
    for (int iu = 0; iu < getNumberOfDataPointsU1(); iu++) {
      double x1 = x1Min + mHelperU1.getDataPoint(iu).u * scaleX1;
      F(x1, x2, &dataPointF[(iv * getNumberOfDataPointsU1() + iu) * mFdimensions]);
    }
  }
  approximateFunction(Fparameters, dataPointF.data());
}

template <typename DataT>
void Spline2DHelper<DataT>::approximateFunctionBatch(
  DataT* Fparameters, double x1Min, double x1Max, double x2Min, double x2Max,
  std::function<void(const std::vector<double>& x1, const std::vector<double>& x2, std::vector<double> f[/*mFdimensions*/])> F,
  unsigned int batchsize) const
{
  /// Create best-fit spline parameters for a given input function F.
  /// F calculates values for a batch of points.
  /// output in Fparameters

  std::vector<double> dataPointF(getNumberOfDataPoints() * mFdimensions);

  double scaleX1 = (x1Max - x1Min) / ((double)mHelperU1.getSpline().getUmax());
  double scaleX2 = (x2Max - x2Min) / ((double)mHelperU2.getSpline().getUmax());

  std::vector<double> x1;
  x1.reserve(batchsize);

  std::vector<double> x2;
  x2.reserve(batchsize);

  std::vector<int> index;
  index.reserve(batchsize);

  std::vector<double> dataPointFTmp[mFdimensions];
  for (int iDim = 0; iDim < mFdimensions; ++iDim) {
    dataPointFTmp[iDim].reserve(batchsize);
  }

  unsigned int counter = 0;
  for (int iv = 0; iv < getNumberOfDataPointsU2(); iv++) {
    double x2Tmp = x2Min + mHelperU2.getDataPoint(iv).u * scaleX2;
    for (int iu = 0; iu < getNumberOfDataPointsU1(); iu++) {
      double x1Tmp = x1Min + mHelperU1.getDataPoint(iu).u * scaleX1;
      x1.emplace_back(x1Tmp);
      x2.emplace_back(x2Tmp);
      index.emplace_back((iv * getNumberOfDataPointsU1() + iu) * mFdimensions);
      ++counter;

      if (counter == batchsize || (iu == (getNumberOfDataPointsU1() - 1) && (iv == (getNumberOfDataPointsU2() - 1)))) {
        counter = 0;
        F(x1, x2, dataPointFTmp);
        unsigned int entries = index.size();

        for (unsigned int i = 0; i < entries; ++i) {
          const unsigned int indexTmp = index[i];
          for (int iDim = 0; iDim < mFdimensions; ++iDim) {
            dataPointF[indexTmp + iDim] = dataPointFTmp[iDim][i];
          }
        }

        x1.clear();
        x2.clear();
        index.clear();
        for (int iDim = 0; iDim < mFdimensions; ++iDim) {
          dataPointFTmp[iDim].clear();
        }
      }
    }
  }
  approximateFunction(Fparameters, dataPointF.data());
}

template <typename DataT>
void Spline2DHelper<DataT>::approximateFunction(
  DataT* Fparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
{
  /// approximate a function given as an array of values at data points

  const int Ndim = mFdimensions;
  const int Ndim2 = 2 * Ndim;
  const int Ndim3 = 3 * Ndim;
  const int Ndim4 = 4 * Ndim;

  int nDataPointsU = getNumberOfDataPointsU1();
  int nDataPointsV = getNumberOfDataPointsU2();

  int nKnotsU = mHelperU1.getSpline().getNumberOfKnots();
  int nKnotsV = mHelperU2.getSpline().getNumberOfKnots();

  std::unique_ptr<double[]> rotDataPointF(new double[nDataPointsU * nDataPointsV * Ndim]); // U DataPoints x V DataPoints :  rotated DataPointF for one output dimension
  std::unique_ptr<double[]> Dv(new double[nKnotsV * nDataPointsU * Ndim]);                 // V knots x U DataPoints

  std::unique_ptr<DataT[]> parU(new DataT[mHelperU1.getSpline().calcNumberOfParameters(Ndim)]);
  std::unique_ptr<DataT[]> parV(new DataT[mHelperU2.getSpline().calcNumberOfParameters(Ndim)]);

  std::unique_ptr<double[]> parUdbl(new double[mHelperU1.getSpline().calcNumberOfParameters(Ndim)]);
  std::unique_ptr<double[]> parVdbl(new double[mHelperU2.getSpline().calcNumberOfParameters(Ndim)]);

  // rotated data points (u,v)->(v,u)

  for (int ipu = 0; ipu < nDataPointsU; ipu++) {
    for (int ipv = 0; ipv < nDataPointsV; ipv++) {
      for (int dim = 0; dim < Ndim; dim++) {
        rotDataPointF[Ndim * (ipu * nDataPointsV + ipv) + dim] = DataPointF[Ndim * (ipv * nDataPointsU + ipu) + dim];
      }
    }
  }

  // get S and S'u at all the knots by interpolating along the U axis

  for (int iKnotV = 0; iKnotV < nKnotsV; ++iKnotV) {
    int ipv = mHelperU2.getKnotDataPoint(iKnotV);
    const double* DataPointFrow = &(DataPointF[Ndim * ipv * nDataPointsU]);
    mHelperU1.approximateFunctionGradually(parU.get(), DataPointFrow);

    for (int i = 0; i < mHelperU1.getSpline().calcNumberOfParameters(Ndim); i++) {
      parUdbl[i] = parU[i];
    }
    for (int iKnotU = 0; iKnotU < nKnotsU; ++iKnotU) {
      DataT* knotPar = &Fparameters[Ndim4 * (iKnotV * nKnotsU + iKnotU)];
      for (int dim = 0; dim < Ndim; ++dim) {
        knotPar[dim] = parU[Ndim * (2 * iKnotU) + dim];                // store S for all the knots
        knotPar[Ndim2 + dim] = parU[Ndim * (2 * iKnotU) + Ndim + dim]; // store S'u for all the knots //SG!!!
      }
    }

    // recalculate F values for all ipu DataPoints at V = ipv
    for (int ipu = 0; ipu < nDataPointsU; ipu++) {
      double splineF[Ndim];
      double u = mHelperU1.getDataPoint(ipu).u;
      mHelperU1.getSpline().interpolateU(Ndim, parUdbl.get(), u, splineF);
      for (int dim = 0; dim < Ndim; dim++) {
        rotDataPointF[(ipu * nDataPointsV + ipv) * Ndim + dim] = splineF[dim];
      }
    }
  }

  // calculate S'v at all data points with V == V of a knot

  for (int ipu = 0; ipu < nDataPointsU; ipu++) {
    const double* DataPointFcol = &(rotDataPointF[ipu * nDataPointsV * Ndim]);
    mHelperU2.approximateFunctionGradually(parV.get(), DataPointFcol);
    for (int iKnotV = 0; iKnotV < nKnotsV; iKnotV++) {
      for (int dim = 0; dim < Ndim; dim++) {
        double dv = parV[(iKnotV * 2 + 1) * Ndim + dim];
        Dv[(iKnotV * nDataPointsU + ipu) * Ndim + dim] = dv;
      }
    }
  }

  // fit S'v and S''_vu at all the knots

  for (int iKnotV = 0; iKnotV < nKnotsV; ++iKnotV) {
    const double* Dvrow = &(Dv[iKnotV * nDataPointsU * Ndim]);
    mHelperU1.approximateFunction(parU.get(), Dvrow);
    for (int iKnotU = 0; iKnotU < nKnotsU; ++iKnotU) {
      for (int dim = 0; dim < Ndim; ++dim) {
        Fparameters[Ndim4 * (iKnotV * nKnotsU + iKnotU) + Ndim + dim] = parU[Ndim * 2 * iKnotU + dim];         // store S'v for all the knots
        Fparameters[Ndim4 * (iKnotV * nKnotsU + iKnotU) + Ndim3 + dim] = parU[Ndim * 2 * iKnotU + Ndim + dim]; // store S''vu for all the knots
      }
    }
  }
}

template <typename DataT>
void Spline2DHelper<DataT>::approximateFunctionViaDataPoints(
  Spline2DContainer<DataT>& spline,
  double x1Min, double x1Max, double x2Min, double x2Max,
  std::function<void(double x1, double x2, double f[/*spline.getYdimensions()*/])> F,
  int nAuxiliaryDataPointsU1, int nAuxiliaryDataPointsU2)
{
  /// Create best-fit spline parameters for a given input function F

  setSpline(spline, nAuxiliaryDataPointsU1, nAuxiliaryDataPointsU2);
  mFdimensions = spline.getYdimensions();
  std::vector<double> dataPointX1(getNumberOfDataPoints());
  std::vector<double> dataPointX2(getNumberOfDataPoints());
  std::vector<double> dataPointF(getNumberOfDataPoints() * mFdimensions);

  double scaleX1 = (x1Max - x1Min) / ((double)mHelperU1.getSpline().getUmax());
  double scaleX2 = (x2Max - x2Min) / ((double)mHelperU2.getSpline().getUmax());

  for (int iv = 0; iv < getNumberOfDataPointsU2(); iv++) {
    double x2 = x2Min + mHelperU2.getDataPoint(iv).u * scaleX2;
    for (int iu = 0; iu < getNumberOfDataPointsU1(); iu++) {
      double x1 = x1Min + mHelperU1.getDataPoint(iu).u * scaleX1;
      int ind = iv * getNumberOfDataPointsU1() + iu;
      dataPointX1[ind] = x1;
      dataPointX2[ind] = x2;
      F(x1, x2, &dataPointF[ind * mFdimensions]);
    }
  }
  approximateDataPoints(spline, spline.getParameters(), x1Min, x1Max, x2Min, x2Max, &dataPointX1[0], &dataPointX2[0], &dataPointF[0], getNumberOfDataPoints());
}

template <typename DataT>
void Spline2DHelper<DataT>::setGrid(Spline2DContainer<DataT>& spline, double x1Min, double x1Max, double x2Min, double x2Max)
{
  mFdimensions = spline.getYdimensions();
  spline.setXrange(x1Min, x1Max, x2Min, x2Max);
  {
    std::vector<int> knots;
    for (int i = 0; i < spline.getGridX1().getNumberOfKnots(); i++) {
      knots.push_back(spline.getGridX1().getKnot(i).getU());
    }
    fGridU.recreate(0, knots.size(), knots.data());
    fGridU.setXrange(x1Min, x1Max);
  }
  {
    std::vector<int> knots;
    for (int i = 0; i < spline.getGridX2().getNumberOfKnots(); i++) {
      knots.push_back(spline.getGridX2().getKnot(i).getU());
    }
    fGridV.recreate(0, knots.size(), knots.data());
    fGridV.setXrange(x2Min, x2Max);
  }
}

template <typename DataT>
void Spline2DHelper<DataT>::getScoefficients(int iu, int iv, double u, double v,
                                             double coeff[16], int indices[16])
{
  const typename Spline1D<double>::Knot& knotU = fGridU.getKnot(iu);
  const typename Spline1D<double>::Knot& knotV = fGridV.getKnot(iv);
  int nu = fGridU.getNumberOfKnots();

  // indices of parameters that are involved in spline calculation, 1D case
  int i00 = (nu * iv + iu) * 4; // values { S, S'v, S'u, S''vu } at {u0, v0}
  int i01 = i00 + 4 * nu;       // values { ... } at {u0, v1}
  int i10 = i00 + 4;
  int i11 = i01 + 4;

  double dSl, dDl, dSr, dDr;
  Spline1DHelper<double>::getScoefficients(knotU, u, dSl, dDl, dSr, dDr);
  double dSd, dDd, dSu, dDu;
  Spline1DHelper<double>::getScoefficients(knotV, v, dSd, dDd, dSu, dDu);

  // A = Parameters + i00,  B = Parameters + i01
  // S = dSl * (dSd * A[0] + dDd * A[1]) + dDl * (dSd * A[2] + dDd * A[3]) +
  //     dSr * (dSd * A[4] + dDd * A[5]) + dDr * (dSd * A[6] + dDd * A[7]) +
  //     dSl * (dSu * B[0] + dDu * B[1]) + dDl * (dSu * B[2] + dDu * B[3]) +
  //     dSr * (dSu * B[4] + dDu * B[5]) + dDr * (dSu * B[6] + dDu * B[7]);

  double c[16] = {dSl * dSd, dSl * dDd, dDl * dSd, dDl * dDd,
                  dSr * dSd, dSr * dDd, dDr * dSd, dDr * dDd,
                  dSl * dSu, dSl * dDu, dDl * dSu, dDl * dDu,
                  dSr * dSu, dSr * dDu, dDr * dSu, dDr * dDu};
  for (int i = 0; i < 16; i++) {
    coeff[i] = c[i];
  }
  for (int i = 0; i < 4; i++) {
    indices[0 + i] = i00 + i;
    indices[4 + i] = i10 + i;
    indices[8 + i] = i01 + i;
    indices[12 + i] = i11 + i;
  }
}

template <typename DataT>
void Spline2DHelper<DataT>::approximateDataPoints(
  Spline2DContainer<DataT>& spline, DataT* splineParameters, double x1Min, double x1Max, double x2Min, double x2Max,
  const double dataPointX1[], const double dataPointX2[], const double dataPointF[/*getNumberOfDataPoints() x nFdim*/],
  int nDataPoints)
{
  /// Create best-fit spline parameters for a given input function F

  setGrid(spline, x1Min, x1Max, x2Min, x2Max);

  int nFdim = spline.getYdimensions();
  int nu = fGridU.getNumberOfKnots();
  int nv = fGridV.getNumberOfKnots();

  const int nPar = 4 * spline.getNumberOfKnots(); // n parameters for 1-dimensional F

  SymMatrixSolver solver(nPar, nFdim);

  for (int iPoint = 0; iPoint < nDataPoints; ++iPoint) {
    double u = fGridU.convXtoU(dataPointX1[iPoint]);
    double v = fGridV.convXtoU(dataPointX2[iPoint]);
    int iu = fGridU.getLeftKnotIndexForU(u);
    int iv = fGridV.getLeftKnotIndexForU(v);
    double c[16];
    int ind[16];
    getScoefficients(iu, iv, u, v, c, ind);

    // S(u,v) = sum c[i]*Parameters[ind[i]]

    for (int i = 0; i < 16; i++) {
      for (int j = i; j < 16; j++) {
        solver.A(ind[i], ind[j]) += c[i] * c[j];
      }
    }

    for (int iDim = 0; iDim < nFdim; iDim++) {
      double f = (double)dataPointF[iPoint * nFdim + iDim];
      for (int i = 0; i < 16; i++) {
        solver.B(ind[i], iDim) += f * c[i];
      }
    }
  } // data points

  // add extra smoothness for a case some data is missing
  for (int iu = 0; iu < nu - 1; iu++) {
    for (int iv = 0; iv < nv - 1; iv++) {
      int smoothPoint[4][2] = {
        {-1, -1},
        {-1, +1},
        {+1, -1},
        {+1, +1}};
      for (int iSet = 0; iSet < 4; iSet++) {
        int pu = iu + smoothPoint[iSet][0];
        int pv = iv + smoothPoint[iSet][1];
        int ip = (nu * pv + pu) * 4;
        if (pu < 0 || pv < 0 || pu >= nu || pv >= nv) {
          continue;
        }
        double c[17];
        int ind[17];
        getScoefficients(iu, iv, fGridU.getKnot(pu).u, fGridV.getKnot(pv).u, c, ind);
        c[16] = -1.;
        ind[16] = ip;
        // S = sum c[i]*Par[ind[i]]
        double w = 1.e-8;
        for (int i = 0; i < 17; i++) {
          for (int j = i; j < 17; j++) {
            solver.A(ind[i], ind[j]) += w * c[i] * c[j];
          }
        }
      }
    }
  }

  solver.solve();
  for (int i = 0; i < nPar; i++) {
    for (int iDim = 0; iDim < nFdim; iDim++) {
      splineParameters[i * nFdim + iDim] = solver.B(i, iDim);
    }
  }
}

#ifndef GPUCA_ALIROOT_LIB
template <typename DataT>
int Spline2DHelper<DataT>::test(const bool draw, const bool drawDataPoints)
{
  using namespace std;

  const int Ndim = 3;

  const int Fdegree = 4;

  double Fcoeff[Ndim][4 * (Fdegree + 1) * (Fdegree + 1)];

  constexpr int nKnots = 10;
  constexpr int nAuxiliaryPoints = 2;
  constexpr int uMax = nKnots; //* 3;

  auto F = [&](double u, double v, double Fuv[]) {
    const double scale = TMath::Pi() / uMax;
    double uu = u * scale;
    double vv = v * scale;
    double cosu[Fdegree + 1], sinu[Fdegree + 1], cosv[Fdegree + 1], sinv[Fdegree + 1];
    double ui = 0, vi = 0;
    for (int i = 0; i <= Fdegree; i++, ui += uu, vi += vv) {
      GPUCommonMath::SinCosd(ui, sinu[i], cosu[i]);
      GPUCommonMath::SinCosd(vi, sinv[i], cosv[i]);
    }
    for (int dim = 0; dim < Ndim; dim++) {
      double f = 0; // Fcoeff[dim][0]/2;
      for (int i = 1; i <= Fdegree; i++) {
        for (int j = 1; j <= Fdegree; j++) {
          double* c = &(Fcoeff[dim][4 * (i * Fdegree + j)]);
          f += c[0] * cosu[i] * cosv[j];
          f += c[1] * cosu[i] * sinv[j];
          f += c[2] * sinu[i] * cosv[j];
          f += c[3] * sinu[i] * sinv[j];
        }
      }
      Fuv[dim] = f;
    }
  };

  TCanvas* canv = nullptr;
  TNtuple* nt = nullptr;
  TNtuple* knots = nullptr;

  auto ask = [&]() -> bool {
    if (!canv) {
      return 0;
    }
    canv->Update();
    LOG(info) << "type 'q ' to exit";
    std::string str;
    std::getline(std::cin, str);
    return (str != "q" && str != ".q");
  };

  LOG(info) << "Test 2D interpolation with the compact spline";

  int nTries = 100;

  if (draw) {
    canv = new TCanvas("cQA", "Spline2D  QA", 1500, 800);
    nTries = 10000;
  }

  long double statDf = 0;
  long double statDf1D = 0;
  long double statN = 0;

  auto statTime = std::chrono::nanoseconds::zero();

  for (int seed = 1; seed < nTries + 1; seed++) {
    // LOG(info) << "next try.." ;

    gRandom->SetSeed(seed);

    for (int dim = 0; dim < Ndim; dim++) {
      for (int i = 0; i < 4 * (Fdegree + 1) * (Fdegree + 1); i++) {
        Fcoeff[dim][i] = gRandom->Uniform(-1, 1);
      }
    }

    Spline2D<DataT, Ndim> spline;

    int knotsU[nKnots], knotsV[nKnots];
    do {
      knotsU[0] = 0;
      knotsV[0] = 0;
      double du = 1. * uMax / (nKnots - 1);
      for (int i = 1; i < nKnots; i++) {
        knotsU[i] = (int)(i * du); // + gRandom->Uniform(-du / 3, du / 3);
        knotsV[i] = (int)(i * du); // + gRandom->Uniform(-du / 3, du / 3);
      }
      knotsU[nKnots - 1] = uMax;
      knotsV[nKnots - 1] = uMax;
      spline.recreate(nKnots, knotsU, nKnots, knotsV);

      if (nKnots != spline.getGridX1().getNumberOfKnots() ||
          nKnots != spline.getGridX2().getNumberOfKnots()) {
        LOG(info) << "warning: n knots changed during the initialisation " << nKnots
                  << " -> " << spline.getNumberOfKnots();
        continue;
      }
    } while (0);

    std::string err = FlatObject::stressTest(spline);
    if (!err.empty()) {
      LOG(info) << "error at FlatObject functionality: " << err;
      return -1;
    } else {
      // LOG(info) << "flat object functionality is ok" ;
    }

    // Ndim-D spline

    auto startTime = std::chrono::high_resolution_clock::now();
    spline.approximateFunctionViaDataPoints(0., uMax, 0., uMax, F, nAuxiliaryPoints, nAuxiliaryPoints);
    auto stopTime = std::chrono::high_resolution_clock::now();
    statTime += std::chrono::duration_cast<std::chrono::nanoseconds>(stopTime - startTime);

    //if (itry == 0)
    if (0) {
      TFile outf("testSpline2D.root", "recreate");
      if (outf.IsZombie()) {
        LOG(info) << "Failed to open output file testSpline2D.root ";
      } else {
        const char* name = "spline2Dtest";
        spline.writeToFile(outf, name);
        Spline2D<DataT, Ndim>* p = Spline2D<DataT, Ndim>::readFromFile(outf, name);
        if (p == nullptr) {
          LOG(info) << "Failed to read Spline1DOld from file testSpline1DOld.root ";
        } else {
          spline = *p;
        }
        outf.Close();
      }
    }

    // 1-D splines for each of Ndim dimensions

    Spline2D<DataT, 1> splines1D[Ndim];

    for (int dim = 0; dim < Ndim; dim++) {
      auto F1 = [&](double x1, double x2, double f[]) {
        double ff[Ndim];
        F(x1, x2, ff);
        f[0] = ff[dim];
      };
      splines1D[dim].recreate(nKnots, knotsU, nKnots, knotsV);
      splines1D[dim].approximateFunctionViaDataPoints(0., uMax, 0., uMax, F1, nAuxiliaryPoints, nAuxiliaryPoints);
    }

    double stepU = .1;
    for (double u = 0; u < uMax; u += stepU) {
      for (double v = 0; v < uMax; v += stepU) {
        double f[Ndim];
        F(u, v, f);
        DataT s[Ndim];
        spline.interpolate(u, v, s);
        for (int dim = 0; dim < Ndim; dim++) {
          statDf += (s[dim] - f[dim]) * (s[dim] - f[dim]);
          DataT s1 = splines1D[dim].interpolate(u, v);
          statDf1D += (s[dim] - s1) * (s[dim] - s1);
        }
        statN += Ndim;
        // LOG(info) << u << " " << v << ": f " << f << " s " << s << " df "
        //   << s - f << " " << sqrt(statDf / statN) ;
      }
    }
    // LOG(info) << "Spline2D standard deviation   : " << sqrt(statDf / statN)
    //   ;

    if (draw) {
      delete nt;
      delete knots;
      nt = new TNtuple("nt", "nt", "u:v:f:s");
      knots = new TNtuple("knots", "knots", "type:u:v:s");
      double stepU = .3;
      for (double u = 0; u < uMax; u += stepU) {
        for (double v = 0; v < uMax; v += stepU) {
          double f[Ndim];
          F(u, v, f);
          DataT s[Ndim];
          spline.interpolate(u, v, s);
          nt->Fill(u, v, f[0], s[0]);
        }
      }
      nt->SetMarkerStyle(8);

      nt->SetMarkerSize(.5);
      nt->SetMarkerColor(kBlue);
      nt->Draw("s:u:v", "", "");

      nt->SetMarkerColor(kGray);
      nt->SetMarkerSize(2.);
      nt->Draw("f:u:v", "", "same");

      nt->SetMarkerSize(.5);
      nt->SetMarkerColor(kBlue);
      nt->Draw("s:u:v", "", "same");

      for (int i = 0; i < nKnots; i++) {
        for (int j = 0; j < nKnots; j++) {
          double u = spline.getGridX1().getKnot(i).u;
          double v = spline.getGridX2().getKnot(j).u;
          DataT s[Ndim];
          spline.interpolate(u, v, s);
          knots->Fill(1, u, v, s[0]);
        }
      }

      knots->SetMarkerStyle(8);
      knots->SetMarkerSize(1.5);
      knots->SetMarkerColor(kRed);
      knots->SetMarkerSize(1.5);
      knots->Draw("s:u:v", "type==1", "same"); // knots

      if (drawDataPoints) {
        Spline2DHelper<DataT> helper;
        helper.setSpline(spline, 4, 4);
        for (int ipu = 0; ipu < helper.getHelperU1().getNumberOfDataPoints(); ipu++) {
          const typename Spline1DHelperOld<DataT>::DataPoint& pu = helper.getHelperU1().getDataPoint(ipu);
          for (int ipv = 0; ipv < helper.getHelperU2().getNumberOfDataPoints(); ipv++) {
            const typename Spline1DHelperOld<DataT>::DataPoint& pv = helper.getHelperU2().getDataPoint(ipv);
            if (pu.isKnot && pv.isKnot) {
              continue;
            }
            DataT s[Ndim];
            spline.interpolate(pu.u, pv.u, s);
            knots->Fill(2, pu.u, pv.u, s[0]);
          }
        }
        knots->SetMarkerColor(kBlack);
        knots->SetMarkerSize(1.);
        knots->Draw("s:u:v", "type==2", "same"); // data points
      }

      if (!ask()) {
        break;
      }
    }
  }
  // delete canv;
  // delete nt;
  // delete knots;

  statDf = sqrt(statDf / statN);
  statDf1D = sqrt(statDf1D / statN);

  LOG(info) << "\n std dev for Spline2D   : " << statDf;
  LOG(info) << " mean difference between 1-D and " << Ndim
            << "-D splines   : " << statDf1D;
  LOG(info) << " approximation time " << statTime.count() / 1000. / nTries << " ms";

  if (statDf < 0.15 && statDf1D < 1.e-20) {
    LOG(info) << "Everything is fine";
  } else {
    LOG(info) << "Something is wrong!!";
    return -2;
  }

  return 0;
}
#endif

template class GPUCA_NAMESPACE::gpu::Spline2DHelper<float>;
template class GPUCA_NAMESPACE::gpu::Spline2DHelper<double>;
