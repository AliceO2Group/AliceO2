// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Spline182.cxx
/// \brief Implementation of SplineHelper2D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

#include "SplineHelper2D.h"
#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TDecompBK.h"

using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT>
SplineHelper2D<DataT>::SplineHelper2D() : mError(), mFdimensions(0), mHelperU1(), mHelperU2()
{
}

template <typename DataT>
int SplineHelper2D<DataT>::storeError(int code, const char* msg)
{
  mError = msg;
  return code;
}

template <typename DataT>
void SplineHelper2D<DataT>::approximateFunction(
  DataT* Fparameters, DataT x1Min, DataT x1Max, DataT x2Min, DataT x2Max,
  std::function<void(DataT x1, DataT x2, DataT f[/*spline.getFdimensions()*/])> F) const
{
  /// Create best-fit spline parameters for a given input function F
  /// output in Fparameters

  std::vector<DataT> dataPointF(getNumberOfDataPoints() * mFdimensions);

  double scaleX1 = (x1Max - x1Min) / ((double)mHelperU1.getSpline().getUmax());
  double scaleX2 = (x2Max - x2Min) / ((double)mHelperU2.getSpline().getUmax());

  for (int iv = 0; iv < getNumberOfDataPointsU2(); iv++) {
    DataT x2 = x2Min + mHelperU2.getDataPoint(iv).u * scaleX2;
    for (int iu = 0; iu < getNumberOfDataPointsU1(); iu++) {
      DataT x1 = x1Min + mHelperU1.getDataPoint(iu).u * scaleX1;
      F(x1, x2, &dataPointF[(iv * getNumberOfDataPointsU1() + iu) * mFdimensions]);
    }
  }
  approximateFunction(Fparameters, dataPointF.data());
}

template <typename DataT>
void SplineHelper2D<DataT>::approximateFunctionBatch(
  DataT* Fparameters, DataT x1Min, DataT x1Max, DataT x2Min, DataT x2Max,
  std::function<void(const std::vector<DataT>& x1, const std::vector<DataT>& x2, std::vector<DataT> f[/*mFdimensions*/])> F,
  unsigned int batchsize) const
{
  /// Create best-fit spline parameters for a given input function F.
  /// F calculates values for a batch of points.
  /// output in Fparameters

  std::vector<DataT> dataPointF(getNumberOfDataPoints() * mFdimensions);

  double scaleX1 = (x1Max - x1Min) / ((double)mHelperU1.getSpline().getUmax());
  double scaleX2 = (x2Max - x2Min) / ((double)mHelperU2.getSpline().getUmax());

  std::vector<DataT> x1;
  x1.reserve(batchsize);

  std::vector<DataT> x2;
  x2.reserve(batchsize);

  std::vector<int> index;
  index.reserve(batchsize);

  std::vector<DataT> dataPointFTmp[mFdimensions];
  for (unsigned int iDim = 0; iDim < mFdimensions; ++iDim) {
    dataPointFTmp[iDim].reserve(batchsize);
  }

  unsigned int counter = 0;
  for (int iv = 0; iv < getNumberOfDataPointsU2(); iv++) {
    DataT x2Tmp = x2Min + mHelperU2.getDataPoint(iv).u * scaleX2;
    for (int iu = 0; iu < getNumberOfDataPointsU1(); iu++) {
      DataT x1Tmp = x1Min + mHelperU1.getDataPoint(iu).u * scaleX1;
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
          for (unsigned int iDim = 0; iDim < mFdimensions; ++iDim) {
            dataPointF[indexTmp + iDim] = dataPointFTmp[iDim][i];
          }
        }

        x1.clear();
        x2.clear();
        index.clear();
        for (unsigned int iDim = 0; iDim < mFdimensions; ++iDim) {
          dataPointFTmp[iDim].clear();
        }
      }
    }
  }
  approximateFunction(Fparameters, dataPointF.data());
}

template <typename DataT>
void SplineHelper2D<DataT>::approximateFunction(
  DataT* Fparameters, const DataT DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
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

  std::unique_ptr<DataT[]> rotDataPointF(new DataT[nDataPointsU * nDataPointsV * Ndim]); // U DataPoints x V DataPoints :  rotated DataPointF for one output dimension
  std::unique_ptr<DataT[]> Dv(new DataT[nKnotsV * nDataPointsU * Ndim]);                 // V knots x U DataPoints

  std::unique_ptr<DataT[]> parU(new DataT[mHelperU1.getSpline().getNumberOfParameters(Ndim)]);
  std::unique_ptr<DataT[]> parV(new DataT[mHelperU2.getSpline().getNumberOfParameters(Ndim)]);

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
    const DataT* DataPointFrow = &(DataPointF[Ndim * ipv * nDataPointsU]);
    mHelperU1.approximateFunctionGradually(parU.get(), DataPointFrow);

    for (int iKnotU = 0; iKnotU < nKnotsU; ++iKnotU) {
      DataT* knotPar = &Fparameters[Ndim4 * (iKnotV * nKnotsU + iKnotU)];
      for (int dim = 0; dim < Ndim; ++dim) {
        knotPar[dim] = parU[Ndim * (2 * iKnotU) + dim];                // store S for all the knots
        knotPar[Ndim2 + dim] = parU[Ndim * (2 * iKnotU) + Ndim + dim]; // store S'u for all the knots //SG!!!
      }
    }

    // recalculate F values for all ipu DataPoints at V = ipv
    for (int ipu = 0; ipu < nDataPointsU; ipu++) {
      DataT splineF[Ndim];
      DataT u = mHelperU1.getDataPoint(ipu).u;
      mHelperU1.getSpline().interpolateU(Ndim, parU.get(), u, splineF);
      for (int dim = 0; dim < Ndim; dim++) {
        rotDataPointF[(ipu * nDataPointsV + ipv) * Ndim + dim] = splineF[dim];
      }
    }
  }

  // calculate S'v at all data points with V == V of a knot

  for (int ipu = 0; ipu < nDataPointsU; ipu++) {
    const DataT* DataPointFcol = &(rotDataPointF[ipu * nDataPointsV * Ndim]);
    mHelperU2.approximateFunctionGradually(parV.get(), DataPointFcol);
    for (int iKnotV = 0; iKnotV < nKnotsV; iKnotV++) {
      for (int dim = 0; dim < Ndim; dim++) {
        DataT dv = parV[(iKnotV * 2 + 1) * Ndim + dim];
        Dv[(iKnotV * nDataPointsU + ipu) * Ndim + dim] = dv;
      }
    }
  }

  // fit S'v and S''_vu at all the knots

  for (int iKnotV = 0; iKnotV < nKnotsV; ++iKnotV) {
    const DataT* Dvrow = &(Dv[iKnotV * nDataPointsU * Ndim]);
    mHelperU1.approximateFunction(parU.get(), Dvrow);
    for (int iKnotU = 0; iKnotU < nKnotsU; ++iKnotU) {
      for (int dim = 0; dim < Ndim; ++dim) {
        Fparameters[Ndim4 * (iKnotV * nKnotsU + iKnotU) + Ndim + dim] = parU[Ndim * 2 * iKnotU + dim];         // store S'v for all the knots
        Fparameters[Ndim4 * (iKnotV * nKnotsU + iKnotU) + Ndim3 + dim] = parU[Ndim * 2 * iKnotU + Ndim + dim]; // store S''vu for all the knots
      }
    }
  }
}

template class GPUCA_NAMESPACE::gpu::SplineHelper2D<float>;
template class GPUCA_NAMESPACE::gpu::SplineHelper2D<double>;

#endif
