// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Spline2DHelper.cxx
/// \brief Implementation of Spline2DHelper class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

#include "Spline2DHelper.h"
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
  for (unsigned int iDim = 0; iDim < mFdimensions; ++iDim) {
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
int Spline2DHelper<DataT>::test(const bool draw, const bool drawDataPoints)
{
  using namespace std;

  const int Ndim = 3;

  const int Fdegree = 4;

  double Fcoeff[Ndim][4 * (Fdegree + 1) * (Fdegree + 1)];

  constexpr int nKnots = 4;
  constexpr int nAuxiliaryPoints = 1;
  constexpr int uMax = nKnots * 3;

  auto F = [&](double u, double v, double Fuv[]) {
    const double scale = TMath::Pi() / uMax;
    double uu = u * scale;
    double vv = v * scale;
    double cosu[Fdegree + 1], sinu[Fdegree + 1], cosv[Fdegree + 1], sinv[Fdegree + 1];
    double ui = 0, vi = 0;
    for (int i = 0; i <= Fdegree; i++, ui += uu, vi += vv) {
      GPUCommonMath::SinCos(ui, sinu[i], cosu[i]);
      GPUCommonMath::SinCos(vi, sinv[i], cosv[i]);
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
    cout << "type 'q ' to exit" << endl;
    std::string str;
    std::getline(std::cin, str);
    return (str != "q" && str != ".q");
  };

  std::cout << "Test 2D interpolation with the compact spline" << std::endl;

  int nTries = 10;

  if (draw) {
    canv = new TCanvas("cQA", "Spline2D  QA", 1500, 800);
    nTries = 10000;
  }

  long double statDf = 0;
  long double statDf1D = 0;
  long double statN = 0;

  for (int seed = 1; seed < nTries + 1; seed++) {
    //cout << "next try.." << endl;

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
        cout << "warning: n knots changed during the initialisation " << nKnots
             << " -> " << spline.getNumberOfKnots() << std::endl;
        continue;
      }
    } while (0);

    std::string err = FlatObject::stressTest(spline);
    if (!err.empty()) {
      cout << "error at FlatObject functionality: " << err << endl;
      return -1;
    } else {
      // cout << "flat object functionality is ok" << endl;
    }

    // Ndim-D spline
    spline.approximateFunction(0., uMax, 0., uMax, F, 4, 4);

    //if (itry == 0)
    if (1) {
      TFile outf("testSpline2D.root", "recreate");
      if (outf.IsZombie()) {
        cout << "Failed to open output file testSpline2D.root " << std::endl;
      } else {
        const char* name = "spline2Dtest";
        spline.writeToFile(outf, name);
        Spline2D<DataT, Ndim>* p = Spline2D<DataT, Ndim>::readFromFile(outf, name);
        if (p == nullptr) {
          cout << "Failed to read Spline1DOld from file testSpline1DOld.root " << std::endl;
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
      splines1D[dim].approximateFunction(0., uMax, 0., uMax, F1, 4, 4);
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
        // cout << u << " " << v << ": f " << f << " s " << s << " df "
        //   << s - f << " " << sqrt(statDf / statN) << std::endl;
      }
    }
    // cout << "Spline2D standard deviation   : " << sqrt(statDf / statN)
    //   << std::endl;

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
          const typename Spline1DHelper<DataT>::DataPoint& pu = helper.getHelperU1().getDataPoint(ipu);
          for (int ipv = 0; ipv < helper.getHelperU2().getNumberOfDataPoints(); ipv++) {
            const typename Spline1DHelper<DataT>::DataPoint& pv = helper.getHelperU2().getDataPoint(ipv);
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

  cout << "\n std dev for Spline2D   : " << statDf << std::endl;
  cout << " mean difference between 1-D and " << Ndim
       << "-D splines   : " << statDf1D << std::endl;

  if (statDf < 0.15 && statDf1D < 1.e-20) {
    cout << "Everything is fine" << endl;
  } else {
    cout << "Something is wrong!!" << endl;
    return -2;
  }

  return 0;
}

template class GPUCA_NAMESPACE::gpu::Spline2DHelper<float>;
template class GPUCA_NAMESPACE::gpu::Spline2DHelper<double>;

#endif
