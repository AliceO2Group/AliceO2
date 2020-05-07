// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Spline2D.cxx
/// \brief Implementation of Spline2D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "Rtypes.h"
#endif

#include "Spline2D.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "TRandom.h"
#include "Riostream.h"
#include "TMath.h"
#include "SplineHelper2D.h"
#include "TCanvas.h"
#include "TNtuple.h"
#include "TFile.h"
#include "GPUCommonMath.h"

templateClassImp(GPUCA_NAMESPACE::gpu::Spline2DBase);

#endif

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT, bool isConsistentT>
Spline2DBase<DataT, isConsistentT>::Spline2DBase(int nDim)
  : FlatObject(), mFdim(nDim), mGridU1(), mGridU2(), mFparameters(nullptr)
{
  recreate(2, 2);
}

template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::destroy()
{
  /// See FlatObject for description
  mGridU1.destroy();
  mGridU2.destroy();
  FlatObject::destroy();
}

template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);

  const size_t u2Offset = alignSize(mGridU1.getFlatBufferSize(), mGridU2.getBufferAlignmentBytes());
  int parametersOffset = u2Offset;
  //int bufferSize = parametersOffset;
  mFparameters = nullptr;

  if (isConsistentT) {
    parametersOffset = alignSize(u2Offset + mGridU2.getFlatBufferSize(), getParameterAlignmentBytes());
    //bufferSize = parametersOffset + getSizeOfParameters();
    mFparameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);
  }

  mGridU1.setActualBufferAddress(mFlatBufferPtr);
  mGridU2.setActualBufferAddress(mFlatBufferPtr + u2Offset);
}

template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  char* bufferU = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mGridU1.getFlatBufferPtr());
  char* bufferV = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mGridU2.getFlatBufferPtr());
  mGridU1.setFutureBufferAddress(bufferU);
  mGridU2.setFutureBufferAddress(bufferV);
  if (isConsistentT) {
    mFparameters = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mFparameters);
  } else {
    mFparameters = nullptr;
  }
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::print() const
{
  printf(" Irregular Spline 2D: \n");
  printf(" grid U1: \n");
  mGridU1.print();
  printf(" grid U2: \n");
  mGridU2.print();
}

#if !defined(GPUCA_GPUCODE)
template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::cloneFromObject(const Spline2DBase<DataT, isConsistentT>& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description
  if (isConsistentT && mFdim != obj.mFdim) {
    assert(0);
    return;
  }

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  char* bufferU = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGridU1.getFlatBufferPtr());
  char* bufferV = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGridU2.getFlatBufferPtr());

  mGridU1.cloneFromObject(obj.mGridU1, bufferU);
  mGridU2.cloneFromObject(obj.mGridU2, bufferV);

  if (isConsistentT) {
    mFparameters = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mFparameters);
  } else {
    mFparameters = nullptr;
  }
}

template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::recreate(
  int numberOfKnotsU1, const int knotsU1[], int numberOfKnotsU2, const int knotsU2[])
{
  /// Constructor for an irregular spline

  FlatObject::startConstruction();

  mGridU1.recreate(numberOfKnotsU1, knotsU1, 0);
  mGridU2.recreate(numberOfKnotsU2, knotsU2, 0);

  const size_t u2Offset = alignSize(mGridU1.getFlatBufferSize(), mGridU2.getBufferAlignmentBytes());
  int parametersOffset = u2Offset + mGridU2.getFlatBufferSize();
  int bufferSize = parametersOffset;
  mFparameters = nullptr;

  if (isConsistentT) {
    parametersOffset = alignSize(bufferSize, getParameterAlignmentBytes());
    bufferSize = parametersOffset + getSizeOfParameters();
  }

  FlatObject::finishConstruction(bufferSize);

  mGridU1.moveBufferTo(mFlatBufferPtr);
  mGridU2.moveBufferTo(mFlatBufferPtr + u2Offset);
  if (isConsistentT) {
    mFparameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);
    for (int i = 0; i < getNumberOfParameters(); i++) {
      mFparameters[i] = 0;
    }
  }
}

template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::recreate(
  int numberOfKnotsU1, int numberOfKnotsU2)
{
  /// Constructor for a regular spline

  FlatObject::startConstruction();

  mGridU1.recreate(numberOfKnotsU1, 0);

  mGridU2.recreate(numberOfKnotsU2, 0);

  const size_t u2Offset = alignSize(mGridU1.getFlatBufferSize(), mGridU2.getBufferAlignmentBytes());
  int parametersOffset = u2Offset + mGridU2.getFlatBufferSize();
  int bufferSize = parametersOffset;
  mFparameters = nullptr;

  if (isConsistentT) {
    parametersOffset = alignSize(bufferSize, getParameterAlignmentBytes());
    bufferSize = parametersOffset + getSizeOfParameters();
  }

  FlatObject::finishConstruction(bufferSize);

  mGridU1.moveBufferTo(mFlatBufferPtr);
  mGridU2.moveBufferTo(mFlatBufferPtr + u2Offset);

  if (isConsistentT) {
    mFparameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);
    for (int i = 0; i < getNumberOfParameters(); i++) {
      mFparameters[i] = 0;
    }
  }
}

#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB) // code invisible on GPU and in the standalone compilation and in AliRoot

template <typename DataT, bool isConsistentT>
int Spline2DBase<DataT, isConsistentT>::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  return FlatObject::writeToFile(*this, outf, name);
}

template <typename DataT, bool isConsistentT>
Spline2DBase<DataT, isConsistentT>* Spline2DBase<DataT, isConsistentT>::readFromFile(
  TFile& inpf, const char* name)
{
  /// read a class object from the file
  return FlatObject::readFromFile<Spline2DBase<DataT, isConsistentT>>(inpf, name);
}

template <typename DataT, bool isConsistentT>
void Spline2DBase<DataT, isConsistentT>::approximateFunction(
  DataT x1Min, DataT x1Max, DataT x2Min, DataT x2Max,
  std::function<void(DataT x1, DataT x2, DataT f[])> F,
  int nAxiliaryDataPointsU1, int nAxiliaryDataPointsU2)
{
  /// approximate a function F with this spline
  SplineHelper2D<DataT> helper;
  helper.approximateFunction(*this, x1Min, x1Max, x2Min, x2Max, F, nAxiliaryDataPointsU1, nAxiliaryDataPointsU2);
}

template <typename DataT, bool isConsistentT>
int Spline2DBase<DataT, isConsistentT>::test(const bool draw, const bool drawDataPoints)
{
  using namespace std;

  const int Ndim = 3;

  const int Fdegree = 4;

  double Fcoeff[Ndim][4 * (Fdegree + 1) * (Fdegree + 1)];

  constexpr int nKnots = 4;
  constexpr int nAxiliaryPoints = 1;
  constexpr int uMax = nKnots * 3;

  auto F = [&](DataT u, DataT v, DataT Fuv[]) {
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

      if (nKnots != spline.getGridU1().getNumberOfKnots() ||
          nKnots != spline.getGridU2().getNumberOfKnots()) {
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
          cout << "Failed to read Spline1D from file testSpline1D.root " << std::endl;
        } else {
          spline = *p;
        }
        outf.Close();
      }
    }

    // 1-D splines for each of Ndim dimensions

    Spline2D<DataT, 1> splines1D[Ndim];

    for (int dim = 0; dim < Ndim; dim++) {
      auto F1 = [&](DataT x1, DataT x2, DataT f[]) {
        DataT ff[Ndim];
        F(x1, x2, ff);
        f[0] = ff[dim];
      };
      splines1D[dim].recreate(nKnots, knotsU, nKnots, knotsV);
      splines1D[dim].approximateFunction(0., uMax, 0., uMax, F1, 4, 4);
    }

    double stepU = .1;
    for (double u = 0; u < uMax; u += stepU) {
      for (double v = 0; v < uMax; v += stepU) {
        DataT f[Ndim];
        F(u, v, f);
        DataT s[Ndim];
        DataT s1;
        spline.interpolate(u, v, s);
        for (int dim = 0; dim < Ndim; dim++) {
          statDf += (s[dim] - f[dim]) * (s[dim] - f[dim]);
          splines1D[dim].interpolate(u, v, &s1);
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
          DataT f[Ndim];
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
          double u = spline.getGridU1().getKnot(i).u;
          double v = spline.getGridU2().getKnot(j).u;
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
        SplineHelper2D<DataT> helper;
        helper.setSpline(spline, 4, 4);
        for (int ipu = 0; ipu < helper.getHelperU1().getNumberOfDataPoints(); ipu++) {
          const typename SplineHelper1D<DataT>::DataPoint& pu = helper.getHelperU1().getDataPoint(ipu);
          for (int ipv = 0; ipv < helper.getHelperU2().getNumberOfDataPoints(); ipv++) {
            const typename SplineHelper1D<DataT>::DataPoint& pv = helper.getHelperU2().getDataPoint(ipv);
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

#endif // GPUCA_GPUCODE

template class GPUCA_NAMESPACE::gpu::Spline2DBase<float, false>;
template class GPUCA_NAMESPACE::gpu::Spline2DBase<float, true>;
template class GPUCA_NAMESPACE::gpu::Spline2DBase<double, false>;
template class GPUCA_NAMESPACE::gpu::Spline2DBase<double, true>;
