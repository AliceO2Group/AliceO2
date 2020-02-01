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
#endif

using namespace GPUCA_NAMESPACE::gpu;

void Spline2D::destroy()
{
  /// See FlatObject for description
  mGridU.destroy();
  mGridV.destroy();
  FlatObject::destroy();
}

#if !defined(GPUCA_GPUCODE)
void Spline2D::cloneFromObject(const Spline2D& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  char* bufferU = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGridU.getFlatBufferPtr());
  char* bufferV = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGridV.getFlatBufferPtr());

  mGridU.cloneFromObject(obj.mGridU, bufferU);
  mGridV.cloneFromObject(obj.mGridV, bufferV);
}

void Spline2D::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}
#endif

void Spline2D::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  char* bufferU = FlatObject::relocatePointer(mFlatBufferPtr, actualFlatBufferPtr, mGridU.getFlatBufferPtr());
  char* bufferV = FlatObject::relocatePointer(mFlatBufferPtr, actualFlatBufferPtr, mGridV.getFlatBufferPtr());
  mGridU.setActualBufferAddress(bufferU);
  mGridV.setActualBufferAddress(bufferV);
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
}

void Spline2D::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  char* bufferU = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mGridU.getFlatBufferPtr());
  char* bufferV = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mGridV.getFlatBufferPtr());
  mGridU.setFutureBufferAddress(bufferU);
  mGridV.setFutureBufferAddress(bufferV);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

#if !defined(GPUCA_GPUCODE)
void Spline2D::constructKnots(int numberOfKnotsU, const int knotsU[], int numberOfKnotsV, const int knotsV[])
{
  /// Constructor
  ///
  /// Number of created knots may differ from the input values:
  /// - Edge knots {0} and {Umax/Vmax} will be added if they are not present.
  /// - Duplicated knots, knots with a negative coordinate will be deleted
  /// - At least 2 knots for each axis will be created
  ///
  /// \param numberOfKnotsU     Number of knots in knotsU[] array
  /// \param knotsU             Array of knot positions (integer values)
  ///
  /// \param numberOfKnotsV     Number of knots in knotsV[] array
  /// \param knotsV             Array of knot positions (integer values)
  ///

  FlatObject::startConstruction();

  mGridU.constructKnots(numberOfKnotsU, knotsU);
  mGridV.constructKnots(numberOfKnotsV, knotsV);

  size_t vOffset = alignSize(mGridU.getFlatBufferSize(), mGridV.getBufferAlignmentBytes());

  FlatObject::finishConstruction(vOffset + mGridV.getFlatBufferSize());

  mGridU.moveBufferTo(mFlatBufferPtr);
  mGridV.moveBufferTo(mFlatBufferPtr + vOffset);
}

void Spline2D::constructKnotsRegular(int numberOfKnotsU, int numberOfKnotsV)
{
  /// Constructor for a regular spline
  /// \param numberOfKnotsU     U axis: Number of knots in knots[] array
  /// \param numberOfKnotsV     V axis: Number of knots in knots[] array
  ///

  FlatObject::startConstruction();

  mGridU.constructKnotsRegular(numberOfKnotsU);
  mGridV.constructKnotsRegular(numberOfKnotsV);

  size_t vOffset = alignSize(mGridU.getFlatBufferSize(), mGridV.getBufferAlignmentBytes());

  FlatObject::finishConstruction(vOffset + mGridV.getFlatBufferSize());

  mGridU.moveBufferTo(mFlatBufferPtr);
  mGridV.moveBufferTo(mFlatBufferPtr + vOffset);
}
#endif

void Spline2D::print() const
{
#if !defined(GPUCA_GPUCODE)
  std::cout << " Irregular Spline 2D: " << std::endl;
  std::cout << " grid U: " << std::endl;
  mGridU.print();
  std::cout << " grid V: " << std::endl;
  mGridV.print();
#endif
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation

int Spline2D::test(const bool draw, const bool drawDataPoints)
{
  using namespace std;

  const int Ndim = 3;

  const int Fdegree = 4;

  double Fcoeff[Ndim][4 * (Fdegree + 1) * (Fdegree + 1)];

  int nKnots = 4;
  const int nAxiliaryPoints = 1;
  int uMax = nKnots * 3;

  auto F = [&](float u, float v, float Fuv[]) -> void {
    double uu = u * TMath::Pi() / uMax;
    double vv = v * TMath::Pi() / uMax;
    for (int dim = 0; dim < Ndim; dim++) {
      double f = 0; // Fcoeff[dim][0]/2;
      for (int i = 1; i <= Fdegree; i++) {
        double cosu = TMath::Cos(i * uu);
        double sinu = TMath::Sin(i * uu);
        for (int j = 1; j <= Fdegree; j++) {
          double* c = &(Fcoeff[dim][4 * (i * Fdegree + j)]);
          double cosv = TMath::Cos(j * vv);
          double sinv = TMath::Sin(j * vv);
          f += c[0] * cosu * cosv;
          f += c[1] * cosu * sinv;
          f += c[2] * sinu * cosv;
          f += c[3] * sinu * sinv;
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
    canv = new TCanvas("cQA", "Spline1D  QA", 2000, 1000);
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

    GPUCA_NAMESPACE::gpu::SplineHelper2D helper;
    GPUCA_NAMESPACE::gpu::Spline2D spline;
    // spline.constructKnotsRegular(nKnots, nKnots);

    do {
      int knotsU[nKnots], knotsV[nKnots];
      knotsU[0] = 0;
      knotsV[0] = 0;
      double du = 1. * uMax / (nKnots - 1);
      for (int i = 1; i < nKnots; i++) {
        knotsU[i] = (int)(i * du); // + gRandom->Uniform(-du / 3, du / 3);
        knotsV[i] = (int)(i * du); // + gRandom->Uniform(-du / 3, du / 3);
      }
      knotsU[nKnots - 1] = uMax;
      knotsV[nKnots - 1] = uMax;
      spline.constructKnots(nKnots, knotsU, nKnots, knotsV);

      if (nKnots != spline.getGridU().getNumberOfKnots() ||
          nKnots != spline.getGridV().getNumberOfKnots()) {
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

    int err2 = helper.setSpline(spline, nAxiliaryPoints, nAxiliaryPoints);
    if (err2 != 0) {
      cout << "Error by spline construction: " << helper.getLastError()
           << std::endl;
      return -1;
    }

    // Ndim-D spline

    std::unique_ptr<float[]> parameters = helper.constructParameters(Ndim, F, 0., uMax, 0., uMax);

    // 1-D splines for each of Ndim dimensions

    std::unique_ptr<float[]> parameters1D[Ndim];
    {
      std::unique_ptr<float[]> dataPoints1D[Ndim];
      for (int dim = 0; dim < Ndim; dim++) {
        parameters1D[dim].reset(new float[spline.getNumberOfParameters(1)]);
        dataPoints1D[dim].reset(new float[helper.getNumberOfDataPoints()]);
      }

      int nPointsU = helper.getNumberOfDataPointsU();
      int nPointsV = helper.getNumberOfDataPointsV();
      for (int ipv = 0; ipv < nPointsV; ipv++) {
        float v = helper.getHelperV().getDataPoint(ipv).u;
        for (int ipu = 0; ipu < nPointsU; ipu++) {
          float u = helper.getHelperU().getDataPoint(ipu).u;
          float Fuv[Ndim];
          F(u, v, Fuv);
          int ip = ipv * nPointsU + ipu;
          for (int dim = 0; dim < Ndim; dim++) {
            dataPoints1D[dim][ip] = Fuv[dim];
          }
        }
      }

      for (int dim = 0; dim < Ndim; dim++) {
        helper.constructParameters(1, dataPoints1D[dim].get(), parameters1D[dim].get());
      }
    }

    double stepU = .1;
    for (double u = 0; u < uMax; u += stepU) {
      for (double v = 0; v < uMax; v += stepU) {
        float f[Ndim];
        F(u, v, f);
        float s[Ndim];
        float s1;
        spline.interpolate(Ndim, parameters.get(), u, v, s);
        for (int dim = 0; dim < Ndim; dim++) {
          statDf += (s[dim] - f[dim]) * (s[dim] - f[dim]);
          spline.interpolate(1, parameters1D[dim].get(), u, v, &s1);
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

      double stepU = .1;
      for (double u = 0; u < uMax; u += stepU) {
        for (double v = 0; v < uMax; v += stepU) {
          float f[Ndim];
          F(u, v, f);
          float s[Ndim];
          spline.interpolate(Ndim, parameters.get(), u, v, s);
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
          double u = spline.getGridU().getKnot(i).u;
          double v = spline.getGridV().getKnot(j).u;
          float s[Ndim];
          spline.interpolate(Ndim, parameters.get(), u, v, s);
          knots->Fill(1, u, v, s[0]);
        }
      }

      knots->SetMarkerStyle(8);
      knots->SetMarkerSize(1.5);
      knots->SetMarkerColor(kRed);
      knots->SetMarkerSize(1.5);
      knots->Draw("s:u:v", "type==1", "same"); // knots

      if (drawDataPoints) {
        for (int ipu = 0; ipu < helper.getHelperU().getNumberOfDataPoints(); ipu++) {
          const SplineHelper1D::DataPoint& pu = helper.getHelperU().getDataPoint(ipu);
          for (int ipv = 0; ipv < helper.getHelperV().getNumberOfDataPoints(); ipv++) {
            const SplineHelper1D::DataPoint& pv = helper.getHelperV().getDataPoint(ipv);
            if (pu.isKnot && pv.isKnot) {
              continue;
            }
            float s[Ndim];
            spline.interpolate(Ndim, parameters.get(), pu.u, pv.u, s);
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

  cout << "\n std dev for Compact Spline   : " << statDf << std::endl;
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
