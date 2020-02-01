// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Spline1D.cxx
/// \brief Implementation of Spline1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "Spline1D.h"
#include <cmath>

#if !defined(GPUCA_GPUCODE) // code invisible on GPU
#include <vector>
#include <algorithm>
#include <iostream>
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "TRandom.h"
#include "Riostream.h"
#include "TMath.h"
#include "SplineHelper1D.h"
#include "TCanvas.h"
#include "TNtuple.h"
#include "TH1.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

void Spline1D::destroy()
{
  /// See FlatObject for description
  mNumberOfKnots = 0;
  mUmax = 0;
  mUtoKnotMap = nullptr;
  FlatObject::destroy();
}

#if !defined(GPUCA_GPUCODE)
void Spline1D::cloneFromObject(const Spline1D& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mNumberOfKnots = obj.mNumberOfKnots;
  mUmax = obj.mUmax;
  mUtoKnotMap = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mUtoKnotMap);
}

void Spline1D::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  mUtoKnotMap = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, mUtoKnotMap);
}
#endif

void Spline1D::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  mUtoKnotMap = FlatObject::relocatePointer(mFlatBufferPtr, actualFlatBufferPtr, mUtoKnotMap);
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
}

void Spline1D::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  mUtoKnotMap = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mUtoKnotMap);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

#if !defined(GPUCA_GPUCODE)

void Spline1D::constructKnots(int numberOfKnots, const int inputKnots[])
{
  /// Constructor
  ///
  /// Number of created knots may differ from the input values:
  /// - Edge knots {0} and {Umax} will be added if they are not present.
  /// - Duplicated knots, knots with a negative coordinate will be deleted
  /// - At least 2 knots will be created
  ///
  /// \param numberOfKnots     Number of knots in knots[] array
  /// \param knots             Array of knot positions (integer values)
  ///

  FlatObject::startConstruction();

  std::vector<int> knotU;

  { // reorganize knots

    std::vector<int> tmp;
    for (int i = 0; i < numberOfKnots; i++)
      tmp.push_back(inputKnots[i]);
    std::sort(tmp.begin(), tmp.end());

    knotU.push_back(0); // obligatory knot at 0.0

    for (int i = 0; i < numberOfKnots; ++i) {
      if (knotU.back() < tmp[i]) {
        knotU.push_back(tmp[i]);
      }
    }
    if (knotU.back() < 1) {
      knotU.push_back(1);
    }
  }

  mNumberOfKnots = knotU.size();
  mUmax = knotU.back();
  int uToKnotMapOffset = mNumberOfKnots * sizeof(Spline1D::Knot);

  FlatObject::finishConstruction(uToKnotMapOffset + (mUmax + 1) * sizeof(int));

  mUtoKnotMap = reinterpret_cast<int*>(mFlatBufferPtr + uToKnotMapOffset);

  Spline1D::Knot* s = getKnotsNonConst();

  for (int i = 0; i < mNumberOfKnots; i++) {
    s[i].u = knotU[i];
  }

  for (int i = 0; i < mNumberOfKnots - 1; i++) {
    s[i].Li = 1. / (s[i + 1].u - s[i].u); // do division in double
  }

  s[mNumberOfKnots - 1].Li = 0.f; // the value will not be used, we define it for consistency

  // Set up the map (integer U) -> (knot index)

  int* map = getUtoKnotMapNonConst();

  int iKnotMax = mNumberOfKnots - 2;

  //
  // With iKnotMax=nKnots-2 we map the U==Umax coordinate to the last [nKnots-2, nKnots-1] segment.
  // This trick allows one to avoid a special condition for this edge case.
  // Any U from [0,Umax] is mapped to some knot_i such, that the next knot_i+1 always exist
  //

  for (int u = 0, iKnot = 0; u <= mUmax; u++) {
    if ((knotU[iKnot + 1] == u) && (iKnot < iKnotMax)) {
      iKnot = iKnot + 1;
    }
    map[u] = iKnot;
  }
}

void Spline1D::constructKnotsRegular(int numberOfKnots)
{
  /// Constructor for a regular spline
  /// \param numberOfKnots     Number of knots
  ///

  if (numberOfKnots < 2) {
    numberOfKnots = 2;
  }

  std::vector<int> knots(numberOfKnots);
  for (int i = 0; i < numberOfKnots; i++) {
    knots[i] = i;
  }
  constructKnots(numberOfKnots, knots.data());
}
#endif

void Spline1D::print() const
{
#if !defined(GPUCA_GPUCODE)
  std::cout << " Compact Spline 1D: " << std::endl;
  std::cout << "  mNumberOfKnots = " << mNumberOfKnots << std::endl;
  std::cout << "  mUmax = " << mUmax << std::endl;
  std::cout << "  mUtoKnotMap = " << (void*)mUtoKnotMap << std::endl;
  std::cout << "  knots: ";
  for (int i = 0; i < mNumberOfKnots; i++) {
    std::cout << getKnot(i).u << " ";
  }
  std::cout << std::endl;
#endif
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation

int Spline1D::test(const bool draw, const bool drawDataPoints)
{
  using namespace std;

  const int Ndim = 5;

  const int Fdegree = 4;

  double Fcoeff[Ndim][2 * (Fdegree + 1)];

  int nKnots = 4;
  const int nAxiliaryPoints = 1;
  int uMax = nKnots * 3;

  auto F = [&](float u, float f[]) -> void {
    double uu = u * TMath::Pi() / uMax;
    for (int dim = 0; dim < Ndim; dim++) {
      f[dim] = 0; // Fcoeff[0]/2;
      for (int i = 1; i <= Fdegree; i++) {
        f[dim] += Fcoeff[dim][2 * i] * TMath::Cos(i * uu) +
                  Fcoeff[dim][2 * i + 1] * TMath::Sin(i * uu);
      }
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

  std::cout << "Test 1D interpolation with the compact spline" << std::endl;

  int nTries = 100;

  if (draw) {
    canv = new TCanvas("cQA", "Spline1D  QA", 2000, 1000);
    nTries = 10000;
  }

  double statDf1 = 0;
  double statDf2 = 0;
  double statDf1D = 0;
  double statN = 0;

  int seed = 1;
  for (int itry = 0; itry < nTries; itry++) {

    for (int dim = 0; dim < Ndim; dim++) {
      gRandom->SetSeed(seed++);
      for (int i = 0; i < 2 * (Fdegree + 1); i++) {
        Fcoeff[dim][i] = gRandom->Uniform(-1, 1);
      }
    }
    SplineHelper1D helper;
    Spline1D spline(2);

    do {
      int knotsU[nKnots];
      knotsU[0] = 0;
      double du = 1. * uMax / (nKnots - 1);
      for (int i = 1; i < nKnots; i++) {
        knotsU[i] = (int)(i * du); // + gRandom->Uniform(-du / 3, du / 3);
      }
      knotsU[nKnots - 1] = uMax;
      spline.constructKnots(nKnots, knotsU);

      if (nKnots != spline.getNumberOfKnots()) {
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

    nKnots = spline.getNumberOfKnots();

    helper.setSpline(spline, nAxiliaryPoints);

    std::unique_ptr<float[]> parameters1 =
      helper.constructParameters(Ndim, F, 0., spline.getUmax());

    std::unique_ptr<float[]> parameters2 =
      helper.constructParametersGradually(Ndim, F, 0., spline.getUmax());

    // 1-D splines for each dimension
    std::unique_ptr<float[]> parameters1D[Ndim];
    {
      std::vector<float> dataPoints(helper.getNumberOfDataPoints());
      for (int dim = 0; dim < Ndim; dim++) {
        parameters1D[dim].reset(new float[spline.getNumberOfParameters(1)]);
        for (int i = 0; i < helper.getNumberOfDataPoints(); i++) {
          float f[Ndim];
          F(helper.getDataPoint(i).u, f);
          dataPoints[i] = f[dim];
        }
        helper.constructParameters(1, dataPoints.data(), parameters1D[dim].get());
      }
    }

    float stepU = 1.e-2;
    for (double u = 0; u < uMax; u += stepU) {
      float f[Ndim], s1[Ndim], s2[Ndim];
      F(u, f);
      spline.interpolate(Ndim, parameters1.get(), u, s1);
      spline.interpolate(Ndim, parameters2.get(), u, s2);
      for (int dim = 0; dim < Ndim; dim++) {
        statDf1 += (s1[dim] - f[dim]) * (s1[dim] - f[dim]);
        statDf2 += (s2[dim] - f[dim]) * (s2[dim] - f[dim]);
        float s1D;
        spline.interpolate(1, parameters1D[dim].get(), u, &s1D);
        statDf1D += (s1D - s1[dim]) * (s1D - s1[dim]);
      }
      statN += Ndim;
    }
    // cout << "std dev Compact   : " << sqrt(statDf1 / statN) << std::endl;

    if (draw) {
      delete nt;
      delete knots;
      nt = new TNtuple("nt", "nt", "u:f:s");
      float drawMax = -1.e20;
      float drawMin = 1.e20;
      float stepU = 1.e-4;
      for (double u = 0; u < uMax; u += stepU) {
        float f[Ndim], s[Ndim];
        F(u, f);
        spline.interpolate(Ndim, parameters1.get(), u, s);
        nt->Fill(u, f[0], s[0]);
        drawMax = std::max(drawMax, std::max(f[0], s[0]));
        drawMin = std::min(drawMin, std::min(f[0], s[0]));
      }

      nt->SetMarkerStyle(8);

      {
        TNtuple* ntRange = new TNtuple("ntRange", "nt", "u:f");
        drawMin -= 0.1 * (drawMax - drawMin);

        ntRange->Fill(0, drawMin);
        ntRange->Fill(0, drawMax);
        ntRange->Fill(uMax, drawMin);
        ntRange->Fill(uMax, drawMax);
        ntRange->SetMarkerColor(kWhite);
        ntRange->SetMarkerSize(0.1);
        ntRange->Draw("f:u", "", "");
        delete ntRange;
      }

      nt->SetMarkerColor(kGray);
      nt->SetMarkerSize(2.);
      nt->Draw("f:u", "", "P,same");

      nt->SetMarkerSize(.5);
      nt->SetMarkerColor(kBlue);
      nt->Draw("s:u", "", "P,same");

      knots = new TNtuple("knots", "knots", "type:u:s");
      for (int i = 0; i < nKnots; i++) {
        double u = spline.getKnot(i).u;
        float s[Ndim];
        spline.interpolate(Ndim, parameters1.get(), u, s);
        knots->Fill(1, u, s[0]);
      }

      knots->SetMarkerStyle(8);
      knots->SetMarkerSize(1.5);
      knots->SetMarkerColor(kRed);
      knots->SetMarkerSize(1.5);
      knots->Draw("s:u", "type==1", "same"); // knots

      if (drawDataPoints) {
        for (int j = 0; j < helper.getNumberOfDataPoints(); j++) {
          const SplineHelper1D::DataPoint& p = helper.getDataPoint(j);
          if (p.isKnot) {
            continue;
          }
          float s[Ndim];
          spline.interpolate(Ndim, parameters1.get(), p.u, s);
          knots->Fill(2, p.u, s[0]);
        }
        knots->SetMarkerColor(kBlack);
        knots->SetMarkerSize(1.);
        knots->Draw("s:u", "type==2", "same"); // data points
      }

      if (!ask()) {
        break;
      }
    } // draw
  }
  //delete canv;
  //delete nt;
  //delete knots;

  statDf1 = sqrt(statDf1 / statN);
  statDf2 = sqrt(statDf2 / statN);
  statDf1D = sqrt(statDf1D / statN);

  cout << "\n std dev for Compact Spline   : " << statDf1 << " / " << statDf2
       << std::endl;
  cout << " mean difference between 1-D and " << Ndim
       << "-D splines   : " << statDf1D << std::endl;

  if (statDf1 < 0.05 && statDf2 < 0.06 && statDf1D < 1.e-20) {
    cout << "Everything is fine" << endl;
  } else {
    cout << "Something is wrong!!" << endl;
    return -2;
  }
  return 0;
}

#endif // GPUCA_GPUCODE
