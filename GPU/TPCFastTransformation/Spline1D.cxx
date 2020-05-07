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

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "Rtypes.h"
#endif

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
#include "TFile.h"
#include "GPUCommonMath.h"

templateClassImp(GPUCA_NAMESPACE::gpu::Spline1D);

#endif

using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT>
void Spline1D<DataT>::destroy()
{
  /// See FlatObject for description
  mNumberOfKnots = 0;
  mUmax = 0;
  mFdimensions = 0;
  mXmin = 0.;
  mXtoUscale = 1.;
  mUtoKnotMap = nullptr;
  mFparameters = nullptr;
  FlatObject::destroy();
}

#if !defined(GPUCA_GPUCODE)
template <typename DataT>
void Spline1D<DataT>::cloneFromObject(const Spline1D& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mNumberOfKnots = obj.mNumberOfKnots;
  mUmax = obj.mUmax;
  mFdimensions = obj.mFdimensions;
  mXmin = obj.mXmin;
  mXtoUscale = obj.mXtoUscale;
  mUtoKnotMap = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mUtoKnotMap);
  mFparameters = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mFparameters);
}

template <typename DataT>
void Spline1D<DataT>::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  mUtoKnotMap = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, mUtoKnotMap);
  mFparameters = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, mFparameters);
}
#endif

template <typename DataT>
void Spline1D<DataT>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  //mUtoKnotMap = FlatObject::relocatePointer(mFlatBufferPtr, actualFlatBufferPtr, mUtoKnotMap);
  //mFparameters = FlatObject::relocatePointer(mFlatBufferPtr, actualFlatBufferPtr, mFparameters);

  const int uToKnotMapOffset = mNumberOfKnots * sizeof(Spline1D::Knot);
  const int parametersOffset = uToKnotMapOffset + (mUmax + 1) * sizeof(int);
  //int bufferSize = parametersOffset + getSizeOfParameters(mFdimensions);

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  mUtoKnotMap = reinterpret_cast<int*>(mFlatBufferPtr + uToKnotMapOffset);
  mFparameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);
}

template <typename DataT>
void Spline1D<DataT>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  mUtoKnotMap = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mUtoKnotMap);
  mFparameters = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mFparameters);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

#if !defined(GPUCA_GPUCODE)

template <typename DataT>
void Spline1D<DataT>::recreate(int numberOfKnots, const int inputKnots[], int nFdimensions)
{
  /// Main constructor for an irregular spline
  ///
  /// Number of created knots may differ from the input values:
  /// - Duplicated knots will be deleted
  /// - At least 2 knots will be created
  ///
  /// \param numberOfKnots     Number of knots in knots[] array
  /// \param knots             Array of relative knot positions (integer values)
  ///

  FlatObject::startConstruction();

  std::vector<int> knotU;

  { // sort knots
    std::vector<int> tmp;
    for (int i = 0; i < numberOfKnots; i++) {
      tmp.push_back(inputKnots[i]);
    }
    std::sort(tmp.begin(), tmp.end());

    knotU.push_back(0); //  first knot at 0

    for (unsigned int i = 1; i < tmp.size(); ++i) {
      int u = tmp[i] - tmp[0];
      if (knotU.back() < u) { // remove duplicated knots
        knotU.push_back(u);
      }
    }
    if (knotU.back() < 1) { // at least 2 knots
      knotU.push_back(1);
    }
  }

  mNumberOfKnots = knotU.size();
  mUmax = knotU.back();
  mFdimensions = nFdimensions;
  mXmin = 0.;
  mXtoUscale = 1.;

  const int uToKnotMapOffset = mNumberOfKnots * sizeof(Spline1D::Knot);
  const int parametersOffset = uToKnotMapOffset + (mUmax + 1) * sizeof(int);
  const int bufferSize = parametersOffset + getSizeOfParameters(mFdimensions);

  FlatObject::finishConstruction(bufferSize);

  mUtoKnotMap = reinterpret_cast<int*>(mFlatBufferPtr + uToKnotMapOffset);
  mFparameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);

  Knot* s = getKnots();

  for (int i = 0; i < mNumberOfKnots; i++) {
    s[i].u = knotU[i];
  }

  for (int i = 0; i < mNumberOfKnots - 1; i++) {
    s[i].Li = 1. / (s[i + 1].u - s[i].u); // do division in double
  }

  s[mNumberOfKnots - 1].Li = 0.; // the value will not be used, we define it for consistency

  // Set up the map (integer U) -> (knot index)

  int* map = getUtoKnotMap();

  const int iKnotMax = mNumberOfKnots - 2;

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

  for (int i = 0; i < getNumberOfParameters(mFdimensions); i++) {
    mFparameters[i] = 0.f;
  }
}

template <typename DataT>
void Spline1D<DataT>::recreate(int numberOfKnots, int nFdimensions)
{
  /// Constructor for a regular spline
  /// \param numberOfKnots     Number of knots

  if (numberOfKnots < 2) {
    numberOfKnots = 2;
  }

  std::vector<int> knots(numberOfKnots);
  for (int i = 0; i < numberOfKnots; i++) {
    knots[i] = i;
  }
  recreate(numberOfKnots, knots.data(), nFdimensions);
}

#endif

template <typename DataT>
void Spline1D<DataT>::print() const
{
  printf(" Compact Spline 1D: \n");
  printf("  mNumberOfKnots = %d \n", mNumberOfKnots);
  printf("  mUmax = %d\n", mUmax);
  printf("  mUtoKnotMap = %p \n", (void*)mUtoKnotMap);
  printf("  knots: ");
  for (int i = 0; i < mNumberOfKnots; i++) {
    printf("%d ", (int)getKnot(i).u);
  }
  printf("\n");
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)
template <typename DataT>
int Spline1D<DataT>::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  return FlatObject::writeToFile(*this, outf, name);
}

template <typename DataT>
Spline1D<DataT>* Spline1D<DataT>::readFromFile(TFile& inpf, const char* name)
{
  /// read a class object from the file
  return FlatObject::readFromFile<Spline1D<DataT>>(inpf, name);
}

template <typename DataT>
void Spline1D<DataT>::approximateFunction(DataT xMin, DataT xMax,
                                          std::function<void(DataT x, DataT f[/*mFdimensions*/])> F,
                                          int nAxiliaryDataPoints)
{
  /// Approximate F with this spline
  setXrange(xMin, xMax);
  SplineHelper1D<DataT> helper;
  helper.approximateFunction(*this, xMin, xMax, F, nAxiliaryDataPoints);
}

template <typename DataT>
int Spline1D<DataT>::test(const bool draw, const bool drawDataPoints)
{
  using namespace std;

  // input function F

  const int Ndim = 5;
  const int Fdegree = 4;
  double Fcoeff[Ndim][2 * (Fdegree + 1)];

  auto F = [&](DataT x, DataT f[]) -> void {
    double cosx[Fdegree + 1], sinx[Fdegree + 1];
    double xi = 0;
    for (int i = 0; i <= Fdegree; i++, xi += x) {
      GPUCommonMath::SinCos(xi, sinx[i], cosx[i]);
    }
    for (int dim = 0; dim < Ndim; dim++) {
      f[dim] = 0; // Fcoeff[0]/2;
      for (int i = 1; i <= Fdegree; i++) {
        f[dim] += Fcoeff[dim][2 * i] * cosx[i] + Fcoeff[dim][2 * i + 1] * sinx[i];
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
    canv = new TCanvas("cQA", "Spline1D  QA", 1000, 600);
    nTries = 10000;
  }

  double statDf1 = 0;
  double statDf2 = 0;
  double statDf1D = 0;
  double statN = 0;

  int seed = 1;

  for (int itry = 0; itry < nTries; itry++) {

    // init random F
    for (int dim = 0; dim < Ndim; dim++) {
      gRandom->SetSeed(seed++);
      for (int i = 0; i < 2 * (Fdegree + 1); i++) {
        Fcoeff[dim][i] = gRandom->Uniform(-1, 1);
      }
    }

    // spline

    int nKnots = 4;
    const int uMax = nKnots * 3;

    Spline1D spline1;
    int knotsU[nKnots];

    do { // set knots randomly
      knotsU[0] = 0;
      double du = 1. * uMax / (nKnots - 1);
      for (int i = 1; i < nKnots; i++) {
        knotsU[i] = (int)(i * du); // + gRandom->Uniform(-du / 3, du / 3);
      }
      knotsU[nKnots - 1] = uMax;
      spline1.recreate(nKnots, knotsU, Ndim);
      if (nKnots != spline1.getNumberOfKnots()) {
        cout << "warning: n knots changed during the initialisation " << nKnots
             << " -> " << spline1.getNumberOfKnots() << std::endl;
        continue;
      }
    } while (0);

    std::string err = FlatObject::stressTest(spline1);
    if (!err.empty()) {
      cout << "error at FlatObject functionality: " << err << endl;
      return -1;
    } else {
      // cout << "flat object functionality is ok" << endl;
    }

    nKnots = spline1.getNumberOfKnots();
    int nAxiliaryPoints = 1;
    Spline1D spline2; //(spline1);
    spline2.cloneFromObject(spline1, nullptr);

    spline1.approximateFunction(0., TMath::Pi(), F, nAxiliaryPoints);

    //if (itry == 0)
    {
      TFile outf("testSpline1D.root", "recreate");
      if (outf.IsZombie()) {
        cout << "Failed to open output file testSpline1D.root " << std::endl;
      } else {
        const char* name = "spline1Dtest";
        spline1.writeToFile(outf, name);
        Spline1D<DataT>* p = spline1.readFromFile(outf, name);
        if (p == nullptr) {
          cout << "Failed to read Spline1D from file testSpline1D.root " << std::endl;
        } else {
          spline1 = *p;
        }
        outf.Close();
      }
    }

    SplineHelper1D<DataT> helper;
    helper.setSpline(spline2, Ndim, nAxiliaryPoints);
    helper.approximateFunctionGradually(spline2, 0., TMath::Pi(), F, nAxiliaryPoints);

    // 1-D splines for each dimension
    Spline1D splines3[Ndim];
    {
      for (int dim = 0; dim < Ndim; dim++) {
        auto F3 = [&](DataT u, DataT f[]) -> void {
          DataT ff[Ndim];
          F(u, ff);
          f[0] = ff[dim];
        };
        splines3[dim].recreate(nKnots, knotsU, 1);
        splines3[dim].approximateFunction(0., TMath::Pi(), F3, nAxiliaryPoints);
      }
    }

    double stepX = 1.e-2;
    for (double x = 0; x < TMath::Pi(); x += stepX) {
      DataT f[Ndim], s1[Ndim], s2[Ndim];
      F(x, f);
      spline1.interpolate(x, s1);
      spline2.interpolate(x, s2);
      for (int dim = 0; dim < Ndim; dim++) {
        statDf1 += (s1[dim] - f[dim]) * (s1[dim] - f[dim]);
        statDf2 += (s2[dim] - f[dim]) * (s2[dim] - f[dim]);
        DataT s1D = splines3[dim].interpolate(x);
        statDf1D += (s1D - s1[dim]) * (s1D - s1[dim]);
      }
      statN += Ndim;
    }
    // cout << "std dev   : " << sqrt(statDf1 / statN) << std::endl;

    if (draw) {
      delete nt;
      delete knots;
      nt = new TNtuple("nt", "nt", "u:f:s");
      DataT drawMax = -1.e20;
      DataT drawMin = 1.e20;
      DataT stepX = 1.e-4;
      for (double x = 0; x < TMath::Pi(); x += stepX) {
        DataT f[Ndim], s[Ndim];
        F(x, f);
        spline1.interpolate(x, s);
        nt->Fill(spline1.convXtoU(x), f[0], s[0]);
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
        double u = spline1.getKnot(i).u;
        DataT s[Ndim];
        spline1.interpolate(spline1.convUtoX(u), s);
        knots->Fill(1, u, s[0]);
      }

      knots->SetMarkerStyle(8);
      knots->SetMarkerSize(1.5);
      knots->SetMarkerColor(kRed);
      knots->SetMarkerSize(1.5);
      knots->Draw("s:u", "type==1", "same"); // knots

      if (drawDataPoints) {
        for (int j = 0; j < helper.getNumberOfDataPoints(); j++) {
          const typename SplineHelper1D<DataT>::DataPoint& p = helper.getDataPoint(j);
          if (p.isKnot) {
            continue;
          }
          DataT s[Ndim];
          spline1.interpolate(spline1.convUtoX(p.u), s);
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

#endif // GPUCA_ALIGPUCODE

template class GPUCA_NAMESPACE::gpu::Spline1D<float>;
template class GPUCA_NAMESPACE::gpu::Spline1D<double>;
