// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  IrregularSpline1D.cxx
/// \brief Implementation of IrregularSpline1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "IrregularSpline1D.h"
#include <cmath>
#include <vector>

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

using namespace GPUCA_NAMESPACE::gpu;

IrregularSpline1D::IrregularSpline1D() : FlatObject(), mNumberOfKnots(0), mNumberOfAxisBins(0), mBin2KnotMapOffset(0)
{
  /// Default constructor. Creates an empty uninitialised object
}

void IrregularSpline1D::destroy()
{
  /// See FlatObject for description
  mNumberOfKnots = 0;
  mNumberOfAxisBins = 0;
  mBin2KnotMapOffset = 0;
  FlatObject::destroy();
}

void IrregularSpline1D::cloneFromObject(const IrregularSpline1D& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mNumberOfKnots = obj.mNumberOfKnots;
  mNumberOfAxisBins = obj.mNumberOfAxisBins;
  mBin2KnotMapOffset = obj.mBin2KnotMapOffset;
}

void IrregularSpline1D::construct(int numberOfKnots, const float inputKnots[], int numberOfAxisBins)
{
  /// Constructor.
  /// Initialises the spline with a grid with numberOfKnots knots in the interval [0,1]
  /// array inputKnots[] has numberOfKnots entries, ordered from 0. to 1.
  /// knots on the edges u==0. & u==1. are obligatory
  ///
  /// The number of knots created and their values may change during initialisation:
  /// - Edge knots 0.f and 1.f will be added if they are not present.
  /// - Knot values are rounded to closest axis bins: k*1./numberOfAxisBins.
  /// - Knots which are too close to each other will be merged
  /// - At least 5 knots and at least 4 axis bins will be created for consistency reason
  ///
  /// \param numberOfKnots     Number of knots in knots[] array
  /// \param knots             Array of knots.
  /// \param numberOfAxisBins Number of axis bins to map U coordinate to
  ///                          an appropriate [knot(i),knot(i+1)] interval.
  ///                          The knot positions have a "granularity" of 1./(numberOfAxisBins-1)
  ///

  numberOfAxisBins -= 1;

  FlatObject::startConstruction();

  if (numberOfAxisBins < 4) {
    numberOfAxisBins = 4;
  }

  std::vector<int> vKnotBins;

  { // reorganize knots

    int lastBin = numberOfAxisBins; // last bin starts with U value 1.f, therefore it is outside of the [0.,1.] interval

    vKnotBins.push_back(0); // obligatory knot at 0.0

    for (int i = 0; i < numberOfKnots; ++i) {
      int bin = (int)roundf(inputKnots[i] * numberOfAxisBins);
      if (bin <= vKnotBins.back() || bin >= lastBin) {
        continue; // same knot
      }
      vKnotBins.push_back(bin);
    }

    vKnotBins.push_back(lastBin); // obligatory knot at 1.0

    if (vKnotBins.size() < 5) { // too less knots, make a grid with 5 knots
      vKnotBins.clear();
      vKnotBins.push_back(0);
      vKnotBins.push_back((int)roundf(0.25 * numberOfAxisBins));
      vKnotBins.push_back((int)roundf(0.50 * numberOfAxisBins));
      vKnotBins.push_back((int)roundf(0.75 * numberOfAxisBins));
      vKnotBins.push_back(lastBin);
    }
  }

  mNumberOfKnots = vKnotBins.size();
  mNumberOfAxisBins = numberOfAxisBins;
  mBin2KnotMapOffset = mNumberOfKnots * sizeof(IrregularSpline1D::Knot);

  FlatObject::finishConstruction(mBin2KnotMapOffset + (numberOfAxisBins + 1) * sizeof(int));

  IrregularSpline1D::Knot* s = getKnotsNonConst();

  for (int i = 0; i < mNumberOfKnots; i++) {
    s[i].u = vKnotBins[i] / ((double)mNumberOfAxisBins); // do division in double
  }

  { // values will not be used, we define them for consistency
    int i = 0;
    double du = (s[i + 1].u - s[i].u);
    double x3 = (s[i + 2].u - s[i].u) / du;
    s[i].scale = 1. / du;
    s[i].scaleL0 = 0.; // undefined
    s[i].scaleL2 = 0.; // undefined
    s[i].scaleR2 = (x3 - 2.) / (x3 - 1.);
    s[i].scaleR3 = 1. / (x3 * (x3 - 1.));
  }

  for (int i = 1; i < mNumberOfKnots - 2; i++) {
    double du = (s[i + 1].u - s[i].u);
    double x0 = (s[i - 1].u - s[i].u) / du;
    double x3 = (s[i + 2].u - s[i].u) / du;
    s[i].scale = 1. / du;
    s[i].scaleL0 = -1. / (x0 * (x0 - 1.));
    s[i].scaleL2 = x0 / (x0 - 1.);
    s[i].scaleR2 = (x3 - 2.) / (x3 - 1.);
    s[i].scaleR3 = 1. / (x3 * (x3 - 1.));
  }

  { // values will not be used, we define them for consistency
    int i = mNumberOfKnots - 2;
    double du = (s[i + 1].u - s[i].u);
    double x0 = (s[i - 1].u - s[i].u) / du;
    s[i].scale = 1. / du;
    s[i].scaleL0 = -1. / (x0 * (x0 - 1.));
    s[i].scaleL2 = x0 / (x0 - 1.);
    s[i].scaleR2 = 0; // undefined
    s[i].scaleR3 = 0; // undefined
  }

  { // values will not be used, we define them for consistency
    int i = mNumberOfKnots - 1;
    s[i].scale = 0;   // undefined
    s[i].scaleL0 = 0; // undefined
    s[i].scaleL2 = 0; // undefined
    s[i].scaleR2 = 0; // undefined
    s[i].scaleR3 = 0; // undefined
  }

  // Set up map (U bin) -> (knot index)

  int* map = getBin2KnotMapNonConst();

  int iKnotMin = 1;
  int iKnotMax = mNumberOfKnots - 3;

  //
  // With iKnotMin=1, iKnotMax=nKnots-3 we release edge intervals:
  //
  // Map U coordinates from the first segment [knot0,knot1] to the second segment [knot1,knot2].
  // Map coordinates from segment [knot{n-2},knot{n-1}] to the previous segment [knot{n-3},knot{n-4}]
  //
  // This trick allows one to use splines without special conditions for edge cases.
  // Any U from [0,1] is mapped to some knote i, where i-1, i, i+1, and i+2 knots always exist
  //

  for (int iBin = 0, iKnot = iKnotMin; iBin <= mNumberOfAxisBins; iBin++) {
    if ((iKnot < iKnotMax) && vKnotBins[iKnot + 1] == iBin) {
      iKnot = iKnot + 1;
    }
    map[iBin] = iKnot;
  }
}

void IrregularSpline1D::constructRegular(int numberOfKnots)
{
  /// Constructor for a regular spline
  /// \param numberOfKnots     Number of knots
  ///

  if (numberOfKnots < 5)
    numberOfKnots = 5;

  std::vector<float> knots(numberOfKnots);
  double du = 1. / (numberOfKnots - 1.);
  for (int i = 1; i < numberOfKnots - 1; i++) {
    knots[i] = i * du;
  }
  knots[0] = 0.f;
  knots[numberOfKnots - 1] = 1.f;
  construct(numberOfKnots, knots.data(), numberOfKnots);
}

void IrregularSpline1D::Print() const
{
#if !defined(GPUCA_GPUCODE)
  std::cout << " Irregular Spline 1D: " << std::endl;
  std::cout << "  mNumberOfKnots = " << mNumberOfKnots << std::endl;
  std::cout << "  mNumberOfAxisBins = " << mNumberOfAxisBins << std::endl;
  std::cout << "  mBin2KnotMapOffset = " << mBin2KnotMapOffset << std::endl;
  std::cout << "  knots: ";
  for (int i = 0; i < mNumberOfKnots; i++) {
    std::cout << getKnot(i).u << " ";
  }
  std::cout << std::endl;
#endif
}
