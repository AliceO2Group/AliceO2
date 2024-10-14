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

/// \file  IrregularSpline1D.cxx
/// \brief Implementation of IrregularSpline1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "IrregularSpline1D.h"
#include "GPUCommonLogger.h"

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

void IrregularSpline1D::construct(int32_t numberOfKnots, const float inputKnots[], int32_t numberOfAxisBins)
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

  std::vector<int32_t> vKnotBins;

  { // reorganize knots

    int32_t lastBin = numberOfAxisBins; // last bin starts with U value 1.f, therefore it is outside of the [0.,1.] interval

    vKnotBins.push_back(0); // obligatory knot at 0.0

    for (int32_t i = 0; i < numberOfKnots; ++i) {
      int32_t bin = (int32_t)roundf(inputKnots[i] * numberOfAxisBins);
      if (bin <= vKnotBins.back() || bin >= lastBin) {
        continue; // same knot
      }
      vKnotBins.push_back(bin);
    }

    vKnotBins.push_back(lastBin); // obligatory knot at 1.0

    if (vKnotBins.size() < 5) { // too less knots, make a grid with 5 knots
      vKnotBins.clear();
      vKnotBins.push_back(0);
      vKnotBins.push_back((int32_t)roundf(0.25 * numberOfAxisBins));
      vKnotBins.push_back((int32_t)roundf(0.50 * numberOfAxisBins));
      vKnotBins.push_back((int32_t)roundf(0.75 * numberOfAxisBins));
      vKnotBins.push_back(lastBin);
    }
  }

  mNumberOfKnots = vKnotBins.size();
  mNumberOfAxisBins = numberOfAxisBins;
  mBin2KnotMapOffset = mNumberOfKnots * sizeof(IrregularSpline1D::Knot);

  FlatObject::finishConstruction(mBin2KnotMapOffset + (numberOfAxisBins + 1) * sizeof(int32_t));

  IrregularSpline1D::Knot* s = getKnotsNonConst();

  for (int32_t i = 0; i < mNumberOfKnots; i++) {
    s[i].u = vKnotBins[i] / ((double)mNumberOfAxisBins); // do division in double
  }

  { // values will not be used, we define them for consistency
    int32_t i = 0;
    double du = (s[i + 1].u - s[i].u);
    double x3 = (s[i + 2].u - s[i].u) / du;
    s[i].scale = 1. / du;
    s[i].scaleL0 = 0.; // undefined
    s[i].scaleL2 = 0.; // undefined
    s[i].scaleR2 = (x3 - 2.) / (x3 - 1.);
    s[i].scaleR3 = 1. / (x3 * (x3 - 1.));
  }

  for (int32_t i = 1; i < mNumberOfKnots - 2; i++) {
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
    int32_t i = mNumberOfKnots - 2;
    double du = (s[i + 1].u - s[i].u);
    double x0 = (s[i - 1].u - s[i].u) / du;
    s[i].scale = 1. / du;
    s[i].scaleL0 = -1. / (x0 * (x0 - 1.));
    s[i].scaleL2 = x0 / (x0 - 1.);
    s[i].scaleR2 = 0; // undefined
    s[i].scaleR3 = 0; // undefined
  }

  { // values will not be used, we define them for consistency
    int32_t i = mNumberOfKnots - 1;
    s[i].scale = 0;   // undefined
    s[i].scaleL0 = 0; // undefined
    s[i].scaleL2 = 0; // undefined
    s[i].scaleR2 = 0; // undefined
    s[i].scaleR3 = 0; // undefined
  }

  // Set up map (U bin) -> (knot index)

  int32_t* map = getBin2KnotMapNonConst();

  int32_t iKnotMin = 1;
  int32_t iKnotMax = mNumberOfKnots - 3;

  //
  // With iKnotMin=1, iKnotMax=nKnots-3 we release edge intervals:
  //
  // Map U coordinates from the first segment [knot0,knot1] to the second segment [knot1,knot2].
  // Map coordinates from segment [knot{n-2},knot{n-1}] to the previous segment [knot{n-3},knot{n-4}]
  //
  // This trick allows one to use splines without special conditions for edge cases.
  // Any U from [0,1] is mapped to some knote i, where i-1, i, i+1, and i+2 knots always exist
  //

  for (int32_t iBin = 0, iKnot = iKnotMin; iBin <= mNumberOfAxisBins; iBin++) {
    if ((iKnot < iKnotMax) && vKnotBins[iKnot + 1] == iBin) {
      iKnot = iKnot + 1;
    }
    map[iBin] = iKnot;
  }
}

void IrregularSpline1D::constructRegular(int32_t numberOfKnots)
{
  /// Constructor for a regular spline
  /// \param numberOfKnots     Number of knots
  ///

  if (numberOfKnots < 5) {
    numberOfKnots = 5;
  }

  std::vector<float> knots(numberOfKnots);
  double du = 1. / (numberOfKnots - 1.);
  for (int32_t i = 1; i < numberOfKnots - 1; i++) {
    knots[i] = i * du;
  }
  knots[0] = 0.f;
  knots[numberOfKnots - 1] = 1.f;
  construct(numberOfKnots, knots.data(), numberOfKnots);
}

void IrregularSpline1D::print() const
{
#if !defined(GPUCA_GPUCODE)
  LOG(info) << " Irregular Spline 1D: ";
  LOG(info) << "  mNumberOfKnots = " << mNumberOfKnots;
  LOG(info) << "  mNumberOfAxisBins = " << mNumberOfAxisBins;
  LOG(info) << "  mBin2KnotMapOffset = " << mBin2KnotMapOffset;
  LOG(info) << "  knots: ";
  for (int32_t i = 0; i < mNumberOfKnots; i++) {
    LOG(info) << getKnot(i).u << " ";
  }
  LOG(info);
#endif
}
