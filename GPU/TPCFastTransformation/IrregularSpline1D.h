// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  IrregularSpline1D.h
/// \brief Definition of IrregularSpline1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_IRREGULARSPLINE1D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_IRREGULARSPLINE1D_H

#include "GPUCommonDef.h"
#include "FlatObject.h"
#include "GPUCommonDef.h"

#ifndef __OPENCL__
#include <cstddef>
#include <memory>
#include <cstring>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The IrregularSpline1D class represents one-dimensional spline interpolation on nonunifom (irregular) grid.
///
/// The class is flat C structure. No virtual methods, no ROOT types are used.
/// It is designed for spline parameterisation of TPC transformation.
///
/// ---
/// The spline interpolates a generic function F:[0,1]->R.
///
/// Let's call the function parameter U, the function value F.
/// The interpolation is performed on n knots {U0==0., U1, .., Un==1.}
/// with given function values {F0, ..., Fn} at the knots.
///
/// An interpolation in each interval between two knots is performed by 3-th degree polynom.
/// The polynoms cross the Fi values at the knots and have contnious 1-th derivative.
/// For performance reasons, the polynoms are created locally on-the-fly
/// using only four values of F at four neighbouring knots.
/// Therefore unlike the classical splines, the second derivative may not be continious.
///
/// The knots should belong to interval [0,1], the distance between knots is (almost) arbitrary.
///
/// Nothing which depends on F is stored in the class.
/// Therefore one can use the same class for different F functions.
/// The function values {F0,..Fn} have to be provided by user for each call.
///
/// The class performs a fast search of a spline interval: (float U ) -> [int iKnot, int iKnot+1 ).
/// For that purpose, initial U coordinates of the knots are rounded to the closest i*1./nAxisBins values.
///
/// The minimal number of knots is 5, the minimal number of axis bins is 4
///
/// Knots U0=0. and Un=1. are always present. They are added automatically when they are not set by an user.
///
/// Number of knots and they U coordinates may change during initialisation!
///
/// User should provide function values Fi for all !constructed! knots.
///
/// ------------ Edge correction ------------
///
/// The function values at both edges should be corrected beforehand via spline.correctEdges( f ); method.
/// It is needed for the fast spline mathematics to work correctly.
///
/// Explanation:
///
/// To calculate a spline at interval [Ui,U{i+1}), we need to know the function values at 4 knots: {i-1,i,i+1,i+2}
/// As the knots {-1} and {n} do not exist, the edge intervals [U0,U1) and [U{n-2},U{n-1}] need special treatment.
/// Thus, the algorithm needs to have 3 big branches containing different math :(
///
/// To avoid the branches in the code, we do a trick:
///
/// Function values for the interval [U0,U1) are constructed using a spline polynom from the next interval [U1,U2).
/// To do so, all U values from the first interval are assigned to the second interval [U1,U2) in the U->(knot interval) map.
///
/// This approach has no branches, but has another problem: the spline from the second interval [U1,U2)
///  will not necessarily cross the original F0 value at u=U0.
///
/// To fix this, we modify the function value F0 in the way, that the spline
/// from [U1,U2) also crosses the original F0 value at u=U0
///
/// The same trick is performed for the last interval [U{n-2},U{n-1})
///
/// ------------
///
///
///  Example of creating a spline:
///
///  const int nKnots=5;
///  float knots[nKnots] = {0., 0.25, 0.5, 0.7, 1.};
///  IrregularSpline1D spline;
///  spline.construct(nKnots, knots, 4);
///  float f[nKnots] = { 3.5, 2.0, 1.4, 3.8, 2.3};
///  spline.correctEdges( f );
///  spline.getSpline( f, 0. ); // == 3.5
///  spline.getSpline( f, 0.1 ); // == some interpolated value
///  spline.getSpline( f, 0.25 ); // == 2.0
///  spline.getSpline( f, 0.5  ); // == 1.4
///  spline.getSpline( f, 1.  ); // == 2.3
///
class IrregularSpline1D : public FlatObject
{
 public:
  ///
  /// \brief The struct represents a knot(i) and interval [ knot(i), knot(i+1) ]
  ///
  struct Knot {
    float u; ///< u coordinate of the knot i

    /// some useful values for spline calculation:

    float scale;   ///< scale for u = inverse length of a  segment [i, i+1]:  1./L[i,i+1]
    float scaleL0; ///< a coefficient at f(i-1) for f'(i) calculation
    float scaleL2; ///< a coefficient at f(i+1) for f'(i) calculation
    float scaleR2; ///< a coefficient at f(i+1) for f'(i+1) calculation
    float scaleR3; ///< a coefficient at f(i+2) for f'(i+1) calculation
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor. Creates an empty uninitialised object
  IrregularSpline1D();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject instead
  IrregularSpline1D(const IrregularSpline1D&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject instead
  IrregularSpline1D& operator=(const IrregularSpline1D&) CON_DELETE;

  /// Destructor
  ~IrregularSpline1D() CON_DEFAULT;

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Memory alignment

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

  /// Construction interface

  void cloneFromObject(const IrregularSpline1D& obj, char* newFlatBufferPtr);
  void destroy();

  /// Making the data buffer external

  using FlatObject::releaseInternalBuffer;
#ifndef GPUCA_GPUCODE
  using FlatObject::moveBufferTo;
#endif

  /// Moving the class with its external buffer to another location

  using FlatObject::setActualBufferAddress;
  using FlatObject::setFutureBufferAddress;

  /// _______________  Construction interface  ________________________

  /// Constructor
  ///
  /// Number of knots created and their values may differ from the input values:
  /// - Edge knots 0.f and 1.f will be added if they are not present.
  /// - Knot values are rounded to closest axis bins: k*1./numberOfAxisBins.
  /// - Knots which are too close to each other will be merged
  /// - At least 5 knots and at least 4 axis bins will be created for consistency reason
  ///
  /// \param numberOfKnots     Number of knots in knots[] array
  /// \param knots             Array of knots.
  /// \param numberOfAxisBins Number of axis bins to map U coordinate to
  ///                          an appropriate [knot(i),knot(i+1)] interval.
  ///                          The knot positions have a "granularity" of 1./numberOfAxisBins
  ///
  void construct(int numberOfKnots, const float knots[], int numberOfAxisBins);

  /// Constructor for a regular spline
  void constructRegular(int numberOfKnotsU);

  /// _______________  Main functionality   ________________________

  /// Correction of data values at both edge knots.
  ///
  /// It is needed for the fast spline mathematics to work correctly. See explanation at the class comment above.
  ///
  /// \param data array of function values. It has the size of getNumberOfKnots()
  template <typename T>
  GPUd() void correctEdges(T* data) const;

  /// Get interpolated value for f(u) using spline at knot "knot1" and function values at knots {knot_0,knot_1,knot_2,knot_3}
  template <typename T>
  GPUd() static T getSpline(const IrregularSpline1D::Knot& knot1, T f0, T f1, T f2, T f3, float u);

  /// Get interpolated value for f(u) using data array correctedData[getNumberOfKnots()] with corrected edges
  template <typename T>
  GPUd() T getSpline(const T correctedData[], float u) const;

  /// Get number of knots
  GPUd() int getNumberOfKnots() const { return mNumberOfKnots; }

  /// Get index of associated knot for a given U coordinate.
  ///
  /// Note: U values from the first interval are mapped to the second inrerval.
  /// Values from the last interval are mapped to the previous interval.
  ///
  GPUd() int getKnotIndex(float u) const;

  /// Get i-th knot, no border check performed!
  GPUd() const IrregularSpline1D::Knot& getKnot(int i) const { return getKnots()[i]; }

  /// Get array of knots
  GPUd() const IrregularSpline1D::Knot* getKnots() const { return reinterpret_cast<const IrregularSpline1D::Knot*>(mFlatBufferPtr); }

  /// Get minimal required alignment for the class
  static constexpr size_t getClassAlignmentBytes() { return 8; }

  /// Get minimal required alignment for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return 8; }

  /// Get minimal required alignment for the spline data
  static constexpr size_t getDataAlignmentBytes() { return 8; }

  /// technical stuff

  /// Get a map  (U axis bin index) -> (corresponding knot index)
  GPUd() const int* getBin2KnotMap() const { return reinterpret_cast<const int*>(mFlatBufferPtr + mBin2KnotMapOffset); }

  /// Get number of axis bins
  int getNumberOfAxisBins() const { return mNumberOfAxisBins; }

  /// Get coefficients for edge correction
  ///
  /// Let's the interpolated function has values f0, f1, f2, f3 at knots u0, u1, u2, u3
  /// The corrected value of f0 is calculated as:
  /// f0_corr = c0*f0 + c1*f1 + c2*f2 + c3*f3
  ///
  /// The coefficients ci are calculated in double precision because they are temporary
  /// and used only at the initialisation phase. So we can pay a price for the higher accuracy here.
  ///
  GPUd() static void getEdgeCorrectionCoefficients(double u0, double u1, double u2, double u3, double& c0, double& c1, double& c2, double& c3);

  /// Print method
  void Print() const;

 private:
  /// Non-const accessor to knots array
  IrregularSpline1D::Knot* getKnotsNonConst() { return reinterpret_cast<IrregularSpline1D::Knot*>(mFlatBufferPtr); }

  /// Non-const accessor to bins->knots map
  int* getBin2KnotMapNonConst() { return reinterpret_cast<int*>(mFlatBufferPtr + mBin2KnotMapOffset); }

  ///
  /// ====  Data members   ====
  ///

  int mNumberOfKnots;              ///< n knots on the grid
  int mNumberOfAxisBins;           ///< number of axis bins
  unsigned int mBin2KnotMapOffset; ///< pointer to (axis bin) -> (knot) map in mFlatBufferPtr array
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

template <typename T>
GPUdi() T IrregularSpline1D::getSpline(const IrregularSpline1D::Knot& knot1, T f0, T f1, T f2, T f3, float u)
{
  /// static method
  /// Get interpolated value for f(u) using spline at knot "knot1" and function values at knots {knot_0,knot_1,knot_2,knot_3}

  f0 -= f1;
  f2 -= f1;
  f3 -= f1;

  T x = T((u - knot1.u) * knot1.scale);              // scaled u
  T z1 = T(f0 * knot1.scaleL0 + f2 * knot1.scaleL2); // scaled u derivative at the knot 1
  T z2 = T(f2 * knot1.scaleR2 + f3 * knot1.scaleR3); // scaled u derivative at the knot 2

  T x2 = x * x;

  // f(x) = ax^3 + bx^2 + cx + d
  //
  // f(0) = f1
  // f(1) = f2
  // f'(0) = z1
  // f'(1) = z2
  //

  // T d = f1;
  // T c = z1;
  T a = -f2 - f2 + z1 + z2;
  T b = f2 - z1 - a;
  return a * x * x2 + b * x2 + z1 * x + f1;
}

template <typename T>
GPUdi() T IrregularSpline1D::getSpline(const T correctedData[], float u) const
{
  /// Get interpolated value for f(u) using data array correctedData[getNumberOfKnots()] with corrected edges
  int iknot = getKnotIndex(u);
  const IrregularSpline1D::Knot& knot = getKnot(iknot);
  const T* f = correctedData + iknot - 1;
  return getSpline(knot, f[0], f[1], f[2], f[3], u);
}

GPUdi() int IrregularSpline1D::getKnotIndex(float u) const
{
  /// get i: u is in [knot_i, knot_{i+1})
  int ibin = (int)(u * mNumberOfAxisBins);
  if (ibin < 0) {
    ibin = 0;
  }
  if (ibin > mNumberOfAxisBins - 1) {
    ibin = mNumberOfAxisBins - 1;
  }
  return getBin2KnotMap()[ibin];
}

GPUdi() void IrregularSpline1D::getEdgeCorrectionCoefficients(double u0, double u1, double u2, double u3, double& c0, double& c1, double& c2, double& c3)
{
  /// static method
  /// get edge  correction for f(u0):
  /// f0corr = c0*f0 + c1*f1 + c2*f2 + c3*f3

  double du = u2 - u1;
  double x0 = (u0 - u1) / du;
  // double x1 = ( u1 - u1 )/du = 0
  // double x2 = ( u2 - u1 )/du = 1
  double x3 = (u3 - u1) / du;

  double cL0 = -1. / (x0 * (x0 - 1.));
  double cL2 = x0 / (x0 - 1.);
  double cR2 = (x3 - 2.) / (x3 - 1.);
  double cR3 = 1. / (x3 * (x3 - 1.));

  // Fi = fi - f1

  // z1 = F0corr*cL0 + F2*cL2; // scaled u derivative at the knot 1
  // z2 = F2*cR2 + F3*cR3; // scaled u derivative at the knot 2

  // a =  -2F2 +   z1 + z2
  // b =   3F2 - 2*z1 - z2 ;
  // c =           z1
  // d = 0

  // F(x) = ax^3 + bx^2 + cx + d
  // F(x) = (-2*x^3+3*x^2)*F2 + (x^3-2*x^2+x)*z1 + (x^3-x^2)*z2
  // F(x) = (-2*x^3+3*x^2)*F2 + (x^3-2*x^2+x)*z1 + (x^3-x^2)*(cR2*F2 + cR3*F3)
  // F(x) = (-2*x^3+3*x^2+(x^3-x^2)*cR2)*F2 + (x^3-2*x^2+x)*z1 + (x^3-x^2)*cR3*F3
  // F(x) = x^2(-2*x+3+(x-1)*cR2)*F2 + x*(x-1)^2*z1 + x^2(x-1)*cR3*F3

  // z1*x*(x-1)^2 = F(x) - x^2(-2*x+3+(x-1)*cR2)*F2 - x^2(x-1)*cR3*F3
  // z1*x0*(x0-1)^2 = F0 - x0^2(-2*x0+3+(x0-1)*cR2)*F2 - x0^2(x0-1)*cR3*F3

  double x01 = x0 - 1.;

  // coeff. for z1 at F0, F1, F2, F3:

  c0 = 1. / (x0 * x01 * x01);
  c1 = 0;
  c2 = -(-2 * x0 + 3. + x01 * cR2) * x0 / (x01 * x01);
  c3 = -x0 * cR3 / x01;

  // F0corr = (z1 - F2*cL2)/cL0;

  c0 = c0 / cL0;
  c1 = c1 / cL0;
  c2 = (c2 - cL2) / cL0;
  c3 = c3 / cL0;

  // F0corr = c0*F0 + c1*F1 + C2*F2 + c3*F3;
  // f0corr-f1 = c0*(f0-f1) + c1*(f1-f1) + c2*(f2-f1) + c3*(f3-f1);
  // f0corr-f1 = c0*(f0-f1) + c2*(f2-f1) + c3*(f3-f1);
  // f0corr = c0*f0 + c2*f2 + c3*f3 + (1-c0-c2-c3)*f1;

  c1 = 1. - c0 - c2 - c3;
}

template <typename T>
GPUdi() void IrregularSpline1D::correctEdges(T* data) const
{
  const IrregularSpline1D::Knot* s = getKnots();
  double c0, c1, c2, c3;
  getEdgeCorrectionCoefficients(s[0].u, s[1].u, s[2].u, s[3].u, c0, c1, c2, c3);
  data[0] = c0 * data[0] + c1 * data[1] + c2 * data[2] + c3 * data[3];
  int i = mNumberOfKnots - 1;
  getEdgeCorrectionCoefficients(s[i - 0].u, s[i - 1].u, s[i - 2].u, s[i - 3].u, c0, c1, c2, c3);
  data[i] = c0 * data[i - 0] + c1 * data[i - 1] + c2 * data[i - 2] + c3 * data[i - 3];
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
