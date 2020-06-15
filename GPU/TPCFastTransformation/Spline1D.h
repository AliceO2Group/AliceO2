// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Spline1D.h
/// \brief Definition of Spline1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE1D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE1D_H

#include "GPUCommonDef.h"
#include "FlatObject.h"
#if !defined(GPUCA_GPUCODE)
#include <functional>
#endif

class TFile;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The Spline1D class performs a cubic spline interpolation on an one-dimensional nonunifom grid.
///
/// The class is a flat C structure. It inherits from the FlatObjects.
/// No virtual methods, no ROOT types are used.
///
/// --- Interpolation ---
///
/// The spline approximates a function F:[xMin,xMax]->R^m
///
/// The function parameter is called X, the function value is called F. F value may be multi-dimensional.
/// The spline value is called S(x).
///
/// The interpolation is performed on N knots by 3-rd degree polynoms.
/// It uses spline values S_i and spline derivatives D_i==S'_i at the knots.
/// The polynoms and they first derivatives are always continuous at the knots.
/// Depending on the initialization of the derivatives D_i, the second derivative may or may not be continious.
///
/// --- Knots ---
///
/// Knot positions are not completelly irregular.
/// There is an internal scaled coordinate, called U, where all N knots have integer positions:
/// {U0==0, U1, .., Un-1==Umax}.
/// It is implemented this way for fast matching of any X value to its neighboring knots.
///
/// For example, three knots with U coordinates U_i={0, 3, 5},
/// being stretched on the X segment [0., 1.], will have X coordinates X_i={0., 3./5., 1.}
///
/// For a few reasons, it is better to minimize gaps between knots.
/// A spline with knots {0,4,8} is the same as the spline with knots {0,1,2},
/// but the later one uses less memory.
///
/// The minimal number of knots is 2.
///
/// ---- External storage of spline parameters for a given F ---
///
/// One can store all F-dependent spline parameters outside of the spline object
/// and provide them at each call of the interpolation.
/// To do so, create a spline with Fdimentions=0,
/// create F parameters via SplineHelper1D class, and use interpolateU(..) method for interpoltion.
///
/// This feature allows one to use the same spline object for approximation of different functions
/// on the same grid of knots.
///
/// ---- Creation of a spline ----
///
/// The splines are best-fit splines. It means, that spline values S_i and derivatives D_i at knots
/// are calibrated such, that they minimize the integral difference between S(x) and F(x).
/// This integral difference is caluclated at all integer values of U coordinate (in particular, at all knots)
/// and at extra nAxiliaryPoints points between the integers.
///
/// nAxiliaryPoints can be set as a parameter of approximateFunction() method.
/// With nAxiliaryPoints==3 the approximation accuracy is noticeably better than with 1 or 2.
/// Higher values usually give a little improvement over 3.
///
/// The value of nAxiliaryPoints has no influence on the interpolation speed,
/// it can only slow down the approximateFunction() method.
///
/// One can also create a spline in a classical way using corresponding method from SplineHelper1D.
///
/// ---- Example of creating a spline ----
///
///  auto F = [&](float x,float &f) {
///   f[0] = x*x+3.f; // F(x)
///  };
///
///  const int nKnots=3;
///  int knots[nKnots] = {0, 1, 5};
///  Spline1D<float> spline( nKnots, knots, 1 ); // prepare memory for 1-dimensional F
///
///  spline.approximateFunction(0., 1., F); // initialize spline to approximate F on a segment [0., 1.]
///
///  float f = spline.interpolate(0.2); // interpolated value at x==0.2
///
///  --- See also Spline1D::test() method for examples
///
///
template <typename DataT>
class Spline1D : public FlatObject
{
 public:
  ///
  /// \brief The struct Knot represents a knot(i) and the interval [ knot(i), knot(i+1) ]
  ///
  struct Knot {
    DataT u;  ///< u coordinate of the knot i (an integer number in float format)
    DataT Li; ///< inverse length of the [knot_i, knot_{i+1}] segment ( == 1.f/ a (small) integer number)
  };

  /// _____________  Version control __________________________

  /// Version number
  GPUhd() static constexpr int getVersion() { return 1; }

  /// _____________  Constructors / destructors __________________________

#if !defined(GPUCA_GPUCODE)
  /// Constructor for a regular spline.
  Spline1D(int numberOfKnots = 2, int nFDimensions = 1);

  /// Constructor for an irregular spline.
  Spline1D(int numberOfKnots, const int knots[], int nFDimensions);

  /// Copy constructor
  Spline1D(const Spline1D&);

  /// Assignment operator
  Spline1D& operator=(const Spline1D&);
#else
  /// Disable constructors for the GPU implementation
  Spline1D() CON_DELETE;
  Spline1D(const Spline1D&) CON_DELETE;
  Spline1D& operator=(const Spline1D&) CON_DELETE;
#endif

  /// Destructor
  ~Spline1D() CON_DEFAULT;

  /// _______________  Construction interface  ________________________

#if !defined(GPUCA_GPUCODE)
  /// Constructor for a regular spline.
  void recreate(int numberOfKnots, int nFDimensions);

  /// Constructor for an irregular spline
  void recreate(int numberOfKnots, const int knots[], int nFDimensions);
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// approximate a function F with this spline.
  void approximateFunction(DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[/*mFdimensions*/])> F,
                           int nAxiliaryDataPoints = 4);
#endif

  /// _______________  Main functionality   ________________________

  /// Get interpolated value for F(x)
  GPUhd() void interpolate(DataT x, GPUgeneric() DataT Sx[/*mFdimensions*/]) const;

  /// Get interpolated value for the first dimension of F(x). (Simplified interface for 1D)
  GPUhd() DataT interpolate(DataT x) const;

  /// _______________  IO   ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// write a class object to the file
  int writeToFile(TFile& outf, const char* name);

  /// read a class object from the file
  static Spline1D* readFromFile(TFile& inpf, const char* name);
#endif

  /// ================ Expert tools   ================================

  /// Get interpolated value {S(u): 1D -> nFdim} at the segment [knotL, next knotR]
  /// using the spline values Sl, Sr and the slopes Dl, Dr
  template <typename T>
  GPUhd() static void interpolateU(int nFdim, const Knot& knotL,
                                   GPUgeneric() const T Sl[/*nFdim*/], GPUgeneric() const T Dl[/*nFdim*/],
                                   GPUgeneric() const T Sr[/*nFdim*/], GPUgeneric() const T Dr[/*nFdim*/],
                                   DataT u, GPUgeneric() T Su[/*nFdim*/]);

  /// Get interpolated value for an nFdim-dimensional F(u) using spline parameters Fparameters.
  /// Fparameters can be created via SplineHelper1D. A border check for u is performed.
  GPUhd() void interpolateU(int nFdim, GPUgeneric() const DataT Fparameters[],
                            DataT u, GPUgeneric() DataT Su[/*nFdim*/]) const;

  /// Get interpolated value for an nFdim-dimensional F(u) using spline parameters Fparameters.
  /// Fparameters can be created via SplineHelper1D. No border check for u is performed.
  GPUhd() void interpolateUnonSafe(int nFdim, GPUgeneric() const DataT Fparameters[],
                                   DataT u, GPUgeneric() DataT Su[/*nFdim*/]) const;

  /// _______________  Getters   ________________________

  /// Get U coordinate of the last knot
  GPUhd() int getUmax() const { return mUmax; }

  /// Get minimal required alignment for the spline parameters
  static size_t getParameterAlignmentBytes(int nFdim)
  {
    size_t s = 2 * sizeof(DataT) * nFdim;
    return (s < 16) ? s : 16;
  }

  /// Size of the parameter array in bytes
  GPUhd() size_t getSizeOfParameters(int nFdim) const
  {
    return sizeof(DataT) * (size_t)getNumberOfParameters(nFdim);
  }

  /// Number of parameters
  GPUhd() int getNumberOfParameters(int nFdim) const
  {
    return (2 * nFdim) * mNumberOfKnots;
  }

  /// Get number of knots
  GPUhd() int getNumberOfKnots() const { return mNumberOfKnots; }

  /// Get the array of knots
  GPUhd() const Spline1D::Knot* getKnots() const { return reinterpret_cast<const Spline1D::Knot*>(mFlatBufferPtr); }

  /// Get i-th knot
  GPUhd() const Spline1D::Knot& getKnot(int i) const { return getKnots()[i < 0 ? 0 : (i >= mNumberOfKnots ? mNumberOfKnots - 1 : i)]; }

  /// Get index of an associated knot for a given U coordinate. Performs a border check.
  GPUhd() int getKnotIndexU(DataT u) const;

  /// Get number of F dimensions
  GPUhd() int getFdimensions() const { return mFdimensions; }

  /// Get number of F parameters
  GPUhd() DataT* getFparameters() { return mFparameters; }

  /// Get number of F parameters
  GPUhd() const DataT* getFparameters() const { return mFparameters; }

  /// _______________  Getters with no border check   ________________________

  /// Get i-th knot. No border check performed!
  GPUhd() const Spline1D::Knot& getKnotNonSafe(int i) const { return getKnots()[i]; }

  /// Get index of an associated knot for a given U coordinate. No border check preformed!
  GPUhd() int getKnotIndexUnonSafe(DataT u) const;

  /// _______________  Technical stuff   ________________________

  /// Get a map (integer U -> corresponding knot index)
  GPUhd() const int* getUtoKnotMap() const { return mUtoKnotMap; }

  /// Convert X coordinate to U
  GPUhd() DataT convXtoU(DataT x) const { return (x - mXmin) * mXtoUscale; }

  /// Convert U coordinate to X
  GPUhd() DataT convUtoX(DataT u) const { return mXmin + u / mXtoUscale; }

  /// Get Xmin
  GPUhd() DataT getXmin() const { return mXmin; }

  /// Get XtoUscale
  GPUhd() DataT getXtoUscale() const { return mXtoUscale; }

  /// Set X range
  void setXrange(DataT xMin, DataT xMax);

  /// Print method
  void print() const;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
  /// Test the class functionality
  static int test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Memory alignment

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

  /// Construction interface

#if !defined(GPUCA_GPUCODE)
  void cloneFromObject(const Spline1D& obj, char* newFlatBufferPtr);
#endif
  void destroy();

  /// Making the parameter buffer external

  using FlatObject::releaseInternalBuffer;

#if !defined(GPUCA_GPUCODE)
  void moveBufferTo(char* newBufferPtr);
#endif

  /// Moving the class with its external buffer to another location

  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

 private:
  /// Non-const accessor to knots array
  Spline1D::Knot* getKnots() { return reinterpret_cast<Spline1D::Knot*>(mFlatBufferPtr); }

  /// Non-const accessor to U->knots map
  int* getUtoKnotMap() { return mUtoKnotMap; }

  /// _____________  Data members  ____________

  int mNumberOfKnots;  ///< n knots on the grid
  int mUmax;           ///< U of the last knot
  DataT mXmin;         ///< X of the first knot
  DataT mXtoUscale;    ///< a scaling factor to convert X to U
  int mFdimensions;    ///< number of F dimensions
  int* mUtoKnotMap;    //! (transient!!) pointer to (integer U -> knot index) map inside the mFlatBufferPtr array
  DataT* mFparameters; //! (transient!!) F-dependent parameters of the spline
#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(Spline1D<DataT>, 1);
#endif
};

///
/// ========================================================================================================
///       Inline implementations of some methods
/// ========================================================================================================
///

#if !defined(GPUCA_GPUCODE)
template <typename DataT>
Spline1D<DataT>::Spline1D(int numberOfKnots, int nFdimensions)
  : FlatObject()
{
  recreate(numberOfKnots, nFdimensions);
}

template <typename DataT>
Spline1D<DataT>::Spline1D(int numberOfKnots, const int knots[], int nFdimensions)
  : FlatObject()
{
  recreate(numberOfKnots, knots, nFdimensions);
}

template <typename DataT>
Spline1D<DataT>::Spline1D(const Spline1D& spline)
  : FlatObject()
{
  cloneFromObject(spline, nullptr);
}

template <typename DataT>
Spline1D<DataT>& Spline1D<DataT>::operator=(const Spline1D& spline)
{
  cloneFromObject(spline, nullptr);
  return *this;
}
#endif

template <typename DataT>
GPUhd() void Spline1D<DataT>::interpolate(DataT x, GPUgeneric() DataT Sx[/*mFdimensions*/]) const
{
  /// Get interpolated value for F(x)
  interpolateU(mFdimensions, mFparameters, convXtoU(x), Sx);
}

template <typename DataT>
GPUhd() DataT Spline1D<DataT>::interpolate(DataT x) const
{
  /// Simplified interface for 1D: get interpolated value for the first dimension of F(x)
  DataT u = convXtoU(x);
  int iknot = getKnotIndexU(u);
  const DataT* d = mFparameters + (2 * iknot) * mFdimensions;
  DataT S = 0.;
  interpolateU(1, getKnotNonSafe(iknot), &(d[0]), &(d[mFdimensions]),
               &(d[2 * mFdimensions]), &(d[3 * mFdimensions]), u, &S);
  return S;
}

template <typename DataT>
template <typename T>
GPUhd() void Spline1D<DataT>::interpolateU(int nFdim, const Spline1D<DataT>::Knot& knotL,
                                           GPUgeneric() const T Sl[/*nFdim*/], GPUgeneric() const T Dl[/*nFdim*/],
                                           GPUgeneric() const T Sr[/*nFdim*/], GPUgeneric() const T Dr[/*nFdim*/],
                                           DataT u, GPUgeneric() T Su[/*nFdim*/])
{
  /// A static method.
  /// Gives interpolated value of N-dimensional S(u) at u
  /// input: Sl,Dl,Sr,Dr[nFdim] - N-dim function values and slopes at knots {knotL,knotR}
  /// output: Su[nFdim] - N-dim interpolated value for S(u)

  T uu = T(u - knotL.u);
  T li = T(knotL.Li);
  T v = uu * li; // scaled u
  for (int dim = 0; dim < nFdim; ++dim) {
    T df = (Sr[dim] - Sl[dim]) * li;
    T a = Dl[dim] + Dr[dim] - df - df;
    T b = df - Dl[dim] - a;
    Su[dim] = ((a * v + b) * v + Dl[dim]) * uu + Sl[dim];
  }

  /* another way to calculate f(u):
  T uu = T(u - knotL.u);
  T v = uu * T(knotL.Li); // scaled u
  T vm1 = v-1;
  T v2 = v * v;
  float cSr = v2*(3-2*v);
  float cSl = 1-cSr;
  float cDl = v*vm1*vm1*knotL.L;
  float cDr = v2*vm1*knotL.L;
  return cSl*Sl + cSr*Sr + cDl*Dl + cDr*Dr;
  */
}

template <typename DataT>
GPUhdi() void Spline1D<DataT>::interpolateU(int nFdim, GPUgeneric() const DataT parameters[], DataT u, GPUgeneric() DataT Su[/*nFdim*/]) const
{
  /// Get interpolated value for F(u) using given spline parameters with a border check
  int iknot = getKnotIndexU(u);
  const DataT* d = parameters + (2 * iknot) * nFdim;
  interpolateU(nFdim, getKnotNonSafe(iknot), &(d[0]), &(d[nFdim]), &(d[2 * nFdim]), &(d[3 * nFdim]), u, Su);
}

template <typename DataT>
GPUhdi() void Spline1D<DataT>::interpolateUnonSafe(int nFdim, GPUgeneric() const DataT parameters[], DataT u, GPUgeneric() DataT Su[/*nFdim*/]) const
{
  /// Get interpolated value for F(u) using given spline parameters without border check
  int iknot = getKnotIndexUnonSafe(u);
  const DataT* d = parameters + (2 * iknot) * nFdim;
  interpolateU(nFdim, getKnotNonSafe(iknot), &(d[0]), &(d[nFdim]), &(d[2 * nFdim]), &(d[3 * nFdim]), u, Su);
}

template <typename DataT>
GPUhdi() int Spline1D<DataT>::getKnotIndexU(DataT u) const
{
  /// Get i: u is in [knot_i, knot_{i+1}) interval
  /// when u is otside of [0, mUmax], return the edge intervals
  int iu = (int)u;
  if (iu < 0) {
    iu = 0;
  }
  if (iu > mUmax) {
    iu = mUmax;
  }
  return getUtoKnotMap()[iu];
}

template <typename DataT>
GPUhdi() int Spline1D<DataT>::getKnotIndexUnonSafe(DataT u) const
{
  /// Get i: u is in [knot_i, knot_{i+1}) interval
  /// no border check! u must be in [0,mUmax]
  return getUtoKnotMap()[(int)u];
}

template <typename DataT>
void Spline1D<DataT>::setXrange(DataT xMin, DataT xMax)
{
  mXmin = xMin;
  mXtoUscale = mUmax / (((double)xMax) - xMin);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
