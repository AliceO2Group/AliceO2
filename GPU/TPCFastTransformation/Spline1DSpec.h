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

/// \file  Spline1DSpec.h
/// \brief Definition of Spline1DSpec class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE1DSPEC_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE1DSPEC_H

#include "GPUCommonDef.h"
#include "FlatObject.h"
#include "SplineUtil.h"

#if !defined(GPUCA_GPUCODE)
#include <functional>
#endif

class TFile;

namespace o2
{
namespace gpu
{

/// ==================================================================================================
/// The class Spline1DContainer is a base class of Spline1D.
/// It contains all the class members and those methods which only depends on the DataT data type.
/// It also contains all non-inlined methods with the implementation in Spline1DSpec.cxx file.
///
/// DataT is a data type, which is supposed to be either double or float.
/// For other possible data types one has to add the corresponding instantiation line
/// at the end of the Spline1DSpec.cxx file
///
template <typename DataT>
class Spline1DContainer : public FlatObject
{
 public:
  /// Named enumeration for the safety level used by some methods
  enum SafetyLevel { kNotSafe,
                     kSafe };

  /// The struct Knot represents the i-th knot and the segment [knot_i, knot_i+1]
  ///
  struct Knot {
    DataT u;  ///< u coordinate of the knot i (an integer number in float format)
    DataT Li; ///< inverse length of the [knot_i, knot_{i+1}] segment ( == 1./ a (small) integer )
    /// Get u as an integer
    GPUd() int32_t getU() const { return (int32_t)(u + 0.1f); }
  };

  /// _____________  Version control __________________________

  /// Version control
  GPUd() static constexpr int32_t getVersion() { return 1; }

  /// _____________  C++ constructors / destructors __________________________

  /// Default constructor, required by the Root IO
  Spline1DContainer() = default;

  /// Disable all other constructors
  Spline1DContainer(const Spline1DContainer&) = delete;

  /// Destructor
  ~Spline1DContainer() = default;

  /// _______________  Construction interface  ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// approximate a function F with this spline
  void approximateFunction(double xMin, double xMax,
                           std::function<void(double x, double f[/*mYdim*/])> F,
                           int32_t nAuxiliaryDataPoints = 4);
#endif

  /// _______________  IO   ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// write a class object to the file
  int32_t writeToFile(TFile& outf, const char* name);

  /// read a class object from the file
  static Spline1DContainer* readFromFile(TFile& inpf, const char* name);
#endif

  /// _______________  Getters   ________________________

  /// Get U coordinate of the last knot
  GPUd() int32_t getUmax() const { return mUmax; }

  /// Get number of Y dimensions
  GPUd() int32_t getYdimensions() const { return mYdim; }

  /// Get minimal required alignment for the spline parameters
  GPUd() size_t getParameterAlignmentBytes() const
  {
    size_t s = 2 * sizeof(DataT) * mYdim;
    return (s < 16) ? s : 16;
  }

  /// Number of parameters
  GPUd() int32_t getNumberOfParameters() const { return calcNumberOfParameters(mYdim); }

  /// Size of the parameter array in bytes
  GPUd() size_t getSizeOfParameters() const { return sizeof(DataT) * getNumberOfParameters(); }

  /// Get a number of knots
  GPUd() int32_t getNumberOfKnots() const { return mNumberOfKnots; }

  /// Get the array of knots
  GPUd() const Knot* getKnots() const { return reinterpret_cast<const Knot*>(mFlatBufferPtr); }

  /// Get i-th knot
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() const Knot& getKnot(int32_t i) const
  {
    if (SafeT == SafetyLevel::kSafe) {
      i = (i < 0) ? 0 : (i >= mNumberOfKnots ? mNumberOfKnots - 1 : i);
    }
    return getKnots()[i];
  }

  /// Get index of an associated knot for a given U coordinate. Performs a boundary check.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() int32_t getLeftKnotIndexForU(DataT u) const;

  /// Get spline parameters
  GPUd() DataT* getParameters() { return mParameters; }

  /// Get spline parameters const
  GPUd() const DataT* getParameters() const { return mParameters; }

  /// _______________  Technical stuff  ________________________

  /// Get a map (integer U -> corresponding knot index)
  GPUd() const int32_t* getUtoKnotMap() const { return mUtoKnotMap; }

  /// Convert X coordinate to U
  GPUd() DataT convXtoU(DataT x) const { return (x - mXmin) * mXtoUscale; }

  /// Convert U coordinate to X
  GPUd() DataT convUtoX(DataT u) const { return mXmin + u / mXtoUscale; }

  /// Get Xmin
  GPUd() DataT getXmin() const { return mXmin; }

  /// Get Xmax
  GPUd() DataT getXmax() const { return mXmin + mUmax / mXtoUscale; }

  /// Get XtoUscale
  GPUd() DataT getXtoUscale() const { return mXtoUscale; }

  /// Set X range
  GPUd() void setXrange(DataT xMin, DataT xMax);

  /// Print method
  void print() const;

  ///  _______________  Expert tools  _______________

  /// Number of parameters for given Y dimensions
  GPUd() int32_t calcNumberOfParameters(int32_t nYdim) const { return (2 * nYdim) * getNumberOfKnots(); }

  ///_______________  Test tools  _______________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
  /// Test the class functionality
  static int32_t test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

#if !defined(GPUCA_GPUCODE)
  void cloneFromObject(const Spline1DContainer& obj, char* newFlatBufferPtr);
  void moveBufferTo(char* newBufferPtr);
#endif

  using FlatObject::releaseInternalBuffer;

  void destroy();
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

 protected:
  /// Non-const accessor to the knots array
  Knot* getKnots() { return reinterpret_cast<Knot*>(mFlatBufferPtr); }

  /// Non-const accessor to U->knots map
  int32_t* getUtoKnotMap() { return mUtoKnotMap; }

#if !defined(GPUCA_GPUCODE)
  /// Constructor for a regular spline
  void recreate(int32_t nYdim, int32_t numberOfKnots);

  /// Constructor for an irregular spline
  void recreate(int32_t nYdim, int32_t numberOfKnots, const int32_t knotU[]);
#endif

  /// _____________  Data members  ____________

  int32_t mYdim = 0;              ///< dimentionality of F
  int32_t mNumberOfKnots = 0;     ///< n knots on the grid
  int32_t mUmax = 0;              ///< U of the last knot
  DataT mXmin = 0;                ///< X of the first knot
  DataT mXtoUscale = 0;           ///< a scaling factor to convert X to U
  int32_t* mUtoKnotMap = nullptr; //! (transient!!) pointer to (integer U -> knot index) map inside the mFlatBufferPtr array
  DataT* mParameters = nullptr;   //! (transient!!) pointer to F-dependent parameters inside the mFlatBufferPtr array

  ClassDefNV(Spline1DContainer, 1);
};

template <typename DataT>
template <typename Spline1DContainer<DataT>::SafetyLevel SafeT>
GPUdi() int32_t Spline1DContainer<DataT>::getLeftKnotIndexForU(DataT u) const
{
  /// Get i: u is in [knot_i, knot_{i+1}) segment
  /// when u is otside of [0, mUmax], return a corresponding edge segment
  int32_t iu = u < 0 ? 0 : (u > (float)mUmax ? mUmax : (int32_t)u);
  if (SafeT == SafetyLevel::kSafe) {
    iu = (iu < 0) ? 0 : (iu > mUmax ? mUmax : iu);
  }
  return getUtoKnotMap()[iu];
}

template <typename DataT>
GPUdi() void Spline1DContainer<DataT>::setXrange(DataT xMin, DataT xMax)
{
  mXmin = xMin;
  double l = ((double)xMax) - xMin;
  if (l < 1.e-8) {
    l = 1.e-8;
  }
  mXtoUscale = mUmax / l;
}

/// ==================================================================================================
///
/// Spline1DSpec class declares different specializations of the Spline1D class.
///
/// The specializations depend on the value of Spline1D's template parameter YdimT.
/// specializations have different constructors and slightly different declarations of methods.
///
/// The meaning of the template parameters:
///
/// \param DataT data type: float or double
/// \param YdimT
///    YdimT > 0 : the number of Y dimensions is known at the compile time and is equal to YdimT
///    YdimT = 0 : the number of Y dimensions will be set in the runtime
///    YdimT < 0 : the number of Y dimensions will be set in the runtime, and it will not exceed abs(XdimT)
/// \param SpecT specialisation number:
///  0 - a parent class for all other specializations
///  1 - nYdim>0: nYdim is set at the compile time
///  2 - nYdim<0: nYdim must be set during runtime
///  3 - specialization where nYdim==1 (a small add-on on top of the other specs)
///
template <typename DataT, int32_t YdimT, int32_t SpecT>
class Spline1DSpec;

/// ==================================================================================================
/// Specialization 0 declares common methods for all other Spline2D specializations.
/// Implementations of the methods may depend on the YdimT value.
///
template <typename DataT, int32_t YdimT>
class Spline1DSpec<DataT, YdimT, 0> : public Spline1DContainer<DataT>
{
  typedef Spline1DContainer<DataT> TBase;

 public:
  typedef typename TBase::SafetyLevel SafetyLevel;
  typedef typename TBase::Knot Knot;

  /// _______________  Interpolation math   ________________________

  /// Get interpolated value S(x)
  GPUd() void interpolate(DataT x, GPUgeneric() DataT S[/*mYdim*/]) const
  {
    interpolateAtU<SafetyLevel::kSafe>(mYdim, mParameters, convXtoU(x), S);
  }

  /// Get interpolated value for an nYdim-dimensional S(u) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtU(int32_t inpYdim, GPUgeneric() const DataT Parameters[],
                             DataT u, GPUgeneric() DataT S[/*nYdim*/]) const
  {
    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const auto nYdim = nYdimTmp.get();
    int32_t iknot = TBase::template getLeftKnotIndexForU<SafeT>(u);
    const DataT* d = Parameters + (2 * nYdim) * iknot;
    interpolateAtU(nYdim, getKnots()[iknot], &(d[0]), &(d[nYdim]), &(d[2 * nYdim]), &(d[3 * nYdim]), u, S);
  }

  /// The main mathematical utility.
  /// Get interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
  /// using the spline values Sl, Sr and the slopes Dl, Dr
  template <typename T>
  GPUd() void interpolateAtU(int32_t inpYdim, const Knot& knotL,
                             GPUgeneric() const T Sl[/*mYdim*/], GPUgeneric() const T Dl[/*mYdim*/],
                             GPUgeneric() const T Sr[/*mYdim*/], GPUgeneric() const T Dr[/*mYdim*/],
                             DataT u, GPUgeneric() T S[/*mYdim*/]) const
  {
    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const auto nYdim = nYdimTmp.get();

    auto [dSdSl, dSdDl, dSdSr, dSdDr] = getSderivativesOverParsAtU<T>(knotL, u);
    for (int32_t dim = 0; dim < nYdim; ++dim) {
      S[dim] = dSdSr * Sr[dim] + dSdSl * Sl[dim] + dSdDl * Dl[dim] + dSdDr * Dr[dim];
    }

    /*
    another way to calculate f(u):

    if (u < (DataT)0) {
      u = (DataT)0;
    }
    if (u > (DataT)TBase::getUmax()) {
      u = (DataT)TBase::getUmax();
    }

    T uu = T(u - knotL.u);
    T li = T(knotL.Li);
    T v = uu * li; // scaled u
    for (int32_t dim = 0; dim < nYdim; ++dim) {
      T df = (Sr[dim] - Sl[dim]) * li;
      T a = Dl[dim] + Dr[dim] - df - df;
      T b = df - Dl[dim] - a;
      S[dim] = ((a * v + b) * v + Dl[dim]) * uu + Sl[dim];
    }
    */
  }

  template <typename T>
  GPUd() std::array<T, 4> getSderivativesOverParsAtU(const Knot& knotL, DataT u) const
  {
    /// Get derivatives of the interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
    /// over the spline parameters Sl(eft), Sr(ight) and the slopes Dl, Dr

    if (u < (DataT)0) {
      u = (DataT)0;
    }
    if (u > (DataT)TBase::getUmax()) {
      u = (DataT)TBase::getUmax();
    }

    u = u - knotL.u;
    T v = u * T(knotL.Li); // scaled u
    T vm1 = v - T(1.);
    T a = u * vm1;
    T v2 = v * v;
    T dSdSr = v2 * (T(3.) - v - v);
    T dSdSl = T(1.) - dSdSr;
    T dSdDl = vm1 * a;
    T dSdDr = v * a;
    // S(u) = dSdSl * Sl + dSdSr * Sr + dSdDl * Dl + dSdDr * Dr;
    return {dSdSl, dSdDl, dSdSr, dSdDr};
  }

  template <typename T>
  GPUd() std::array<T, 8> getSDderivativesOverParsAtU(const Knot& knotL, DataT u) const
  {
    /// Get derivatives of the interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
    /// over the spline values Sl, Sr and the slopes Dl, Dr

    if (u < (DataT)0) {
      u = (DataT)0;
    }
    if (u > (DataT)TBase::getUmax()) {
      u = (DataT)TBase::getUmax();
    }

    u = u - knotL.u;
    T v = u * T(knotL.Li); // scaled u
    T vm1 = v - T(1.);
    T a = u * vm1;
    T v2 = v * v;
    T dSdSr = v2 * (T(3.) - v - v);
    T dSdSl = T(1.) - dSdSr;
    T dSdDl = vm1 * a;
    T dSdDr = v * a;

    T dv = T(knotL.Li);
    T dDdSr = 6. * v * (T(1.) - v) * dv;
    T dDdSl = -dDdSr;
    T dDdDl = vm1 * (v + v + vm1);
    T dDdDr = v * (v + vm1 + vm1);
    // S(u) = dSdSl * Sl + dSdSr * Sr + dSdDl * Dl + dSdDr * Dr;
    // D(u) = dS(u)/du = dDdSl * Sl + dDdSr * Sr + dDdDl * Dl + dDdDr * Dr;
    return {dSdSl, dSdDl, dSdSr, dSdDr, dDdSl, dDdDl, dDdSr, dDdDr};
  }

  using TBase::convXtoU;
  using TBase::getKnot;
  using TBase::getKnots;
  using TBase::getNumberOfKnots;

 protected:
  using TBase::mParameters;
  using TBase::mYdim;
  using TBase::TBase; // inherit constructors and hide them
  ClassDefNV(Spline1DSpec, 0);
};

/// ==================================================================================================
/// Specialization 1: YdimT>0 where the number of Y dimensions is taken from template parameters
/// at the compile time
///
template <typename DataT, int32_t YdimT>
class Spline1DSpec<DataT, YdimT, 1>
  : public Spline1DSpec<DataT, YdimT, 0>
{
  typedef Spline1DContainer<DataT> TVeryBase;
  typedef Spline1DSpec<DataT, YdimT, 0> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;

#if !defined(GPUCA_GPUCODE)
  /// Default constructor
  Spline1DSpec() : Spline1DSpec(2) {}

  /// Constructor for a regular spline
  Spline1DSpec(int32_t numberOfKnots) : TBase()
  {
    recreate(numberOfKnots);
  }
  /// Constructor for an irregular spline
  Spline1DSpec(int32_t numberOfKnots, const int32_t knotU[])
    : TBase()
  {
    recreate(numberOfKnots, knotU);
  }
  /// Copy constructor
  Spline1DSpec(const Spline1DSpec& v) : TBase()
  {
    TBase::cloneFromObject(v, nullptr);
  }
  /// Constructor for a regular spline
  void recreate(int32_t numberOfKnots) { TBase::recreate(YdimT, numberOfKnots); }

  /// Constructor for an irregular spline
  void recreate(int32_t numberOfKnots, const int32_t knotU[])
  {
    TBase::recreate(YdimT, numberOfKnots, knotU);
  }
#endif

  /// Get number of Y dimensions
  GPUd() constexpr int32_t getYdimensions() const { return YdimT; }

  /// Get minimal required alignment for the spline parameters
  GPUd() constexpr size_t getParameterAlignmentBytes() const
  {
    size_t s = 2 * sizeof(DataT) * YdimT;
    return (s < 16) ? s : 16;
  }

  /// Number of parameters
  GPUd() int32_t getNumberOfParameters() const { return (2 * YdimT) * getNumberOfKnots(); }

  /// Size of the parameter array in bytes
  GPUd() size_t getSizeOfParameters() const { return (sizeof(DataT) * 2 * YdimT) * getNumberOfKnots(); }

  ///  _______  Expert tools: interpolation with given nYdim and external Parameters _______

  /// Get interpolated value for an YdimT-dimensional S(u) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtU(GPUgeneric() const DataT Parameters[],
                             DataT u, GPUgeneric() DataT S[/*nYdim*/]) const
  {
    TBase::template interpolateAtU<SafeT>(YdimT, Parameters, u, S);
  }

  /// Get interpolated value for an YdimT-dimensional S(u) at the segment [knotL, next knotR]
  /// using the spline values Sl, Sr and the slopes Dl, Dr
  template <typename T>
  GPUd() void interpolateAtU(const typename TBase::Knot& knotL,
                             GPUgeneric() const T Sl[/*mYdim*/], GPUgeneric() const T Dl[/*mYdim*/],
                             GPUgeneric() const T Sr[/*mYdim*/], GPUgeneric() const T Dr[/*mYdim*/],
                             DataT u, GPUgeneric() T S[/*mYdim*/]) const
  {
    TBase::interpolateAtU(YdimT, knotL, Sl, Dl, Sr, Dr, u, S);
  }

  using TBase::getNumberOfKnots;

  /// _______________  Suppress some parent class methods   ________________________
 private:
#if !defined(GPUCA_GPUCODE)
  using TBase::recreate;
#endif
  using TBase::interpolateAtU;
};

/// ==================================================================================================
/// Specialization 2 (YdimT<=0) where the numbaer of Y dimensions
/// must be set in the runtime via a constructor parameter
///
template <typename DataT, int32_t YdimT>
class Spline1DSpec<DataT, YdimT, 2>
  : public Spline1DSpec<DataT, YdimT, 0>
{
  typedef Spline1DContainer<DataT> TVeryBase;
  typedef Spline1DSpec<DataT, YdimT, 0> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;

#if !defined(GPUCA_GPUCODE)
  /// Default constructor
  Spline1DSpec() : Spline1DSpec(0, 2) {}

  /// Constructor for a regular spline
  Spline1DSpec(int32_t nYdim, int32_t numberOfKnots) : TBase()
  {
    TBase::recreate(nYdim, numberOfKnots);
  }
  /// Constructor for an irregular spline
  Spline1DSpec(int32_t nYdim, int32_t numberOfKnots, const int32_t knotU[]) : TBase()
  {
    TBase::recreate(nYdim, numberOfKnots, knotU);
  }
  /// Copy constructor
  Spline1DSpec(const Spline1DSpec& v) : TBase()
  {
    TVeryBase::cloneFromObject(v, nullptr);
  }
  /// Constructor for a regular spline
  void recreate(int32_t nYdim, int32_t numberOfKnots) { TBase::recreate(nYdim, numberOfKnots); }

  /// Constructor for an irregular spline
  void recreate(int32_t nYdim, int32_t numberOfKnots, const int32_t knotU[])
  {
    TBase::recreate(nYdim, numberOfKnots, knotU);
  }
#endif

  ///  _______  Expert tools: interpolation with given nYdim and external Parameters _______

  using TBase::interpolateAtU;
  ClassDefNV(Spline1DSpec, 0);
};

/// ==================================================================================================
/// Specialization 3, where the number of Y dimensions is 1.
///
template <typename DataT>
class Spline1DSpec<DataT, 1, 3>
  : public Spline1DSpec<DataT, 1, SplineUtil::getSpec(999)>
{
  typedef Spline1DSpec<DataT, 1, SplineUtil::getSpec(999)> TBase;

 public:
  using TBase::TBase; // inherit constructors

  /// Simplified interface for 1D: return the interpolated value
  GPUd() DataT interpolate(DataT x) const
  {
    DataT S = 0.;
    TBase::interpolate(x, &S);
    return S;
  }
};

} // namespace gpu
} // namespace o2

#endif
