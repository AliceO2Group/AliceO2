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
#include <type_traits>
#endif

class TFile;

namespace o2
{
namespace gpu
{

/// The struct Knot represents the i-th knot and the segment [knot_i, knot_i+1]
///
template <typename DataT>
struct Knot {
  DataT u;  ///< u coordinate of the knot i (an integer number in float format)
  DataT Li; ///< inverse length of the [knot_i, knot_{i+1}] segment ( == 1./ a (small) integer )
  /// Get u as an integer
  GPUd() int32_t getU() const { return (int32_t)(u + 0.1f); }
};

/// Named enumeration for the safety level used by some methods
enum SafetyLevel { kNotSafe,
                   kSafe };

/// ==================================================================================================
/// The class Spline1DContainerBase is a base class of Spline1D.
/// It contains all the class members and those methods which only depends on the DataT data type.
/// It also contains all non-inlined methods with the implementation in Spline1DSpec.cxx file.
///
/// DataT is a data type, which is supposed to be either double or float.
/// For other possible data types one has to add the corresponding instantiation line
/// at the end of the Spline1DSpec.cxx file
///
template <typename DataT, class FlatBase = FlatObject>
class Spline1DContainerBase : public FlatBase
{
 public:
  /// _____________  Version control __________________________

  /// Version control
  GPUd() static constexpr int32_t getVersion() { return 1; }

  /// _____________  C++ constructors / destructors __________________________

  /// Default constructor, required by the Root IO
  Spline1DContainerBase() = default;

  /// Disable all other constructors
  Spline1DContainerBase(const Spline1DContainerBase&) = delete;

  /// Destructor
  ~Spline1DContainerBase() = default;

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

  /// _______________  Technical stuff  ________________________

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

  ///  _______________  Expert tools  _______________

  /// Number of parameters for given Y dimensions
  GPUd() int32_t calcNumberOfParameters(int32_t nYdim) const { return (2 * nYdim) * getNumberOfKnots(); }

  /// _____________  Data members  ____________
  int32_t mYdim = 0;          ///< dimentionality of F
  int32_t mNumberOfKnots = 0; ///< n knots on the grid
  int32_t mUmax = 0;          ///< U of the last knot
  DataT mXmin = 0;            ///< X of the first knot
  DataT mXtoUscale = 0;       ///< a scaling factor to convert X to U
};

template <typename DataT, typename FlatBase = FlatObject>
class Spline1DContainer; // forward declaration

template <typename DataT>
class Spline1DContainer<DataT, FlatObject> : public Spline1DContainerBase<DataT, FlatObject>
{
 public:
  /// Get a map (integer U -> corresponding knot index)
  GPUd() const int32_t* getUtoKnotMap() const { return mUtoKnotMap; }

  /// Get the array of knots
  GPUd() const Knot<DataT>* getKnots() const { return reinterpret_cast<const Knot<DataT>*>(this->mFlatBufferPtr); }

  /// Get i-th knot
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() const Knot<DataT>& getKnot(int32_t i) const
  {
    if (SafeT == SafetyLevel::kSafe) {
      i = (i < 0) ? 0 : (i >= this->getNumberOfKnots() ? this->getNumberOfKnots() - 1 : i);
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

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// approximate a function F with this spline
  void approximateFunction(double xMin, double xMax, std::function<void(double x, double f[])> F, int32_t nAuxiliaryDataPoints = 4);

  /// write a class object to the file
  int32_t writeToFile(TFile& outf, const char* name);

  /// read a class object from the file
  static Spline1DContainer* readFromFile(TFile& inpf, const char* name);

  /// Test the class functionality
  static int32_t test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

  /// Print method
  void print() const;

#if !defined(GPUCA_GPUCODE)
  void cloneFromObject(const Spline1DContainer& obj, char* newFlatBufferPtr);
  void moveBufferTo(char* newBufferPtr);

  /// Copy schema fields from a spline with a different FlatBase.
  /// mUtoKnotMap and mParameters are set to nullptr; call setActualBufferAddress() afterward.
  template <class OtherFlatBase>
  void importFrom(const Spline1DContainerBase<DataT, OtherFlatBase>& src);
#endif

  void destroy();
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

 protected:
#if !defined(GPUCA_GPUCODE)
  void recreate(int32_t nYdim, int32_t numberOfKnots);
  void recreate(int32_t nYdim, int32_t numberOfKnots, const int32_t knotU[]);
#endif

  /// Non-const accessor to U->knots map
  int32_t* getUtoKnotMap() { return mUtoKnotMap; }

  /// Non-const accessor to the knots array
  Knot<DataT>* getKnots() { return reinterpret_cast<Knot<DataT>*>(this->mFlatBufferPtr); }

  int32_t* mUtoKnotMap = nullptr; //! (transient!!) pointer to (integer U -> knot index) map inside the mFlatBufferPtr array
  DataT* mParameters = nullptr;   //! (transient!!) pointer to F-dependent parameters inside the mFlatBufferPtr array

  ClassDefNV(Spline1DContainer, 1);
};

template <typename DataT>
class Spline1DContainer<DataT, NoFlatObject> : public Spline1DContainerBase<DataT, NoFlatObject>
{
 public:
  /// Get the U->knot-index map from an explicit flat buffer pointer.
  GPUd() const int32_t* getUtoKnotMapFromBuffer(const char* flatBuf) const
  {
    return reinterpret_cast<const int32_t*>(flatBuf + this->getNumberOfKnots() * sizeof(Knot<DataT>));
  }

  /// Map a U coordinate to its left knot index, using an explicit flat buffer pointer.
  /// Use this instead of getLeftKnotIndexForU() on the zero-copy path.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() int32_t getLeftKnotIndexForUFromBuffer(const char* flatBuf, DataT u) const
  {
    int32_t iu = u < 0 ? 0 : (u > (float)this->mUmax ? this->mUmax : (int32_t)u);
    if (SafeT == SafetyLevel::kSafe) {
      iu = (iu < 0) ? 0 : (iu > this->mUmax ? this->mUmax : iu);
    }
    return getUtoKnotMapFromBuffer(flatBuf)[iu];
  }

  /// Get the knot array from an explicit flat buffer pointer.
  /// Use this instead of getKnots() when the object was copied across process
  /// boundaries and mFlatBufferPtr has not been fixed up (zero-copy path).
  GPUd() const Knot<DataT>* getKnotsFromBuffer(const char* flatBuf) const { return reinterpret_cast<const Knot<DataT>*>(flatBuf); }

  /// Get i-th knot from an explicit flat buffer pointer.
  /// Use this instead of getKnot() on the zero-copy path.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() const Knot<DataT>& getKnotFromBuffer(const char* flatBuf, int32_t i) const
  {
    if (SafeT == SafetyLevel::kSafe) {
      i = (i < 0) ? 0 : (i >= this->getNumberOfKnots() ? this->getNumberOfKnots() - 1 : i);
    }
    return getKnotsFromBuffer(flatBuf)[i];
  }

  // Lifecycle no-ops: NoFlatObject splines have no owned buffer.
  void destroy()
  {
    this->mNumberOfKnots = 0;
    this->mUmax = 0;
    this->mYdim = 0;
    this->mXmin = 0.;
    this->mXtoUscale = 1.;
    this->mFlatBufferSize = 0;
  }
  GPUdi() void setActualBufferAddress(char*) {}
  GPUdi() void setFutureBufferAddress(char*) {}

#if !defined(GPUCA_GPUCODE)
  /// Copy schema fields from a spline with a different FlatBase (no pointer members to copy).
  template <class OtherFlatBase>
  void importFrom(const Spline1DContainerBase<DataT, OtherFlatBase>& src)
  {
    this->mYdim = src.getYdimensions();
    this->mNumberOfKnots = src.getNumberOfKnots();
    this->mUmax = src.getUmax();
    this->mXmin = src.getXmin();
    this->mXtoUscale = src.getXtoUscale();
    this->mFlatBufferSize = src.getFlatBufferSize();
  }
#endif
};

template <typename DataT>
template <SafetyLevel SafeT>
GPUdi() int32_t Spline1DContainer<DataT, FlatObject>::getLeftKnotIndexForU(DataT u) const
{
  /// Get i: u is in [knot_i, knot_{i+1}) segment
  /// when u is otside of [0, mUmax], return a corresponding edge segment
  int32_t iu = u < 0 ? 0 : (u > (float)this->mUmax ? this->mUmax : (int32_t)u);
  if (SafeT == SafetyLevel::kSafe) {
    iu = (iu < 0) ? 0 : (iu > this->mUmax ? this->mUmax : iu);
  }
  return getUtoKnotMap()[iu];
}

template <typename DataT, class FlatBase>
GPUdi() void Spline1DContainerBase<DataT, FlatBase>::setXrange(DataT xMin, DataT xMax)
{
  mXmin = xMin;
  double l = ((double)xMax) - xMin;
  if (l < 1.e-8) {
    l = 1.e-8;
  }
  mXtoUscale = this->mUmax / l;
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
template <typename DataT, int32_t YdimT, int32_t SpecT, class FlatBase = FlatObject>
class Spline1DSpec;

/// ==================================================================================================
/// Specialization 0 declares common methods for all other Spline2D specializations.
/// Implementations of the methods may depend on the YdimT value.
///
template <typename DataT, int32_t YdimT, class FlatBase>
class Spline1DSpec<DataT, YdimT, 0, FlatBase> : public Spline1DContainer<DataT, FlatBase>
{
  using Container = Spline1DContainer<DataT, FlatBase>;

 public:
  using KnotType = Knot<DataT>;

  /// _______________  Interpolation math   ________________________

  /// Get interpolated value S(x)  [FlatObject path only]
  GPUd() void interpolate(DataT x, GPUgeneric() DataT S[/*mYdim*/]) const
  {
    if constexpr (std::is_same_v<FlatBase, FlatObject>) {
      interpolateAtU<SafetyLevel::kSafe>(this->mYdim, this->mParameters, this->convXtoU(x), S);
    }
  }

  /// Get interpolated value for an nYdim-dimensional S(u) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtU(int32_t inpYdim, GPUgeneric() const DataT Parameters[], DataT u, GPUgeneric() DataT S[/*nYdim*/]) const
  {
    if constexpr (std::is_same_v<FlatBase, FlatObject>) {
      const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
      const auto nYdim = nYdimTmp.get();
      int32_t iknot = this->template getLeftKnotIndexForU<SafeT>(u);
      const DataT* d = Parameters + (2 * nYdim) * iknot;
      interpolateAtU(nYdim, this->getKnots()[iknot], &(d[0]), &(d[nYdim]), &(d[2 * nYdim]), &(d[3 * nYdim]), u, S);
    }
  }

  /// The main mathematical utility.
  /// Get interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
  /// using the spline values Sl, Sr and the slopes Dl, Dr
  template <typename T>
  GPUd() void interpolateAtU(int32_t inpYdim, const KnotType& knotL,
                             GPUgeneric() const T Sl[/*mYdim*/], GPUgeneric() const T Dl[/*mYdim*/],
                             GPUgeneric() const T Sr[/*mYdim*/], GPUgeneric() const T Dr[/*mYdim*/],
                             DataT u, GPUgeneric() T S[/*mYdim*/]) const
  {
    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const auto nYdim = nYdimTmp.get();

    T dSdSl, dSdDl, dSdSr, dSdDr;
    getSderivativesOverParsAtU<T>(knotL, u, dSdSl, dSdDl, dSdSr, dSdDr);

    for (int32_t dim = 0; dim < nYdim; ++dim) {
      S[dim] = dSdSr * Sr[dim] + dSdSl * Sl[dim] + dSdDl * Dl[dim] + dSdDr * Dr[dim];
    }

    /*
    another way to calculate f(u):

    if (u < (DataT)0) {
      u = (DataT)0;
    }
    if (u > (DataT)ParentSpec::getUmax()) {
      u = (DataT)ParentSpec::getUmax();
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
  GPUd() void getSderivativesOverParsAtU(const KnotType& knotL, DataT u, T& dSdSl, T& dSdDl, T& dSdSr, T& dSdDr) const
  {
    /// Get derivatives of the interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
    /// over the spline parameters Sl(eft), Sr(ight) and the slopes Dl, Dr

    if (u < (DataT)0) {
      u = (DataT)0;
    }
    if (u > (DataT)Container::getUmax()) {
      u = (DataT)Container::getUmax();
    }

    u = u - knotL.u;
    T v = u * T(knotL.Li); // scaled u
    T vm1 = v - T(1.);
    T a = u * vm1;
    T v2 = v * v;
    dSdSr = v2 * (T(3.) - v - v);
    dSdSl = T(1.) - dSdSr;
    dSdDl = vm1 * a;
    dSdDr = v * a;
    // S(u) = dSdSl * Sl + dSdSr * Sr + dSdDl * Dl + dSdDr * Dr;
  }

  template <typename T>
  GPUd() void getSDderivativesOverParsAtU(const KnotType& knotL, DataT u, T& dSdSl, T& dSdDl, T& dSdSr, T& dSdDr, T& dDdSl, T& dDdDl, T& dDdSr, T& dDdDr) const
  {
    /// Get derivatives of the interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
    /// over the spline values Sl, Sr and the slopes Dl, Dr

    if (u < (DataT)0) {
      u = (DataT)0;
    }
    if (u > (DataT)Container::getUmax()) {
      u = (DataT)Container::getUmax();
    }

    u = u - knotL.u;
    T v = u * T(knotL.Li); // scaled u
    T vm1 = v - T(1.);
    T a = u * vm1;
    T v2 = v * v;
    dSdSr = v2 * (T(3.) - v - v);
    dSdSl = T(1.) - dSdSr;
    dSdDl = vm1 * a;
    dSdDr = v * a;

    T dv = T(knotL.Li);
    dDdSr = T(6.) * v * (T(1.) - v) * dv;
    dDdSl = -dDdSr;
    dDdDl = vm1 * (v + v + vm1);
    dDdDr = v * (v + vm1 + vm1);
    // S(u) = dSdSl * Sl + dSdSr * Sr + dSdDl * Dl + dSdDr * Dr;
    // D(u) = dS(u)/du = dDdSl * Sl + dDdSr * Sr + dDdDl * Dl + dDdDr * Dr;
  }
};

/// ==================================================================================================
/// Specialization 1: YdimT>0 where the number of Y dimensions is taken from template parameters
/// at the compile time
///
template <typename DataT, int32_t YdimT, class FlatBase>
class Spline1DSpec<DataT, YdimT, 1, FlatBase> : public Spline1DSpec<DataT, YdimT, 0, FlatBase>
{
  using ParentSpec = Spline1DSpec<DataT, YdimT, 0, FlatBase>;

 public:
#if !defined(GPUCA_GPUCODE)
  /// Default constructor — skips recreate for NoFlatObject (no owned buffer)
  Spline1DSpec() : ParentSpec()
  {
    if constexpr (!std::is_same_v<FlatBase, NoFlatObject>) {
      recreate(2);
    }
  }

  /// Constructor for a regular spline
  Spline1DSpec(int32_t numberOfKnots) : ParentSpec() { recreate(numberOfKnots); }

  /// Constructor for an irregular spline
  Spline1DSpec(int32_t numberOfKnots, const int32_t knotU[]) : ParentSpec() { recreate(numberOfKnots, knotU); }

  /// Copy constructor
  Spline1DSpec(const Spline1DSpec& v) : ParentSpec() { ParentSpec::cloneFromObject(v, nullptr); }

  /// Constructor for a regular spline
  void recreate(int32_t numberOfKnots) { ParentSpec::recreate(YdimT, numberOfKnots); }

  /// Constructor for an irregular spline
  void recreate(int32_t numberOfKnots, const int32_t knotU[]) { ParentSpec::recreate(YdimT, numberOfKnots, knotU); }
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
  GPUd() int32_t getNumberOfParameters() const { return (2 * YdimT) * this->getNumberOfKnots(); }

  /// Size of the parameter array in bytes
  GPUd() size_t getSizeOfParameters() const { return (sizeof(DataT) * 2 * YdimT) * this->getNumberOfKnots(); }

  ///  _______  Expert tools: interpolation with given nYdim and external Parameters _______

  /// Get interpolated value for an YdimT-dimensional S(u) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtU(GPUgeneric() const DataT Parameters[], DataT u, GPUgeneric() DataT S[/*nYdim*/]) const
  {
    ParentSpec::template interpolateAtU<SafeT>(YdimT, Parameters, u, S);
  }

  /// Get interpolated value for an YdimT-dimensional S(u) at the segment [knotL, next knotR]
  /// using the spline values Sl, Sr and the slopes Dl, Dr
  template <typename T>
  GPUd() void interpolateAtU(const typename ParentSpec::KnotType& knotL,
                             GPUgeneric() const T Sl[/*mYdim*/], GPUgeneric() const T Dl[/*mYdim*/],
                             GPUgeneric() const T Sr[/*mYdim*/], GPUgeneric() const T Dr[/*mYdim*/],
                             DataT u, GPUgeneric() T S[/*mYdim*/]) const
  {
    ParentSpec::interpolateAtU(YdimT, knotL, Sl, Dl, Sr, Dr, u, S);
  }
};

/// ==================================================================================================
/// Specialization 2 (YdimT<=0) where the numbaer of Y dimensions
/// must be set in the runtime via a constructor parameter
///
template <typename DataT, int32_t YdimT, class FlatBase>
class Spline1DSpec<DataT, YdimT, 2, FlatBase> : public Spline1DSpec<DataT, YdimT, 0, FlatBase>
{
  using ParentSpec = Spline1DSpec<DataT, YdimT, 0, FlatBase>;
  using Container = Spline1DContainer<DataT, FlatBase>;

 public:
#if !defined(GPUCA_GPUCODE)
  /// Default constructor — skips recreate for NoFlatObject (no owned buffer)
  Spline1DSpec() : ParentSpec()
  {
    if constexpr (!std::is_same_v<FlatBase, NoFlatObject>) {
      ParentSpec::recreate(0, 2);
    }
  }

  /// Constructor for a regular spline
  Spline1DSpec(int32_t nYdim, int32_t numberOfKnots) : ParentSpec()
  {
    ParentSpec::recreate(nYdim, numberOfKnots);
  }

  /// Constructor for an irregular spline
  Spline1DSpec(int32_t nYdim, int32_t numberOfKnots, const int32_t knotU[]) : ParentSpec()
  {
    ParentSpec::recreate(nYdim, numberOfKnots, knotU);
  }

  /// Copy constructor
  Spline1DSpec(const Spline1DSpec& v) : ParentSpec()
  {
    Container::cloneFromObject(v, nullptr);
  }

  /// Constructor for a regular spline
  void recreate(int32_t nYdim, int32_t numberOfKnots) { ParentSpec::recreate(nYdim, numberOfKnots); }

  /// Constructor for an irregular spline
  void recreate(int32_t nYdim, int32_t numberOfKnots, const int32_t knotU[]) { ParentSpec::recreate(nYdim, numberOfKnots, knotU); }
#endif
};

/// ==================================================================================================
/// Specialization 3, where the number of Y dimensions is 1.
///
template <typename DataT, class FlatBase>
class Spline1DSpec<DataT, 1, 3, FlatBase> : public Spline1DSpec<DataT, 1, SplineUtil::getSpec(999), FlatBase>
{
  using ParentSpec = Spline1DSpec<DataT, 1, SplineUtil::getSpec(999), FlatBase>;

 public:
  /// Simplified interface for 1D: return the interpolated value
  GPUd() DataT interpolate(DataT x) const
  {
    DataT S = 0;
    ParentSpec::interpolate(x, &S);
    return S;
  }
};

} // namespace gpu
} // namespace o2

#endif
