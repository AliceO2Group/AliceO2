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

/// \file  Spline2DSpec.h
/// \brief Definition of Spline2DSpec class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE2DSPEC_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE2DSPEC_H

#include "Spline1D.h"
#include "FlatObject.h"
#include "GPUCommonDef.h"
#include "SplineUtil.h"

#if !defined(__CINT__) && !defined(__ROOTCINT__) && !defined(__ROOTCLING__) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_VC) && defined(__cplusplus) && __cplusplus >= 201703L
#include <Vc/Vc>
#include <Vc/SimdArray>
#endif

class TFile;

namespace GPUCA_NAMESPACE
{
namespace gpu
{

/// ==================================================================================================
/// The class Spline2DContainer is a base Spline2D class.
/// It contains all the class members and those methods which only depends on the DataT data type.
/// It also contains all non-inlined methods with the implementation in SplineSpec.cxx file.
///
/// DataT is a data type, which is supposed to be either double or float.
/// For other possible data types one has to add the corresponding instantiation line
/// at the end of the Spline2DSpec.cxx file
///
template <typename DataT>
class Spline2DContainer : public FlatObject
{
 public:
  typedef typename Spline1D<DataT>::SafetyLevel SafetyLevel;
  typedef typename Spline1D<DataT>::Knot Knot;

  /// _____________  Version control __________________________

  /// Version control
  GPUd() static constexpr int getVersion() { return (1 << 16) + Spline1D<DataT>::getVersion(); }

  /// _____________  C++ constructors / destructors __________________________

  /// Default constructor
  Spline2DContainer() CON_DEFAULT;

  /// Disable all other constructors
  Spline2DContainer(const Spline2DContainer&) CON_DELETE;

  /// Destructor
  ~Spline2DContainer() CON_DEFAULT;

  /// _______________  Construction interface  ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// approximate a function F with this spline
  void approximateFunction(double x1Min, double x1Max, double x2Min, double x2Max,
                           std::function<void(double x1, double x2, double f[/*mYdim*/])> F,
                           int nAuxiliaryDataPointsU1 = 4, int nAuxiliaryDataPointsU2 = 4);

  void approximateFunctionViaDataPoints(double x1Min, double x1Max, double x2Min, double x2Max,
                                        std::function<void(double x1, double x2, double f[])> F,
                                        int nAuxiliaryDataPointsX1, int nAuxiliaryDataPointsX2);
#endif

  /// _______________  IO   ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// write a class object to the file
  int writeToFile(TFile& outf, const char* name);

  /// read a class object from the file
  static Spline2DContainer* readFromFile(TFile& inpf, const char* name);
#endif

  /// _______________  Getters   ________________________

  /// Get number of Y dimensions
  GPUd() int getYdimensions() const { return mYdim; }

  /// Get minimal required alignment for the spline parameters
  GPUd() static constexpr size_t getParameterAlignmentBytes() { return 16; }

  /// Number of parameters
  GPUd() int getNumberOfParameters() const { return this->calcNumberOfParameters(mYdim); }

  /// Size of the parameter array in bytes
  GPUd() size_t getSizeOfParameters() const { return sizeof(DataT) * this->getNumberOfParameters(); }

  /// Get a number of knots
  GPUd() int getNumberOfKnots() const { return mGridX1.getNumberOfKnots() * mGridX2.getNumberOfKnots(); }

  /// Get 1-D grid for the X1 coordinate
  GPUd() const Spline1D<DataT>& getGridX1() const { return mGridX1; }

  /// Get 1-D grid for the X2 coordinate
  GPUd() const Spline1D<DataT>& getGridX2() const { return mGridX2; }

  /// Get 1-D grid for X1 or X2 coordinate
  GPUd() const Spline1D<DataT>& getGrid(int ix) const { return (ix == 0) ? mGridX1 : mGridX2; }

  /// Get (u1,u2) of i-th knot
  GPUd() void getKnotU(int iKnot, int& u1, int& u2) const
  {
    u1 = mGridX1.getKnot(iKnot % mGridX1.getNumberOfKnots()).getU();
    u2 = mGridX2.getKnot(iKnot / mGridX1.getNumberOfKnots()).getU();
  }

  /// Get index of a knot (iKnotX1,iKnotX2)
  GPUd() int getKnotIndex(int iKnotX1, int iKnotX2) const
  {
    return mGridX1.getNumberOfKnots() * iKnotX2 + iKnotX1;
  }

  /// Get spline parameters
  GPUd() DataT* getParameters() { return mParameters; }

  /// Get spline parameters const
  GPUd() const DataT* getParameters() const { return mParameters; }

  /// _______________  Technical stuff  ________________________

  /// Get offset of GridX1 flat data in the flat buffer
  GPUd() size_t getGridX1Offset() const { return mGridX1.getFlatBufferPtr() - mFlatBufferPtr; }

  /// Get offset of GridX2 flat data in the flat buffer
  GPUd() size_t getGridX2Offset() const { return mGridX2.getFlatBufferPtr() - mFlatBufferPtr; }

  /// Set X range
  GPUd() void setXrange(DataT x1Min, DataT x1Max, DataT x2Min, DataT x2Max)
  {
    mGridX1.setXrange(x1Min, x1Max);
    mGridX2.setXrange(x2Min, x2Max);
  }

  /// Print method
  void print() const;

  ///  _______________  Expert tools  _______________

  /// Number of parameters for given Y dimensions
  GPUd() int calcNumberOfParameters(int nYdim) const { return (4 * nYdim) * getNumberOfKnots(); }

  ///_______________  Test tools  _______________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB) // code invisible on GPU and in the standalone compilation
  /// Test the class functionality
  static int test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

#if !defined(GPUCA_GPUCODE)
  void cloneFromObject(const Spline2DContainer& obj, char* newFlatBufferPtr);
  void moveBufferTo(char* newBufferPtr);
#endif

  using FlatObject::releaseInternalBuffer;

  void destroy();
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

 protected:
#if !defined(GPUCA_GPUCODE)
  /// Constructor for a regular spline
  void recreate(int nYdim, int nKnotsX1, int nKnotsX2);

  /// Constructor for an irregular spline
  void recreate(int nYdim, int nKnotsX1, const int knotU1[], int nKnotsX2, const int knotU2[]);
#endif

  /// _____________  Data members  ____________

  int mYdim = 0;                ///< dimentionality of F
  Spline1D<DataT> mGridX1;      ///< grid for U axis
  Spline1D<DataT> mGridX2;      ///< grid for V axis
  DataT* mParameters = nullptr; //! (transient!!) F-dependent parameters of the spline

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(Spline2DContainer, 1);
#endif
};

/// ==================================================================================================
///
/// Spline2DSpec class declares different specializations of the Spline2D class.
/// They are the same as the Spline1D specializations. (See Spline1DSpec.h)
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
template <typename DataT, int YdimT, int SpecT>
class Spline2DSpec;

/// ==================================================================================================
/// Specialization 0 declares common methods for all other Spline2D specializations.
/// Implementations of the methods may depend on the YdimT value.
///
template <typename DataT, int YdimT>
class Spline2DSpec<DataT, YdimT, 0>
  : public Spline2DContainer<DataT>
{
  typedef Spline2DContainer<DataT> TBase;

 public:
  typedef typename TBase::SafetyLevel SafetyLevel;
  typedef typename TBase::Knot Knot;

  /// _______________  Interpolation math   ________________________

  /// Get interpolated value S(x)
  GPUd() void interpolate(DataT x1, DataT x2, GPUgeneric() DataT S[/*mYdim*/]) const
  {
    interpolateU<SafetyLevel::kSafe>(mYdim, mParameters, mGridX1.convXtoU(x1), mGridX2.convXtoU(x2), S);
  }

  /// Get interpolated value for an inpYdim-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateUold(int inpYdim, GPUgeneric() const DataT Parameters[],
                              DataT u1, DataT u2, GPUgeneric() DataT S[/*inpYdim*/]) const
  {

    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const int nYdim = nYdimTmp.get();

    const auto maxYdim = SplineUtil::getMaxNdim<YdimT>(inpYdim);
    const int maxYdim4 = 4 * maxYdim.get();

    const auto nYdim2 = nYdim * 2;
    const auto nYdim4 = nYdim * 4;

    const DataT& u = u1;
    const DataT& v = u2;
    int nu = mGridX1.getNumberOfKnots();
    int iu = mGridX1.template getLeftKnotIndexForU<SafeT>(u);
    int iv = mGridX2.template getLeftKnotIndexForU<SafeT>(v);

    const typename TBase::Knot& knotU = mGridX1.template getKnot<SafetyLevel::kNotSafe>(iu);
    const typename TBase::Knot& knotV = mGridX2.template getKnot<SafetyLevel::kNotSafe>(iv);

    const DataT* par00 = Parameters + (nu * iv + iu) * nYdim4; // values { {Y1,Y2,Y3}, {Y1,Y2,Y3}'v, {Y1,Y2,Y3}'u, {Y1,Y2,Y3}''vu } at {u0, v0}
    const DataT* par10 = par00 + nYdim4;                       // values { ... } at {u1, v0}
    const DataT* par01 = par00 + nYdim4 * nu;                  // values { ... } at {u0, v1}
    const DataT* par11 = par01 + nYdim4;                       // values { ... } at {u1, v1}

    DataT Su0[maxYdim4]; // values { {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v0), {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v1) }, at u0
    DataT Du0[maxYdim4]; // derivatives {}'_u  at u0
    DataT Su1[maxYdim4]; // values { {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v0), {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v1) }, at u1
    DataT Du1[maxYdim4]; // derivatives {}'_u  at u1

    for (int i = 0; i < nYdim2; i++) {
      Su0[i] = par00[i];
      Su0[nYdim2 + i] = par01[i];

      Du0[i] = par00[nYdim2 + i];
      Du0[nYdim2 + i] = par01[nYdim2 + i];

      Su1[i] = par10[i];
      Su1[nYdim2 + i] = par11[i];

      Du1[i] = par10[nYdim2 + i];
      Du1[nYdim2 + i] = par11[nYdim2 + i];
    }

    DataT parU[maxYdim4]; // interpolated values { {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v0), {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v1) } at u

    typedef Spline1DSpec<DataT, 4 * YdimT, 0> TGridX1;
    const TGridX1& gridX1 = reinterpret_cast<const TGridX1&>(mGridX1);

    gridX1.interpolateU(nYdim4, knotU, Su0, Du0, Su1, Du1, u, parU);

    const DataT* Sv0 = parU + 0;
    const DataT* Dv0 = parU + nYdim;
    const DataT* Sv1 = parU + nYdim2;
    const DataT* Dv1 = parU + nYdim2 + nYdim;

    typedef Spline1DSpec<DataT, YdimT, 0> TGridX2;
    const TGridX2& gridX2 = reinterpret_cast<const TGridX2&>(mGridX2);
    gridX2.interpolateU(nYdim, knotV, Sv0, Dv0, Sv1, Dv1, v, S);
  }

  /// Get interpolated value for an inpYdim-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateU(int inpYdim, GPUgeneric() const DataT Parameters[],
                           DataT u1, DataT u2, GPUgeneric() DataT S[/*inpYdim*/]) const
  {

    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const int nYdim = nYdimTmp.get();

    // const auto maxYdim = SplineUtil::getMaxNdim<YdimT>(inpYdim);
    // const int maxYdim4 = 4 * maxYdim.get();

    // const auto nYdim2 = nYdim * 2;
    const auto nYdim4 = nYdim * 4;

    const DataT& u = u1;
    const DataT& v = u2;
    int nu = mGridX1.getNumberOfKnots();
    int iu = mGridX1.template getLeftKnotIndexForU<SafeT>(u);
    int iv = mGridX2.template getLeftKnotIndexForU<SafeT>(v);

    const typename TBase::Knot& knotU = mGridX1.template getKnot<SafetyLevel::kNotSafe>(iu);
    const typename TBase::Knot& knotV = mGridX2.template getKnot<SafetyLevel::kNotSafe>(iv);

    const DataT* A = Parameters + (nu * iv + iu) * nYdim4; // values { {Y1,Y2,Y3}, {Y1,Y2,Y3}'v, {Y1,Y2,Y3}'u, {Y1,Y2,Y3}''vu } at {u0, v0}
    const DataT* B = A + nYdim4 * nu;                      // values { ... } at {u0, v1}

    DataT dSl, dDl, dSr, dDr;
    mGridX1.getUderivatives(knotU, u, dSl, dDl, dSr, dDr);
    DataT dSd, dDd, dSu, dDu;
    mGridX2.getUderivatives(knotV, v, dSd, dDd, dSu, dDu);

    // when nYdim == 1:
    // S = dSl * (dSd * A[0] + dDd * A[1]) + dDl * (dSd * A[2] + dDd * A[3]) +
    //     dSr * (dSd * A[4] + dDd * A[5]) + dDr * (dSd * A[6] + dDd * A[7]) +
    //     dSl * (dSu * B[0] + dDu * B[1]) + dDl * (dSu * B[2] + dDu * B[3]) +
    //     dSr * (dSu * B[4] + dDu * B[5]) + dDr * (dSu * B[6] + dDu * B[7]);

    DataT a[8] = {dSl * dSd, dSl * dDd, dDl * dSd, dDl * dDd,
                  dSr * dSd, dSr * dDd, dDr * dSd, dDr * dDd};
    DataT b[8] = {dSl * dSu, dSl * dDu, dDl * dSu, dDl * dDu,
                  dSr * dSu, dSr * dDu, dDr * dSu, dDr * dDu};

    // S = sum a[i]*A[i] + b[i]*B[i]

    for (int dim = 0; dim < nYdim; dim++) {
      S[dim] = 0;
      for (int i = 0; i < 8; i++) {
        S[dim] += a[i] * A[nYdim * i + dim] + b[i] * B[nYdim * i + dim];
      }
    }
  }

 protected:
  using TBase::mGridX1;
  using TBase::mGridX2;
  using TBase::mParameters;
  using TBase::mYdim;
  using TBase::TBase; // inherit constructors and hide them
};

/// ==================================================================================================
/// Specialization 1: YdimT>0 where the number of Y dimensions is taken from template parameters
/// at the compile time
///
template <typename DataT, int YdimT>
class Spline2DSpec<DataT, YdimT, 1>
  : public Spline2DSpec<DataT, YdimT, 0>
{
  typedef Spline2DContainer<DataT> TVeryBase;
  typedef Spline2DSpec<DataT, YdimT, 0> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;

#if !defined(GPUCA_GPUCODE)
  /// Default constructor
  Spline2DSpec() : Spline2DSpec(2, 2) {}

  /// Constructor for a regular spline
  Spline2DSpec(int nKnotsX1, int nKnotsX2) : TBase()
  {
    recreate(nKnotsX1, nKnotsX2);
  }
  /// Constructor for an irregular spline
  Spline2DSpec(int nKnotsX1, const int knotU1[],
               int nKnotsX2, const int knotU2[])
    : TBase()
  {
    recreate(nKnotsX1, knotU1, nKnotsX2, knotU2);
  }
  /// Copy constructor
  Spline2DSpec(const Spline2DSpec& v) : TBase()
  {
    TBase::cloneFromObject(v, nullptr);
  }
  /// Constructor for a regular spline
  void recreate(int nKnotsX1, int nKnotsX2)
  {
    TBase::recreate(YdimT, nKnotsX1, nKnotsX2);
  }

  /// Constructor for an irregular spline
  void recreate(int nKnotsX1, const int knotU1[],
                int nKnotsX2, const int knotU2[])
  {
    TBase::recreate(YdimT, nKnotsX1, knotU1, nKnotsX2, knotU2);
  }
#endif

  /// Get number of Y dimensions
  GPUd() constexpr int getYdimensions() const { return YdimT; }

  /// Number of parameters
  GPUd() int getNumberOfParameters() const { return (4 * YdimT) * getNumberOfKnots(); }

  /// Size of the parameter array in bytes
  GPUd() size_t getSizeOfParameters() const { return (sizeof(DataT) * 4 * YdimT) * getNumberOfKnots(); }

  ///  _______  Expert tools: interpolation with given nYdim and external Parameters _______

  /// Get interpolated value for an YdimT-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateU(GPUgeneric() const DataT Parameters[],
                           DataT u1, DataT u2, GPUgeneric() DataT S[/*nYdim*/]) const
  {
    TBase::template interpolateU<SafeT>(YdimT, Parameters, u1, u2, S);
  }

  /// Get interpolated value for an YdimT-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateUold(GPUgeneric() const DataT Parameters[],
                              DataT u1, DataT u2, GPUgeneric() DataT S[/*nYdim*/]) const
  {
    TBase::template interpolateUold<SafeT>(YdimT, Parameters, u1, u2, S);
  }

  using TBase::getNumberOfKnots;

  /// _______________  Suppress some parent class methods   ________________________
 private:
#if !defined(GPUCA_GPUCODE)
  using TBase::recreate;
#endif
  using TBase::interpolateU;
};

/// ==================================================================================================
/// Specialization 2 (YdimT<=0) where the numbaer of Y dimensions
/// must be set in the runtime via a constructor parameter
///
template <typename DataT, int YdimT>
class Spline2DSpec<DataT, YdimT, 2>
  : public Spline2DSpec<DataT, YdimT, 0>
{
  typedef Spline2DContainer<DataT> TVeryBase;
  typedef Spline2DSpec<DataT, YdimT, 0> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;

#if !defined(GPUCA_GPUCODE)
  /// Default constructor
  Spline2DSpec() : Spline2DSpec(0, 2, 2) {}

  /// Constructor for a regular spline
  Spline2DSpec(int nYdim, int nKnotsX1, int nKnotsX2) : TBase()
  {
    TBase::recreate(nYdim, nKnotsX1, nKnotsX2);
  }

  /// Constructor for an irregular spline
  Spline2DSpec(int nYdim, int nKnotsX1, const int knotU1[],
               int nKnotsX2, const int knotU2[]) : TBase()
  {
    TBase::recreate(nYdim, nKnotsX1, knotU1, nKnotsX2, knotU2);
  }

  /// Copy constructor
  Spline2DSpec(const Spline2DSpec& v) : TBase()
  {
    cloneFromObject(v, nullptr);
  }

  /// Constructor for a regular spline
  void recreate(int nYdim, int nKnotsX1, int nKnotsX2)
  {
    TBase::recreate(nYdim, nKnotsX1, nKnotsX2);
  }

  /// Constructor for an irregular spline
  void recreate(int nYdim, int nKnotsX1, const int knotU1[],
                int nKnotsX2, const int knotU2[])
  {
    TBase::recreate(nYdim, nKnotsX1, knotU1, nKnotsX2, knotU2);
  }
#endif

  ///  _______  Expert tools: interpolation with given nYdim and external Parameters _______

  using TBase::interpolateU;
};

/// ==================================================================================================
/// Specialization 3, where the number of Y dimensions is 1.
///
template <typename DataT>
class Spline2DSpec<DataT, 1, 3>
  : public Spline2DSpec<DataT, 1, SplineUtil::getSpec(999)>
{
  typedef Spline2DSpec<DataT, 1, SplineUtil::getSpec(999)> TBase;

 public:
  using TBase::TBase; // inherit constructors

  /// Simplified interface for 1D: return the interpolated value
  GPUd() DataT interpolate(DataT x1, DataT x2) const
  {
    DataT S = 0.;
    TBase::interpolate(x1, x2, &S);
    return S;
  }

  // this parent method should be public anyhow,
  // but w/o this extra declaration compiler gets confused
  using TBase::interpolate;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
