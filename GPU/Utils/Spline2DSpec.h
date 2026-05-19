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
#if !defined(GPUCA_GPUCODE)
#include <type_traits>
#endif

#if !defined(__CLING__) && !defined(G__ROOT) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_VC)
#include <Vc/Vc>
#include <Vc/SimdArray>
#endif

class TFile;

namespace o2::gpu
{

/// ==================================================================================================
/// The class Spline2DContainerBase is a base Spline2D class.
/// It contains all the class members and those methods which only depends on the DataT data type.
/// It also contains all non-inlined methods with the implementation in SplineSpec.cxx file.
///
/// DataT is a data type, which is supposed to be either double or float.
/// For other possible data types one has to add the corresponding instantiation line
/// at the end of the Spline2DSpec.cxx file
///
template <typename DataT, class FlatBase = FlatObject>
class Spline2DContainerBase : public FlatBase
{
 public:
  /// _____________  Version control __________________________

  /// Version control
  GPUd() static constexpr int32_t getVersion() { return (1 << 16) + Spline1D<DataT>::getVersion(); }

  /// _____________  C++ constructors / destructors __________________________

  /// Default constructor
  Spline2DContainerBase() = default;

  /// Disable all other constructors
  Spline2DContainerBase(const Spline2DContainerBase&) = delete;

  /// Destructor
  ~Spline2DContainerBase() = default;

  /// _______________  Construction interface  ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// approximate a function F with this spline
  void approximateFunction(double x1Min, double x1Max, double x2Min, double x2Max,
                           std::function<void(double x1, double x2, double f[/*mYdim*/])> F,
                           int32_t nAuxiliaryDataPointsU1 = 4, int32_t nAuxiliaryDataPointsU2 = 4);

  void approximateFunctionViaDataPoints(double x1Min, double x1Max, double x2Min, double x2Max,
                                        std::function<void(double x1, double x2, double f[])> F,
                                        int32_t nAuxiliaryDataPointsX1, int32_t nAuxiliaryDataPointsX2);
#endif

  /// _______________  IO   ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// write a class object to the file
  int32_t writeToFile(TFile& outf, const char* name);

  /// read a class object from the file
  static Spline2DContainerBase* readFromFile(TFile& inpf, const char* name);
#endif

  /// _______________  Getters   ________________________

  /// Get number of Y dimensions
  GPUd() int32_t getYdimensions() const { return mYdim; }

  /// Get minimal required alignment for the spline parameters
  GPUd() static constexpr size_t getParameterAlignmentBytes() { return 16; }

  /// Number of parameters
  GPUd() int32_t getNumberOfParameters() const { return this->calcNumberOfParameters(mYdim); }

  /// Size of the parameter array in bytes
  GPUd() size_t getSizeOfParameters() const { return sizeof(DataT) * this->getNumberOfParameters(); }

  /// Get a number of knots
  GPUd() int32_t getNumberOfKnots() const { return mGridX1.getNumberOfKnots() * mGridX2.getNumberOfKnots(); }

  /// Get 1-D grid for the X1 coordinate
  GPUd() const Spline1D<DataT, 0, FlatBase>& getGridX1() const { return mGridX1; }

  /// Get 1-D grid for the X2 coordinate
  GPUd() const Spline1D<DataT, 0, FlatBase>& getGridX2() const { return mGridX2; }

  /// Get 1-D grid for X1 or X2 coordinate
  GPUd() const Spline1D<DataT, 0, FlatBase>& getGrid(int32_t ix) const { return (ix == 0) ? mGridX1 : mGridX2; }

  /// Get (u1,u2) of i-th knot
  GPUd() void getKnotU(int32_t iKnot, int32_t& u1, int32_t& u2) const
  {
    if constexpr (!std::is_same_v<FlatBase, NoFlatObject>) {
      u1 = mGridX1.getKnot(iKnot % mGridX1.getNumberOfKnots()).getU();
      u2 = mGridX2.getKnot(iKnot / mGridX1.getNumberOfKnots()).getU();
    }
  }

  /// Get index of a knot (iKnotX1,iKnotX2)
  GPUd() int32_t getKnotIndex(int32_t iKnotX1, int32_t iKnotX2) const
  {
    return mGridX1.getNumberOfKnots() * iKnotX2 + iKnotX1;
  }

  /// Get spline parameters
  GPUd() DataT* getParameters() { return mParameters; }

  /// Get spline parameters const
  GPUd() const DataT* getParameters() const { return mParameters; }

  /// _______________  Technical stuff  ________________________

  /// Get offset of GridX1 flat data in the flat buffer (only valid for FlatObject-based splines)
  GPUd() size_t getGridX1Offset() const
  {
    if constexpr (!std::is_same_v<FlatBase, NoFlatObject>) {
      return mGridX1.getFlatBufferPtr() - this->mFlatBufferPtr;
    }
    return 0;
  }

  /// Get offset of GridX2 flat data in the flat buffer (only valid for FlatObject-based splines)
  GPUd() size_t getGridX2Offset() const
  {
    if constexpr (!std::is_same_v<FlatBase, NoFlatObject>) {
      return mGridX2.getFlatBufferPtr() - this->mFlatBufferPtr;
    }
    return 0;
  }

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
  GPUd() int32_t calcNumberOfParameters(int32_t nYdim) const { return (4 * nYdim) * getNumberOfKnots(); }

  ///_______________  Test tools  _______________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
  /// Test the class functionality
  static int32_t test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________
#if !defined(GPUCA_GPUCODE)
  void cloneFromObject(const Spline2DContainerBase& obj, char* newFlatBufferPtr);
  void moveBufferTo(char* newBufferPtr);

  /// Copy schema fields (ydim, grid dimensions) from a spline with a different FlatBase.
  /// Used by TPCFastTransformPOD::create() to populate NoFlatObject-based splines.
  template <class OtherFlatBase>
  void importFrom(const Spline2DContainerBase<DataT, OtherFlatBase>& src);
#endif

  void destroy();
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

 protected:
#if !defined(GPUCA_GPUCODE)
  /// Constructor for a regular spline
  void recreate(int32_t nYdim, int32_t nKnotsX1, int32_t nKnotsX2);

  /// Constructor for an irregular spline
  void recreate(int32_t nYdim, int32_t nKnotsX1, const int32_t knotU1[], int32_t nKnotsX2, const int32_t knotU2[]);
#endif

  /// _____________  Data members  ____________

  int32_t mYdim = 0;                    ///< dimentionality of F
  Spline1D<DataT, 0, FlatBase> mGridX1; ///< grid for U axis
  Spline1D<DataT, 0, FlatBase> mGridX2; ///< grid for V axis
  DataT* mParameters = nullptr;         //! (transient!!) F-dependent parameters of the spline
};

template <typename DataT, typename FlatBase = FlatObject>
class Spline2DContainer; // forward declaration

template <typename DataT>
class Spline2DContainer<DataT, FlatObject> : public Spline2DContainerBase<DataT, FlatObject>
{
 public:
  ClassDefNV(Spline2DContainer, 1);
};

template <typename DataT>
class Spline2DContainer<DataT, NoFlatObject> : public Spline2DContainerBase<DataT, NoFlatObject>
{
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
template <typename DataT, int32_t YdimT, int32_t SpecT, class FlatBase = FlatObject>
class Spline2DSpec;

/// ==================================================================================================
/// Specialization 0 declares common methods for all other Spline2D specializations.
/// Implementations of the methods may depend on the YdimT value.
///
template <typename DataT, int32_t YdimT, class FlatBase>
class Spline2DSpec<DataT, YdimT, 0, FlatBase>
  : public Spline2DContainerBase<DataT, FlatBase>
{
 public:
  /// _______________  Interpolation math   ________________________

  /// Get interpolated value S(x)
  GPUd() void interpolate(DataT x1, DataT x2, GPUgeneric() DataT S[/*mYdim*/]) const
  {
    interpolateAtU<SafetyLevel::kSafe>(this->mYdim, this->mParameters, this->mGridX1.convXtoU(x1), this->mGridX2.convXtoU(x2), S);
  }

  /// Get interpolated value for an inpYdim-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtUold(int32_t inpYdim, GPUgeneric() const DataT Parameters[],
                                DataT u1, DataT u2, GPUgeneric() DataT S[/*inpYdim*/]) const
  {

    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const int32_t nYdim = nYdimTmp.get();

    const auto maxYdim = SplineUtil::getMaxNdim<YdimT>(inpYdim);
    const int32_t maxYdim4 = 4 * maxYdim.get();

    const auto nYdim2 = nYdim * 2;
    const auto nYdim4 = nYdim * 4;

    const DataT& u = u1;
    const DataT& v = u2;
    int32_t nu = this->mGridX1.getNumberOfKnots();
    int32_t iu = this->mGridX1.template getLeftKnotIndexForU<SafeT>(u);
    int32_t iv = this->mGridX2.template getLeftKnotIndexForU<SafeT>(v);

    const auto& knotU = this->mGridX1.template getKnot<SafetyLevel::kNotSafe>(iu);
    const auto& knotV = this->mGridX2.template getKnot<SafetyLevel::kNotSafe>(iv);

    const DataT* par00 = Parameters + (nu * iv + iu) * nYdim4; // values { {Y1,Y2,Y3}, {Y1,Y2,Y3}'v, {Y1,Y2,Y3}'u, {Y1,Y2,Y3}''vu } at {u0, v0}
    const DataT* par10 = par00 + nYdim4;                       // values { ... } at {u1, v0}
    const DataT* par01 = par00 + nYdim4 * nu;                  // values { ... } at {u0, v1}
    const DataT* par11 = par01 + nYdim4;                       // values { ... } at {u1, v1}

    DataT Su0[maxYdim4]; // values { {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v0), {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v1) }, at u0
    DataT Du0[maxYdim4]; // derivatives {}'_u  at u0
    DataT Su1[maxYdim4]; // values { {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v0), {Y1,Y2,Y3,Y1'v,Y2'v,Y3'v}(v1) }, at u1
    DataT Du1[maxYdim4]; // derivatives {}'_u  at u1

    for (int32_t i = 0; i < nYdim2; i++) {
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

    using GridX1Base = Spline1DSpec<DataT, 4 * YdimT, 0>;
    const GridX1Base& gridX1 = reinterpret_cast<const GridX1Base&>(this->mGridX1);

    gridX1.interpolateAtU(nYdim4, knotU, Su0, Du0, Su1, Du1, u, parU);

    const DataT* Sv0 = parU + 0;
    const DataT* Dv0 = parU + nYdim;
    const DataT* Sv1 = parU + nYdim2;
    const DataT* Dv1 = parU + nYdim2 + nYdim;

    using GridX2Base = Spline1DSpec<DataT, YdimT, 0>;
    const GridX2Base& gridX2 = reinterpret_cast<const GridX2Base&>(this->mGridX2);
    gridX2.interpolateAtU(nYdim, knotV, Sv0, Dv0, Sv1, Dv1, v, S);
  }

  /// Get interpolated value for an inpYdim-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtU(int32_t inpYdim, GPUgeneric() const DataT Parameters[],
                             DataT u1, DataT u2, GPUgeneric() DataT S[/*inpYdim*/]) const
  {
    if constexpr (!std::is_same_v<FlatBase, FlatObject>) {
      return;
    }

    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const int32_t nYdim = nYdimTmp.get();

    // const auto maxYdim = SplineUtil::getMaxNdim<YdimT>(inpYdim);
    // const int32_t maxYdim4 = 4 * maxYdim.get();

    // const auto nYdim2 = nYdim * 2;
    const auto nYdim4 = nYdim * 4;

    const DataT& u = u1;
    const DataT& v = u2;
    int32_t nu = this->mGridX1.getNumberOfKnots();
    int32_t iu = this->mGridX1.template getLeftKnotIndexForU<SafeT>(u);
    int32_t iv = this->mGridX2.template getLeftKnotIndexForU<SafeT>(v);

    const auto& knotU = this->mGridX1.template getKnot<SafetyLevel::kNotSafe>(iu);
    const auto& knotV = this->mGridX2.template getKnot<SafetyLevel::kNotSafe>(iv);

    const DataT* A = Parameters + (nu * iv + iu) * nYdim4; // values { {Y1,Y2,Y3}, {Y1,Y2,Y3}'v, {Y1,Y2,Y3}'u, {Y1,Y2,Y3}''vu } at {u0, v0}
    const DataT* B = A + nYdim4 * nu;                      // values { ... } at {u0, v1}

    DataT dSl, dDl, dSr, dDr, dSd, dDd, dSu, dDu;
    this->mGridX1.template getSderivativesOverParsAtU<DataT>(knotU, u, dSl, dDl, dSr, dDr);
    this->mGridX2.template getSderivativesOverParsAtU<DataT>(knotV, v, dSd, dDd, dSu, dDu);

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

    for (int32_t dim = 0; dim < nYdim; dim++) {
      S[dim] = 0;
      for (int32_t i = 0; i < 8; i++) {
        S[dim] += a[i] * A[nYdim * i + dim] + b[i] * B[nYdim * i + dim];
      }
    }
  }

  /// Get interpolated parameters (like parameters stored at knots) for an inpYdim-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateParametersAtU(int32_t inpYdim, GPUgeneric() const DataT Parameters[],
                                       DataT u1, DataT u2, GPUgeneric() DataT P[/* 4*inpYdim */]) const
  {

    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const int32_t nYdim = nYdimTmp.get();

    // const auto maxYdim = SplineUtil::getMaxNdim<YdimT>(inpYdim);
    // const int32_t maxYdim4 = 4 * maxYdim.get();

    // const auto nYdim2 = nYdim * 2;
    const auto nYdim4 = nYdim * 4;

    DataT *S = P,
          *Q = P + nYdim,
          *R = P + nYdim * 2,
          *W = P + nYdim * 3;

    const DataT& u = u1;
    const DataT& v = u2;
    int32_t nu = this->mGridX1.getNumberOfKnots();
    int32_t iu = this->mGridX1.template getLeftKnotIndexForU<SafeT>(u);
    int32_t iv = this->mGridX2.template getLeftKnotIndexForU<SafeT>(v);

    const auto& knotU = this->mGridX1.template getKnot<SafetyLevel::kNotSafe>(iu);
    const auto& knotV = this->mGridX2.template getKnot<SafetyLevel::kNotSafe>(iv);

    const DataT* A = Parameters + (nu * iv + iu) * nYdim4; // values { {Y1,Y2,Y3}, {Y1,Y2,Y3}'v, {Y1,Y2,Y3}'u, {Y1,Y2,Y3}''vu } at {u0, v0}
    const DataT* B = A + nYdim4 * nu;                      // values { ... } at {u0, v1}

    DataT dSdSl, dSdDl, dSdSr, dSdDr, dRdSl, dRdDl, dRdSr, dRdDr;
    this->mGridX1.template getSDderivativesOverParsAtU<DataT>(knotU, u, dSdSl, dSdDl, dSdSr, dSdDr, dRdSl, dRdDl, dRdSr, dRdDr);
    DataT dSdSd, dSdDd, dSdSu, dSdDu, dQdSd, dQdDd, dQdSu, dQdDu;
    this->mGridX2.template getSDderivativesOverParsAtU<DataT>(knotV, v, dSdSd, dSdDd, dSdSu, dSdDu, dQdSd, dQdDd, dQdSu, dQdDu);

    // when nYdim == 1:

    // Function value S
    // S = dSdSl * (dSdSd * A[0] + dSdDd * A[1]) + dSdDl * (dSdSd * A[2] + dSdDd * A[3]) +
    //     dSdSr * (dSdSd * A[4] + dSdDd * A[5]) + dSdDr * (dSdSd * A[6] + dSdDd * A[7]) +
    //     dSdSl * (dSdSu * B[0] + dSdDu * B[1]) + dSdDl * (dSdSu * B[2] + dSdDu * B[3]) +
    //     dSdSr * (dSdSu * B[4] + dSdDu * B[5]) + dSdDr * (dSdSu * B[6] + dSdDu * B[7]);

    {
      DataT a[8] = {dSdSl * dSdSd, dSdSl * dSdDd, dSdDl * dSdSd, dSdDl * dSdDd,
                    dSdSr * dSdSd, dSdSr * dSdDd, dSdDr * dSdSd, dSdDr * dSdDd};
      DataT b[8] = {dSdSl * dSdSu, dSdSl * dSdDu, dSdDl * dSdSu, dSdDl * dSdDu,
                    dSdSr * dSdSu, dSdSr * dSdDu, dSdDr * dSdSu, dSdDr * dSdDu};

      // S = sum a[i]*A[i] + b[i]*B[i]

      for (int32_t dim = 0; dim < nYdim; dim++) {
        S[dim] = 0;
        for (int32_t i = 0; i < 8; i++) {
          S[dim] += a[i] * A[nYdim * i + dim] + b[i] * B[nYdim * i + dim];
        }
      }
    }

    // Derivative Q = dS / dv
    // Q = dSdSl * (dQdSd * A[0] + dQdDd * A[1]) + dSdDl * (dQdSd * A[2] + dQdDd * A[3]) +
    //     dSdSr * (dQdSd * A[4] + dQdDd * A[5]) + dSdDr * (dQdSd * A[6] + dQdDd * A[7]) +
    //     dSdSl * (dQdSu * B[0] + dQdDu * B[1]) + dSdDl * (dQdSu * B[2] + dQdDu * B[3]) +
    //     dSdSr * (dQdSu * B[4] + dQdDu * B[5]) + dSdDr * (dQdSu * B[6] + dQdDu * B[7]);

    {
      DataT a[8] = {dSdSl * dQdSd, dSdSl * dQdDd, dSdDl * dQdSd, dSdDl * dQdDd,
                    dSdSr * dQdSd, dSdSr * dQdDd, dSdDr * dQdSd, dSdDr * dQdDd};
      DataT b[8] = {dSdSl * dQdSu, dSdSl * dQdDu, dSdDl * dQdSu, dSdDl * dQdDu,
                    dSdSr * dQdSu, dSdSr * dQdDu, dSdDr * dQdSu, dSdDr * dQdDu};

      // Q = sum a[i]*A[i] + b[i]*B[i]

      for (int32_t dim = 0; dim < nYdim; dim++) {
        Q[dim] = 0;
        for (int32_t i = 0; i < 8; i++) {
          Q[dim] += a[i] * A[nYdim * i + dim] + b[i] * B[nYdim * i + dim];
        }
      }
    }

    // Derivative R = dS / du
    // R = dRdSl * (dSdSd * A[0] + dSdDd * A[1]) + dRdDl * (dSdSd * A[2] + dSdDd * A[3]) +
    //     dRdSr * (dSdSd * A[4] + dSdDd * A[5]) + dRdDr * (dSdSd * A[6] + dSdDd * A[7]) +
    //     dRdSl * (dSdSu * B[0] + dSdDu * B[1]) + dRdDl * (dSdSu * B[2] + dSdDu * B[3]) +
    //     dRdSr * (dSdSu * B[4] + dSdDu * B[5]) + dRdDr * (dSdSu * B[6] + dSdDu * B[7]);

    {
      DataT a[8] = {dRdSl * dSdSd, dRdSl * dSdDd, dRdDl * dSdSd, dRdDl * dSdDd,
                    dRdSr * dSdSd, dRdSr * dSdDd, dRdDr * dSdSd, dRdDr * dSdDd};
      DataT b[8] = {dRdSl * dSdSu, dRdSl * dSdDu, dRdDl * dSdSu, dRdDl * dSdDu,
                    dRdSr * dSdSu, dRdSr * dSdDu, dRdDr * dSdSu, dRdDr * dSdDu};

      // R = sum a[i]*A[i] + b[i]*B[i]

      for (int32_t dim = 0; dim < nYdim; dim++) {
        R[dim] = 0;
        for (int32_t i = 0; i < 8; i++) {
          R[dim] += a[i] * A[nYdim * i + dim] + b[i] * B[nYdim * i + dim];
        }
      }
    }

    // cross-derivative W = (dS)^2 / du / dv
    // W = dRdSl * (dQdSd * A[0] + dQdDd * A[1]) + dRdDl * (dQdSd * A[2] + dQdDd * A[3]) +
    //     dRdSr * (dQdSd * A[4] + dQdDd * A[5]) + dRdDr * (dQdSd * A[6] + dQdDd * A[7]) +
    //     dRdSl * (dQdSu * B[0] + dQdDu * B[1]) + dRdDl * (dQdSu * B[2] + dQdDu * B[3]) +
    //     dRdSr * (dQdSu * B[4] + dQdDu * B[5]) + dRdDr * (dQdSu * B[6] + dQdDu * B[7]);

    {
      DataT a[8] = {dRdSl * dQdSd, dRdSl * dQdDd, dRdDl * dQdSd, dRdDl * dQdDd,
                    dRdSr * dQdSd, dRdSr * dQdDd, dRdDr * dQdSd, dRdDr * dQdDd};
      DataT b[8] = {dRdSl * dQdSu, dRdSl * dQdDu, dRdDl * dQdSu, dRdDl * dQdDu,
                    dRdSr * dQdSu, dRdSr * dQdDu, dRdDr * dQdSu, dRdDr * dQdDu};

      // W = sum a[i]*A[i] + b[i]*B[i]

      for (int32_t dim = 0; dim < nYdim; dim++) {
        W[dim] = 0;
        for (int32_t i = 0; i < 8; i++) {
          W[dim] += a[i] * A[nYdim * i + dim] + b[i] * B[nYdim * i + dim];
        }
      }
    }
  }

  /// Zero-copy-safe interpolation.
  ///
  /// Identical to interpolateAtU() but takes explicit flat buffer pointers for
  /// the two 1-D grids instead of relying on the internal (potentially stale)
  /// mFlatBufferPtr inside mGridX1 / mGridX2.
  ///
  /// Use this overload when the spline object was transported across DPL/FairMQ
  /// process boundaries via shared memory
  /// called (zero-copy, read-only buffer).
  ///
  /// How to obtain the buffer pointers from TPCFastTransformPOD:
  ///
  ///   const char* splineFlatBuf = podBuf + pod.getFlatBufferOffset(scenarioID);
  ///   // gridX1 is always at offset 0 of the spline flat buffer:
  ///   const char* gridX1FlatBuf = splineFlatBuf;
  ///   // gridX2 comes after gridX1 (use the offset stored in the spline object):
  ///   const char* gridX2FlatBuf = splineFlatBuf + spline.getGridX2Offset();
  ///
  /// \param gridX1FlatBuf  Pointer to the flat buffer of mGridX1
  /// \param gridX2FlatBuf  Pointer to the flat buffer of mGridX2
  /// \param inpYdim        Number of Y dimensions
  /// \param Parameters     Spline correction data for this (sector, row, splineID)
  /// \param u1, u2         Interpolation coordinates
  /// \param S              Output array of length inpYdim
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtUZeroCopy(const char* gridX1FlatBuf,
                                     const char* gridX2FlatBuf,
                                     int32_t inpYdim,
                                     GPUgeneric() const DataT Parameters[],
                                     DataT u1, DataT u2,
                                     GPUgeneric() DataT S[/*inpYdim*/]) const
  {
    const auto nYdimTmp = SplineUtil::getNdim<YdimT>(inpYdim);
    const int32_t nYdim = nYdimTmp.get();
    const auto nYdim4 = nYdim * 4;

    const DataT& u = u1;
    const DataT& v = u2;

    // getNumberOfKnots() is safe: mNumberOfKnots is a plain int stored directly
    // in the Spline1DContainer struct, not behind mFlatBufferPtr.
    int32_t nu = this->mGridX1.getNumberOfKnots();

    // Use buffer-aware accessors instead of mGridX1.getLeftKnotIndexForU() and
    // mGridX1.getKnot(). Both of the standard versions dereference mFlatBufferPtr
    // (via mUtoKnotMap and the knot array), which is stale after cross-process copy.
    int32_t iu = this->mGridX1.getLeftKnotIndexForUFromBuffer(gridX1FlatBuf, u);
    int32_t iv = this->mGridX2.getLeftKnotIndexForUFromBuffer(gridX2FlatBuf, v);

    const auto& knotU = this->mGridX1.template getKnotFromBuffer<kNotSafe>(gridX1FlatBuf, iu);
    const auto& knotV = this->mGridX2.template getKnotFromBuffer<kNotSafe>(gridX2FlatBuf, iv);

    const DataT* A = Parameters + (nu * iv + iu) * nYdim4;
    const DataT* B = A + nYdim4 * nu;

    // getSderivativesOverParsAtU() is pure math on the Knot struct fields {u, Li}.
    // It does NOT touch mFlatBufferPtr, so it is safe on the zero-copy path.
    DataT dSl, dDl, dSr, dDr, dSd, dDd, dSu, dDu;
    this->mGridX1.template getSderivativesOverParsAtU<DataT>(knotU, u, dSl, dDl, dSr, dDr);
    this->mGridX2.template getSderivativesOverParsAtU<DataT>(knotV, v, dSd, dDd, dSu, dDu);

    DataT a[8] = {dSl * dSd, dSl * dDd, dDl * dSd, dDl * dDd,
                  dSr * dSd, dSr * dDd, dDr * dSd, dDr * dDd};
    DataT b[8] = {dSl * dSu, dSl * dDu, dDl * dSu, dDl * dDu,
                  dSr * dSu, dSr * dDu, dDr * dSu, dDr * dDu};

    for (int32_t dim = 0; dim < nYdim; dim++) {
      S[dim] = 0;
      for (int32_t i = 0; i < 8; i++) {
        S[dim] += a[i] * A[nYdim * i + dim] + b[i] * B[nYdim * i + dim];
      }
    }
  }
};

/// ==================================================================================================
/// Specialization 1: YdimT>0 where the number of Y dimensions is taken from template parameters
/// at the compile time
///
template <typename DataT, int32_t YdimT, class FlatBase>
class Spline2DSpec<DataT, YdimT, 1, FlatBase> : public Spline2DSpec<DataT, YdimT, 0, FlatBase>
{
  using ParentSpec = Spline2DSpec<DataT, YdimT, 0, FlatBase>;

 public:
#if !defined(GPUCA_GPUCODE)
  /// Default constructor — skips recreate for NoFlatObject (no owned buffer)
  Spline2DSpec() : ParentSpec()
  {
    if constexpr (!std::is_same_v<FlatBase, NoFlatObject>) {
      recreate(2, 2);
    }
  }

  /// Constructor for a regular spline
  Spline2DSpec(int32_t nKnotsX1, int32_t nKnotsX2) : ParentSpec()
  {
    recreate(nKnotsX1, nKnotsX2);
  }
  /// Constructor for an irregular spline
  Spline2DSpec(int32_t nKnotsX1, const int32_t knotU1[], int32_t nKnotsX2, const int32_t knotU2[]) : ParentSpec()
  {
    recreate(nKnotsX1, knotU1, nKnotsX2, knotU2);
  }
  /// Copy constructor
  Spline2DSpec(const Spline2DSpec& v) : ParentSpec()
  {
    ParentSpec::cloneFromObject(v, nullptr);
  }
  /// Constructor for a regular spline
  void recreate(int32_t nKnotsX1, int32_t nKnotsX2) { ParentSpec::recreate(YdimT, nKnotsX1, nKnotsX2); }

  /// Constructor for an irregular spline
  void recreate(int32_t nKnotsX1, const int32_t knotU1[], int32_t nKnotsX2, const int32_t knotU2[]) { ParentSpec::recreate(YdimT, nKnotsX1, knotU1, nKnotsX2, knotU2); }
#endif

  /// Get number of Y dimensions
  GPUd() constexpr int32_t getYdimensions() const { return YdimT; }

  /// Number of parameters
  GPUd() int32_t getNumberOfParameters() const { return (4 * YdimT) * this->getNumberOfKnots(); }

  /// Size of the parameter array in bytes
  GPUd() size_t getSizeOfParameters() const { return (sizeof(DataT) * 4 * YdimT) * this->getNumberOfKnots(); }

  ///  _______  Expert tools: interpolation with given nYdim and external Parameters _______

  /// Get interpolated value for an YdimT-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtU(GPUgeneric() const DataT Parameters[], DataT u1, DataT u2, GPUgeneric() DataT S[/*YdimT*/]) const
  {
    ParentSpec::template interpolateAtU<SafeT>(YdimT, Parameters, u1, u2, S);
  }

  /// Forwarding overload for Spec 1 (compile-time YdimT).
  /// Passes YdimT as inpYdim directly to the Spec 0 implementation.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtUZeroCopy(const char* gridX1FlatBuf, const char* gridX2FlatBuf, GPUgeneric() const DataT Parameters[], DataT u1, DataT u2, GPUgeneric() DataT S[/*YdimT*/]) const
  {
    ParentSpec::template interpolateAtUZeroCopy<SafeT>(gridX1FlatBuf, gridX2FlatBuf, YdimT, Parameters, u1, u2, S);
  }

  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateParametersAtU(GPUgeneric() const DataT Parameters[], DataT u1, DataT u2, GPUgeneric() DataT P[/* 4*YdimT */]) const
  {
    ParentSpec::template interpolateParametersAtU<SafeT>(YdimT, Parameters, u1, u2, P);
  }

  /// Get interpolated value for an YdimT-dimensional S(u1,u2) using spline parameters Parameters.
  template <SafetyLevel SafeT = SafetyLevel::kSafe>
  GPUd() void interpolateAtUold(GPUgeneric() const DataT Parameters[], DataT u1, DataT u2, GPUgeneric() DataT S[/*nYdim*/]) const
  {
    ParentSpec::template interpolateAtUold<SafeT>(YdimT, Parameters, u1, u2, S);
  }
};

/// ==================================================================================================
/// Specialization 2 (YdimT<=0) where the numbaer of Y dimensions
/// must be set in the runtime via a constructor parameter
///
template <typename DataT, int32_t YdimT, class FlatBase>
class Spline2DSpec<DataT, YdimT, 2, FlatBase> : public Spline2DSpec<DataT, YdimT, 0, FlatBase>
{
  using ParentSpec = Spline2DSpec<DataT, YdimT, 0, FlatBase>;

 public:
#if !defined(GPUCA_GPUCODE)
  /// Default constructor — skips recreate for NoFlatObject (no owned buffer)
  Spline2DSpec() : ParentSpec()
  {
    if constexpr (!std::is_same_v<FlatBase, NoFlatObject>) {
      ParentSpec::recreate(0, 2, 2);
    }
  }

  /// Constructor for a regular spline
  Spline2DSpec(int32_t nYdim, int32_t nKnotsX1, int32_t nKnotsX2) : ParentSpec()
  {
    ParentSpec::recreate(nYdim, nKnotsX1, nKnotsX2);
  }

  /// Constructor for an irregular spline
  Spline2DSpec(int32_t nYdim, int32_t nKnotsX1, const int32_t knotU1[], int32_t nKnotsX2, const int32_t knotU2[]) : ParentSpec()
  {
    ParentSpec::recreate(nYdim, nKnotsX1, knotU1, nKnotsX2, knotU2);
  }

  /// Copy constructor
  Spline2DSpec(const Spline2DSpec& v) : ParentSpec()
  {
    cloneFromObject(v, nullptr);
  }

  /// Constructor for a regular spline
  void recreate(int32_t nYdim, int32_t nKnotsX1, int32_t nKnotsX2) { ParentSpec::recreate(nYdim, nKnotsX1, nKnotsX2); }

  /// Constructor for an irregular spline
  void recreate(int32_t nYdim, int32_t nKnotsX1, const int32_t knotU1[], int32_t nKnotsX2, const int32_t knotU2[]) { ParentSpec::recreate(nYdim, nKnotsX1, knotU1, nKnotsX2, knotU2); }
#endif
};

/// ==================================================================================================
/// Specialization 3, where the number of Y dimensions is 1.
///
template <typename DataT, class FlatBase>
class Spline2DSpec<DataT, 1, 3, FlatBase> : public Spline2DSpec<DataT, 1, SplineUtil::getSpec(999), FlatBase>
{
  using ParentSpec = Spline2DSpec<DataT, 1, SplineUtil::getSpec(999), FlatBase>;

 public:
  /// Simplified interface for 1D: return the interpolated value
  GPUd() DataT interpolate(DataT x1, DataT x2) const
  {
    DataT S = 0;
    ParentSpec::interpolate(x1, x2, &S);
    return S;
  }
};
} // namespace o2::gpu

#endif
