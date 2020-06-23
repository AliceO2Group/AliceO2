// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Spline2D.h
/// \brief Definition of Spline2D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE2D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE2D_H

#include "Spline1D.h"
#include "FlatObject.h"
#include "GPUCommonDef.h"

#if !defined(__CINT__) && !defined(__ROOTCINT__) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_VC) && defined(__cplusplus) && __cplusplus >= 201703L
#include <Vc/Vc>
#include <Vc/SimdArray>
#endif

class TFile;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The Spline2D class performs a cubic spline interpolation on an two-dimensional nonunifom grid.
/// The class is an extension of the Spline1D class.
/// See Spline1D.h for more details.
///
/// The spline S(x1,x2) approximates a function F(x1,x2):R^2->R^m,
/// where x1,x2 belong to [x1Min,x1Max] x [x2Min,x2Max]. F value may be multi-dimensional.
///
/// --- Example of creating a spline ---
///
///  auto F = [&](float x1, float x2, float f[] ) {
///   f[0] = 1.f + x1 + x2*x2; // F(x1,x2)
///  };
///  const int nKnotsU=2;
///  const int nKnotsV=3;
///  int knotsU[nKnotsU] = {0, 1};
///  int knotsV[nKnotsV] = {0, 2, 5};
///  Spline2D<float,1> spline(nKnotsU, knotsU, nKnotsV, knotsV ); // prepare memory for 1-dimensional F
///  spline.approximateFunction(0., 1., 0.,1., F); //initialize spline to approximate F on area [0., 1.]x[0., 1.]
///  float S = spline.interpolate(.1, .3 ); // interpolated value at (.1,.3)
///
///  --- See also Spline2D::test();
///

/// Base class to store data members and non-inline methods
template <typename DataT, bool isConsistentT>
class Spline2DBase : public FlatObject
{
 public:
  /// _____________  Version control __________________________

  /// Version control
  GPUhd() static constexpr int getVersion() { return 1; }

  /// _____________  Constructors / destructors __________________________

#if !defined(GPUCA_GPUCODE)
  /// constructor && default constructor for the ROOT streamer
  Spline2DBase(int nDim = 1);
#else
  /// Disable constructors
  Spline2DBase() CON_DELETE;
#endif
  /// Disable constructors
  Spline2DBase(const Spline2DBase&) CON_DELETE;
  Spline2DBase& operator=(const Spline2DBase&) CON_DELETE;

  /// Destructor
  ~Spline2DBase() CON_DEFAULT;

  /// _______________  Construction interface  ________________________

#if !defined(GPUCA_GPUCODE)
  /// Constructor for a regular spline.
  void recreate(int numberOfKnotsU1, int numberOfKnotsU2);

  /// Constructor for an irregular spline
  void recreate(int numberOfKnotsU1, const int knotsU1[], int numberOfKnotsU2, const int knotsU2[]);
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// approximate a function F with this spline.
  void approximateFunction(DataT x1Min, DataT x1Max, DataT x2Min, DataT x2Max,
                           std::function<void(DataT x1, DataT x2, DataT f[])> F,
                           int nAxiliaryDataPointsU1 = 4, int nAxiliaryDataPointsU2 = 4);
#endif

  /// _______________  IO   ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// write a class object to the file
  int writeToFile(TFile& outf, const char* name);

  /// read a class object from the file
  static Spline2DBase* readFromFile(TFile& inpf, const char* name);
#endif

  /// _______________  Getters   ________________________

  /// Get number of F dimensions
  GPUhd() int getFdimensions() const { return mFdim; }

  ///
  GPUhd() static constexpr bool isConsistent() { return isConsistentT; }

  /// Get minimal required alignment for the spline parameters
  GPUhd() static constexpr size_t getParameterAlignmentBytes() { return 16; }

  /// Number of parameters
  GPUhd() int getNumberOfParameters() const { return (4 * mFdim) * getNumberOfKnots(); }

  /// Size of the parameter array in bytes
  GPUhd() size_t getSizeOfParameters() const { return sizeof(DataT) * getNumberOfParameters(); }

  /// Get number total of knots: UxV
  GPUhd() int getNumberOfKnots() const { return mGridU1.getNumberOfKnots() * mGridU2.getNumberOfKnots(); }

  /// Get 1-D grid for U1 coordinate
  GPUhd() const Spline1D<DataT>& getGridU1() const { return mGridU1; }

  /// Get 1-D grid for U2 coordinate
  GPUhd() const Spline1D<DataT>& getGridU2() const { return mGridU2; }

  /// Get 1-D grid for U1 or U2 coordinate
  GPUhd() const Spline1D<DataT>& getGrid(int iu) const { return (iu == 0) ? mGridU1 : mGridU2; }

  /// Get u1,u2 of i-th knot
  GPUhd() void getKnotU(int iKnot, DataT& u1, DataT& u2) const;

  /// Get index of a knot (iKnotU1,iKnotU2)
  GPUhd() int getKnotIndex(int iKnotU1, int iKnotU2) const;

  /// Get number of F parameters
  GPUhd() DataT* getFparameters() { return mFparameters; }

  /// Get number of F parameters
  GPUhd() const DataT* getFparameters() const { return mFparameters; }

  /// _______________  Technical stuff  ________________________

  /// Get offset of GridU flat data in the flat buffer
  GPUhd() size_t getGridU1Offset() const { return mGridU1.getFlatBufferPtr() - mFlatBufferPtr; }

  /// Get offset of GridU2 flat data in the flat buffer
  GPUhd() size_t getGridU2Offset() const { return mGridU2.getFlatBufferPtr() - mFlatBufferPtr; }

  /// Set X range
  GPUhd() void setXrange(DataT x1Min, DataT x1Max, DataT x2Min, DataT x2Max);

  /// Print method
  void print() const;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
  /// Test the class functionality
  static int test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

#if !defined(GPUCA_GPUCODE)
  void cloneFromObject(const Spline2DBase& obj, char* newFlatBufferPtr);
  void moveBufferTo(char* newBufferPtr);
#endif

  using FlatObject::releaseInternalBuffer;

  void destroy();
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

 private:
  int mFdim; ///< dimentionality of F

 protected:                /// _____________  Data members  ____________
  Spline1D<DataT> mGridU1; ///< grid for U axis
  Spline1D<DataT> mGridU2; ///< grid for V axis
  DataT* mFparameters;     //! (transient!!) F-dependent parameters of the spline
#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(Spline2DBase, 1);
#endif
};

///
/// The main Spline class. Contains constructors and interpolation.
///
/// F dimensions can be set as a template parameter (faster; the only option for the GPU)
/// or as a constructor parameter (slower; the only option for the ROOT interpretator).
/// In a compiled CPU code one can use both options.
///
template <typename DataT, int nFdimT = 0, bool isConsistentT = 1>
class Spline2D : public Spline2DBase<DataT, isConsistentT>
{
 public:
  typedef Spline2DBase<DataT, isConsistentT> TBase;

  /// _____________  Constructors / destructors __________________________

#if !defined(GPUCA_GPUCODE)

  /// Constructor for a regular spline && default constructor
  Spline2D(int numberOfKnotsU1 = 2, int numberOfKnotsU2 = 2);

  /// Constructor for an irregular spline
  Spline2D(int numberOfKnotsU1, const int knotsU1[], int numberOfKnotsU2, const int knotsU2[]);

  /// Copy constructor
  Spline2D(const Spline2D&);

  /// Assignment operator
  Spline2D& operator=(const Spline2D&);
#else
  /// Disable constructors for the GPU implementation
  Spline2D() CON_DELETE;
  Spline2D(const Spline2D&) CON_DELETE;
  Spline2D& operator=(const Spline2D&) CON_DELETE;
#endif

  /// Destructor
  ~Spline2D() CON_DEFAULT;

  /// _______________  Main functionality   ________________________

  /// Get interpolated value for F(x1,x2)
  GPUhd() void interpolate(DataT x1, DataT x2, GPUgeneric() DataT S[]) const;

  /// Get interpolated value for the first dimension of F(x1,x2). (Simplified interface for 1D)
  GPUhd() DataT interpolate(DataT x1, DataT x2) const;

  /// Same as interpolate(), but using vectorized calculation.
  GPUhd() void interpolateVec(DataT x1, DataT x2, GPUgeneric() DataT S[]) const;

  /// ================ Expert tools   ================================

  /// Get interpolated value for an nFdim-dimensional F(u) using spline parameters Fparameters.
  /// Fparameters can be created via SplineHelper2D.
  GPUhd() void interpolateU(GPUgeneric() const DataT Fparameters[],
                            DataT u1, DataT u2, GPUgeneric() DataT Su[]) const;

  /// Same as interpolateU(), but using vectorized calculation.
  GPUhd() void interpolateUvec(GPUgeneric() const DataT Fparameters[],
                               DataT u1, DataT u2, GPUgeneric() DataT Su[]) const;

  /// _______________  IO   ________________________

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// write a class object to the file
  using TBase::writeToFile;

  /// read a class object from the file
  static Spline2D* readFromFile(TFile& inpf, const char* name)
  {
    return reinterpret_cast<Spline2D*>(TBase::readFromFile(inpf, name));
  }
#endif

  /// _______________  Getters   ________________________

  /// Get number of F dimensions.
  GPUhd() static constexpr int getFdimensions() { return nFdimT; }

  using TBase::mFparameters;
  using TBase::mGridU1;
  using TBase::mGridU2;
};

///
/// ========================================================================================================
///       Inline implementations of some methods
/// ========================================================================================================
///

template <typename DataT, bool isConsistentT>
GPUhdi() void Spline2DBase<DataT, isConsistentT>::getKnotU(int iKnot, DataT& u1, DataT& u2) const
{
  /// Get u1,u2 of i-th knot
  int nu1 = mGridU1.getNumberOfKnots();
  int iu2 = iKnot / nu1;
  int iu1 = iKnot % nu1;
  u1 = mGridU1.getKnot(iu1).u;
  u2 = mGridU2.getKnot(iu2).u;
}

template <typename DataT, bool isConsistentT>
GPUhdi() int Spline2DBase<DataT, isConsistentT>::getKnotIndex(int iKnotU1, int iKnotU2) const
{
  /// Get index of a knot (iKnotU1,iKnotU2)
  int nu1 = mGridU1.getNumberOfKnots();
  return nu1 * iKnotU2 + iKnotU1;
}

template <typename DataT, bool isConsistentT>
GPUhdi() void Spline2DBase<DataT, isConsistentT>::setXrange(
  DataT x1Min, DataT x1Max, DataT x2Min, DataT x2Max)
{
  mGridU1.setXrange(x1Min, x1Max);
  mGridU2.setXrange(x2Min, x2Max);
}

#if !defined(GPUCA_GPUCODE)

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhdi() Spline2D<DataT, nFdimT, isConsistentT>::
  Spline2D(int numberOfKnotsU1, int numberOfKnotsU2)
  : Spline2DBase<DataT, isConsistentT>(nFdimT)
{
  this->recreate(numberOfKnotsU1, numberOfKnotsU2);
}

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhdi() Spline2D<DataT, nFdimT, isConsistentT>::
  Spline2D(int numberOfKnotsU1, const int knotsU1[],
           int numberOfKnotsU2, const int knotsU2[])
  : Spline2DBase<DataT, isConsistentT>(nFdimT)
{
  this->recreate(numberOfKnotsU1, knotsU1, numberOfKnotsU2, knotsU2);
}

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhdi() Spline2D<DataT, nFdimT, isConsistentT>::
  Spline2D(const Spline2D& spline)
  : Spline2DBase<DataT, isConsistentT>(nFdimT)
{
  this->cloneFromObject(spline, nullptr);
}

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhd() Spline2D<DataT, nFdimT, isConsistentT>& Spline2D<DataT, nFdimT, isConsistentT>::
  operator=(const Spline2D<DataT, nFdimT, isConsistentT>& spline)
{
  this->cloneFromObject(spline, nullptr);
  return *this;
}
#endif

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhdi() void Spline2D<DataT, nFdimT, isConsistentT>::
  interpolate(DataT x1, DataT x2, GPUgeneric() DataT S[]) const
{
  /// Get interpolated value for F(x1,x2)
  assert(isConsistentT);
  if (isConsistentT) {
    interpolateU(mFparameters, mGridU1.convXtoU(x1), mGridU2.convXtoU(x2), S);
  } else {
    for (int i = 0; i < nFdimT; i++) {
      S[i] = -1.;
    }
  }
}

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhdi() DataT Spline2D<DataT, nFdimT, isConsistentT>::
  interpolate(DataT x1, DataT x2) const
{
  /// Simplified interface for 1D: get interpolated value for the first dimension of F(x)

#if defined(GPUCA_GPUCODE)
  DataT S[nFdimT]; // constexpr array size for the GPU compiler
#else
  DataT S[getFdimensions()];
#endif
  interpolate(x1, x2, S);
  return S[0];
}

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhdi() void Spline2D<DataT, nFdimT, isConsistentT>::interpolateVec(
  DataT x1, DataT x2, GPUgeneric() DataT S[]) const
{
  /// Same as interpolate(), but using vectorized calculation
  assert(isConsistentT);
  if (isConsistentT) {
    interpolateUvec(mFparameters, mGridU1.convXtoU(x1), mGridU2.convXtoU(x2), S);
  } else {
    for (int i = 0; i < getFdimensions(); i++) {
      S[i] = 0.;
    }
  }
}

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhdi() void Spline2D<DataT, nFdimT, isConsistentT>::interpolateU(
  GPUgeneric() const DataT Fparameters[],
  DataT u, DataT v, GPUgeneric() DataT S[]) const
{
  /// Get interpolated value for an nFdim-dimensional F(u) using spline parameters Fparameters.
  /// Fparameters can be created via SplineHelper2D.

  int nu = mGridU1.getNumberOfKnots();
  int iu = mGridU1.getKnotIndexU(u);
  int iv = mGridU2.getKnotIndexU(v);

  const typename Spline1D<DataT>::Knot& knotU = mGridU1.getKnot(iu);
  const typename Spline1D<DataT>::Knot& knotV = mGridU2.getKnot(iv);

#if defined(GPUCA_GPUCODE) // constexpr array size for the GPU compiler
  const int nFdim = nFdimT;
#else
  const int nFdim = getFdimensions();
#endif

  const int nFdim2 = nFdimT * 2;
  const int nFdim4 = nFdimT * 4;

  // X:=Sx, Y:=Sy, Z:=Sz

  const DataT* par00 = Fparameters + (nu * iv + iu) * nFdim4; // values { {X,Y,Z}, {X,Y,Z}'v, {X,Y,Z}'u, {X,Y,Z}''vu } at {u0, v0}
  const DataT* par10 = par00 + nFdim4;                        // values { ... } at {u1, v0}
  const DataT* par01 = par00 + nFdim4 * nu;                   // values { ... } at {u0, v1}
  const DataT* par11 = par01 + nFdim4;                        // values { ... } at {u1, v1}

  DataT Su0[nFdim4]; // values { {X,Y,Z,X'v,Y'v,Z'v}(v0), {X,Y,Z,X'v,Y'v,Z'v}(v1) }, at u0
  DataT Du0[nFdim4]; // derivatives {}'_u  at u0
  DataT Su1[nFdim4]; // values { {X,Y,Z,X'v,Y'v,Z'v}(v0), {X,Y,Z,X'v,Y'v,Z'v}(v1) }, at u1
  DataT Du1[nFdim4]; // derivatives {}'_u  at u1

  for (int i = 0; i < nFdim2; i++) {
    Su0[i] = par00[i];
    Su0[nFdim2 + i] = par01[i];

    Du0[i] = par00[nFdim2 + i];
    Du0[nFdim2 + i] = par01[nFdim2 + i];

    Su1[i] = par10[i];
    Su1[nFdim2 + i] = par11[i];

    Du1[i] = par10[nFdim2 + i];
    Du1[nFdim2 + i] = par11[nFdim2 + i];
  }

  DataT parU[nFdim4]; // interpolated values { {X,Y,Z,X'v,Y'v,Z'v}(v0), {X,Y,Z,X'v,Y'v,Z'v}(v1) } at u
  mGridU1.interpolateU(nFdim4, knotU, Su0, Du0, Su1, Du1, u, parU);

  const DataT* Sv0 = parU + 0;
  const DataT* Dv0 = parU + nFdim;
  const DataT* Sv1 = parU + nFdim2;
  const DataT* Dv1 = parU + nFdim2 + nFdim;

  mGridU2.interpolateU(nFdim, knotV, Sv0, Dv0, Sv1, Dv1, v, S);
}

template <typename DataT, int nFdimT, bool isConsistentT>
GPUhdi() void Spline2D<DataT, nFdimT, isConsistentT>::interpolateUvec(
  GPUgeneric() const DataT Fparameters[],
  DataT u1, DataT u2, GPUgeneric() DataT S[]) const
{
  /// Same as interpolateU(), but using vectorized calculation
  interpolateU(mFparameters, u1, u2, S);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
