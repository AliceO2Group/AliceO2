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

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The Spline2D class represents spline interpolation on a two-dimensional nonunifom grid.
/// The class is an extension of the Spline1D class. See Spline1D.h for more details.
///
/// Spline knots and spline parameters are initialized separatelly.
/// The knots are stored inside the Spline2D class, while the parameters must be stored outside
/// and must be provided by the user at each call of the interpolation.
/// This implementation allows one to use the same spline object for interpolation of different functions on the same knots.
///
/// --- Interpolation ---
///
/// The spline S(u,v) interpolates a function F:[u,v]->R^m,
/// where u,v belong to [0,Umax]x[0,Vmax].
///
/// --- Knots ---
///
/// The spline knots belong to [0, Umax][0,Vmax] and have integer coordinates.
///
/// To interpolate a function on an interval other than [0,Umax]x[0,Vmax], one should scale the U/V coordinates and the S derivatives in the parameter array.
///
/// The minimal number of knots is 2, the minimal Umax & Vmax is 1
///
/// --- Function values at knots---
///
/// Nothing which depends on F is stored in the class,
/// therefore one can use the same spline object for interpolation of different input functions.
/// The spline parameters S_i and D_i must be provided by the user for each call.
/// The format of the spline parameters:
///   {
///     { (Fx,Fy,Fz), (Fx,Fy,Fz)'_v, (Fx,Fy,Fz)'_u, (Fx,Fy,Fz)''_vu } at knot 0,
///     {                      ...                      } at knot 1,
///                            ...
///   }
///   The parameter array has to be provided by the user for each call of the interpolation.
///   It can be created for a given input function using SplineHelper2D class.
///
/// ---- Flat Object implementation ----
///
/// The class inherits from the FlatObjects. Copy construction can be only done via the FlatObject interface.
///
/// --- Example of creating a spline ---
///
///  auto F = [&](float u, float v ) -> float {
///   return ...; // F(u,v)
///  };
///  const int nKnotsU=2;
///  const int nKnotsV=3;
///  int knotsU[nKnotsU] = {0, 2};
///  int knotsV[nKnotsV] = {0, 3, 6};
///  Spline2D spline;
///  spline.constructKnots(nKnotsU, knotsU, nKnotsV, knotsV );
///  SplineHelper2D helper;
///  helper.SetSpline( spline, 2, 2);
///  std::unique_ptr<float[]> parameters = helper.constructParameters(1, F, 0.f, 1.f, 0.f, 1.f);
///  float S;
///  spline.interpolate(1, parameters.get(), 0.0, 0.0, &S ); // S == F(0.,0.)
///  spline.interpolate(1, parameters.get(), 1.0, 1.1, &S ); // S == some interpolated value
///  spline.interpolate(1, parameters.get(), 2.0, 3.0, &S ); // S == F(1., 0.5 )
///  spline.interpolate(1, parameters.get(), 2.0, 6.0, &S ); // S == F(1., 1.)
///
///  --- See also Spline2D::test();
///
class Spline2D : public FlatObject
{
 public:
  /// _____________  Version control __________________________

  /// Version number
  GPUd() static constexpr int getVersion() { return 1; }

  /// _____________  Constructors / destructors __________________________

#if !defined(GPUCA_GPUCODE)
  /// Default constructor. Creates a spline with 2 knots.
  Spline2D() : FlatObject(), mGridU(), mGridV() { constructKnotsRegular(2, 2); }

  /// Constructor for an irregular spline.
  Spline2D(int numberOfKnotsU, const int knotsU[], int numberOfKnotsV, const int knotsV[]) : FlatObject(), mGridU(), mGridV()
  {
    constructKnots(numberOfKnotsU, knotsU, numberOfKnotsV, knotsV);
  }

  /// Constructor for a regular spline.
  Spline2D(int numberOfKnotsU, int numberOfKnotsV) : FlatObject()
  {
    constructKnotsRegular(numberOfKnotsU, numberOfKnotsV);
  }
#else
  /// Disable the constructor for the GPU implementation
  Spline2D() CON_DELETE;
#endif

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject() instead
  Spline2D(const Spline2D&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject() instead
  Spline2D& operator=(const Spline2D&) CON_DELETE;

  /// Destructor
  ~Spline2D() CON_DEFAULT;

  /// _______________  Construction interface  ________________________

#if !defined(GPUCA_GPUCODE)
  /// Constructor
  ///
  /// Number of created knots may differ from the input values:
  /// - Edge knots {0} and {Umax/Vmax} will be added if they are not present.
  /// - Duplicated knots, knots with a negative coordinate will be deleted
  /// - At least 2 knots for each axis will be created
  ///
  /// \param numberOfKnotsU     Number of knots in knotsU[] array
  /// \param knotsU             Array of knot positions (integer values)
  ///
  /// \param numberOfKnotsV     Number of knots in knotsV[] array
  /// \param knotsV             Array of knot positions (integer values)
  ///
  void constructKnots(int numberOfKnotsU, const int knotsU[], int numberOfKnotsV, const int knotsV[]);

  /// Constructor for a regular spline. Knots will be placed at the positions i/(numberOfKnots-1)
  void constructKnotsRegular(int numberOfKnotsU, int numberOfKnotsV);
#endif

  /// _______________  Main functionality   ________________________

  /// Get interpolated value for f(u,v)
  template <typename T>
  GPUd() void interpolate(int Ndim, GPUgeneric() const T* parameters, float u, float v, GPUgeneric() T Suv[/*Ndim*/]) const;

  /// Same as interpolate, but using vectorized calculation.
  /// \param parameters should be at least 128-bit aligned
  template <typename T>
  GPUd() void interpolateVec(int Ndim, GPUgeneric() const T* parameters, float u, float v, GPUgeneric() T Suv[/*Ndim*/]) const;

  /// _______________  Getters   ________________________

  /// Get minimal required alignment for the spline parameters

  template <typename T>
  static constexpr size_t getParameterAlignmentBytes(int Ndim)
  {
    return std::min<4 * sizeof(T) * Ndim, 16>;
  }

  /// Size of the parameter array in bytes
  template <typename T>
  GPUd() size_t getSizeOfParameters(int Ndim) const
  {
    return sizeof(T) * (size_t)getNumberOfParameters(Ndim);
  }

  /// Number of parameters
  GPUd() int getNumberOfParameters(int Ndim) const
  {
    return (4 * Ndim) * getNumberOfKnots();
  }

  /// Get number total of knots: UxV
  GPUd() int getNumberOfKnots() const { return mGridU.getNumberOfKnots() * mGridV.getNumberOfKnots(); }

  /// Get 1-D grid for U coordinate
  GPUd() const Spline1D& getGridU() const { return mGridU; }

  /// Get 1-D grid for V coordinate
  GPUd() const Spline1D& getGridV() const { return mGridV; }

  /// Get 1-D grid for U or V coordinate
  GPUd() const Spline1D& getGrid(int uv) const { return (uv == 0) ? mGridU : mGridV; }

  /// Get u,v of i-th knot
  GPUd() void getKnotUV(int iKnot, float& u, float& v) const;

  /// _______________  Technical stuff  ________________________

  /// Get offset of GridU flat data in the flat buffer
  size_t getGridUOffset() const { return mGridU.getFlatBufferPtr() - mFlatBufferPtr; }

  /// Get offset of GridV flat data in the flat buffer
  size_t getGridVOffset() const { return mGridV.getFlatBufferPtr() - mFlatBufferPtr; }

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
  void cloneFromObject(const Spline2D& obj, char* newFlatBufferPtr);
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
  ///
  /// ====  Data members   ====
  ///

  Spline1D mGridU; ///< grid for U axis
  Spline1D mGridV; ///< grid for V axis
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

GPUdi() void Spline2D::getKnotUV(int iKnot, float& u, float& v) const
{
  /// Get u,v of i-th knot
  const Spline1D& gridU = getGridU();
  const Spline1D& gridV = getGridV();
  int nu = gridU.getNumberOfKnots();
  int iv = iKnot / nu;
  int iu = iKnot % nu;
  u = gridU.getKnot(iu).u;
  v = gridV.getKnot(iv).u;
}

template <typename T>
GPUdi() void Spline2D::interpolate(int Ndim, GPUgeneric() const T* parameters, float u, float v, GPUgeneric() T Suv[/*Ndim*/]) const
{
  // Get interpolated value for f(u,v) using parameters[getNumberOfParameters()]

  const Spline1D& gridU = getGridU();
  const Spline1D& gridV = getGridV();
  int nu = gridU.getNumberOfKnots();
  int iu = gridU.getKnotIndex(u);
  int iv = gridV.getKnotIndex(v);

  const Spline1D::Knot& knotU = gridU.getKnot(iu);
  const Spline1D::Knot& knotV = gridV.getKnot(iv);

  const int Ndim2 = Ndim * 2;
  const int Ndim4 = Ndim * 4;

  // X:=Sx, Y:=Sy, Z:=Sz

  const T* par00 = parameters + (nu * iv + iu) * Ndim4; // values { {X,Y,Z}, {X,Y,Z}'v, {X,Y,Z}'u, {X,Y,Z}''vu } at {u0, v0}
  const T* par10 = par00 + Ndim4;                       // values { ... } at {u1, v0}
  const T* par01 = par00 + Ndim4 * nu;                  // values { ... } at {u0, v1}
  const T* par11 = par01 + Ndim4;                       // values { ... } at {u1, v1}

  T Su0[Ndim4]; // values { {X,Y,Z,X'v,Y'v,Z'v}(v0), {X,Y,Z,X'v,Y'v,Z'v}(v1) }, at u0
  T Du0[Ndim4]; // derivatives {}'_u  at u0
  T Su1[Ndim4]; // values { {X,Y,Z,X'v,Y'v,Z'v}(v0), {X,Y,Z,X'v,Y'v,Z'v}(v1) }, at u1
  T Du1[Ndim4]; // derivatives {}'_u  at u1

  for (int i = 0; i < Ndim2; i++) {
    Su0[i] = par00[i];
    Su0[Ndim2 + i] = par01[i];

    Du0[i] = par00[Ndim2 + i];
    Du0[Ndim2 + i] = par01[Ndim2 + i];

    Su1[i] = par10[i];
    Su1[Ndim2 + i] = par11[i];

    Du1[i] = par10[Ndim2 + i];
    Du1[Ndim2 + i] = par11[Ndim2 + i];
  }

  T parU[Ndim4]; // interpolated values { {X,Y,Z,X'v,Y'v,Z'v}(v0), {X,Y,Z,X'v,Y'v,Z'v}(v1) } at u
  gridU.interpolate<T>(Ndim4, knotU, Su0, Du0, Su1, Du1, u, parU);

  const T* Sv0 = parU + 0;
  const T* Dv0 = parU + Ndim;
  const T* Sv1 = parU + Ndim2;
  const T* Dv1 = parU + Ndim2 + Ndim;

  gridV.interpolate<T>(Ndim, knotV, Sv0, Dv0, Sv1, Dv1, v, Suv);
}

template <typename T>
GPUdi() void Spline2D::interpolateVec(int Ndim, GPUgeneric() const T* parameters, float u, float v, GPUgeneric() T Suv[/*Ndim*/]) const
{
  // Same as interpolate, but using vectorized calculation.
  // \param parameters should be at least 128-bit aligned

  /// TODO: vectorize
  interpolate<T>(Ndim, parameters, u, v, Suv);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
