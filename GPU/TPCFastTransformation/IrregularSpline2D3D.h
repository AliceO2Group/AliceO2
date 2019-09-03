// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  IrregularSpline2D3D.h
/// \brief Definition of IrregularSpline2D3D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_IRREGULARSPLINE2D3D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_IRREGULARSPLINE2D3D_H

#include "IrregularSpline1D.h"
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
/// The IrregularSpline2D3D class represents twoo-dimensional spline interpolation on nonunifom (irregular) grid.
///
/// The class is flat C structure. No virtual methods, no ROOT types are used.
/// It is designed for spline parameterisation of TPC transformation.
///
/// ---
/// The spline interpolates a generic function F:[u,v)->(x,y,z),
/// where u,v belong to [0,1]x[0,1]
///
/// It is an extension of IrregularSpline1D class, see IrregularSpline1D.h for more details.
///
/// Important:
///   -- The number of knots and their positions may change during initialisation
///   -- Don't forget to call correctEdges() for the array of F values (see IrregularSpline1D.h )
///
/// ------------
///
///  Example of creating a spline:
///
///  const int nKnotsU=5;
///  const int nKnotsV=6;
///  float knotsU[nKnotsU] = {0., 0.25, 0.5, 0.7, 1.};
///  float knotsV[nKnotsV] = {0., 0.2, 0.3, 0.45, 0.8, 1.};
///  IrregularSpline2D3D spline(nKnotsU, knotsU, 4, nKnotsV, knotsV, 10 );
///  float f[nKnotsU*nKnotsV] = { 3.5, 2.0, 1.4, 3.8, 2.3, .... };
///  spline.correctEdges( f );
///  spline.getSpline( f, 0., 0. ); // == 3.5
///  spline.getSpline( f, 0.1, 0.32 ); // == some interpolated value
///
class IrregularSpline2D3D : public FlatObject
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor. Creates an empty uninitialised object
  IrregularSpline2D3D();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject() instead
  IrregularSpline2D3D(const IrregularSpline2D3D&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject() instead
  IrregularSpline2D3D& operator=(const IrregularSpline2D3D&) CON_DELETE;

  /// Destructor
  ~IrregularSpline2D3D() CON_DEFAULT;

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Memory alignment

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

  /// Construction interface

  void cloneFromObject(const IrregularSpline2D3D& obj, char* newFlatBufferPtr);
  void destroy();

  /// Making the data buffer external

  using FlatObject::releaseInternalBuffer;
  void moveBufferTo(char* newBufferPtr);

  /// Moving the class with its external buffer to another location

  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// _______________  Construction interface  ________________________

  /// Constructor
  ///
  /// Number of knots created and their values may differ from the input values:
  /// - Edge knots 0.f and 1.f will be added if they are not present.
  /// - Knot values are rounded to closest axis bins: k*1./numberOfAxisBins.
  /// - Knots which are too close to each other will be merged
  /// - At least 5 knots and at least 4 axis bins will be created for consistency reason
  ///
  /// \param numberOfKnotsU     U axis: Number of knots in knots[] array
  /// \param knotsU             U axis: Array of knots.
  /// \param numberOfAxisBinsU  U axis: Number of axis bins to map U coordinate to
  ///                           an appropriate [knot(i),knot(i+1)] interval.
  ///                           The knot positions have a "granularity" of 1./numberOfAxisBins
  ///
  /// \param numberOfKnotsV     V axis: Number of knots in knots[] array
  /// \param knotsV             V axis: Array of knots.
  /// \param numberOfAxisBinsV  V axis: Number of axis bins to map U coordinate to
  ///                           an appropriate [knot(i),knot(i+1)] interval.
  ///                           The knot positions have a "granularity" of 1./numberOfAxisBins
  ///
  void construct(int numberOfKnotsU, const float knotsU[], int numberOfAxisBinsU, int numberOfKnotsV, const float knotsV[], int numberOfAxisBinsV);

  /// Constructor for a regular spline
  void constructRegular(int numberOfKnotsU, int numberOfKnotsV);

  /// _______________  Main functionality   ________________________

  /// Correction of data values at edge knots.
  ///
  /// It is needed for the fast spline mathematics to work correctly. See explanation in IrregularSpline1D.h header
  ///
  /// \param data array of function values. It has the size of getNumberOfKnots()
  template <typename T>
  GPUd() void correctEdges(T* data) const;

  /// Get interpolated value for f(u,v) using data array correctedData[getNumberOfKnots()] with corrected edges
  template <typename T>
  GPUd() void getSpline(GPUgeneric() const T* correctedData, float u, float v, GPUgeneric() T& x, GPUgeneric() T& y, GPUgeneric() T& z) const;

  /// Same as getSpline, but using vectorized calculation.
  /// \param correctedData should be at least 128-bit aligned
  GPUd() void getSplineVec(const float* correctedData, float u, float v, float& x, float& y, float& z) const;

  /// Get number total of knots: UxV
  GPUd() int getNumberOfKnots() const { return mGridU.getNumberOfKnots() * mGridV.getNumberOfKnots(); }

  /// Get 1-D grid for U coordinate
  GPUd() const IrregularSpline1D& getGridU() const { return mGridU; }

  /// Get 1-D grid for V coordinate
  GPUd() const IrregularSpline1D& getGridV() const { return mGridV; }

  /// Get 1-D grid for U or V coordinate
  GPUd() const IrregularSpline1D& getGrid(int uv) const { return (uv == 0) ? mGridU : mGridV; }

  /// Get u,v of i-th knot
  GPUd() void getKnotUV(int iKnot, float& u, float& v) const;

  /// Get size of the mFlatBuffer data
  size_t getFlatBufferSize() const { return mFlatBufferSize; }

  /// Get pointer to the flat buffer
  const char* getFlatBufferPtr() const { return mFlatBufferPtr; }

  /// Get minimal required alignment for the class
  static constexpr size_t getClassAlignmentBytes() { return 8; }

  /// Get minimal required alignment for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return 8; }

  /// Get minimal required alignment for the spline data
  static constexpr size_t getDataAlignmentBytes() { return 8; }

  /// technical stuff

  /// Get offset of GridU flat data in the flat buffer
  size_t getGridUOffset() const { return mGridU.getFlatBufferPtr() - mFlatBufferPtr; }

  /// Get offset of GridV flat data in the flat buffer
  size_t getGridVOffset() const { return mGridV.getFlatBufferPtr() - mFlatBufferPtr; }

  /// Print method
  void print() const;

 private:

  ///
  /// ====  Data members   ====
  ///

  IrregularSpline1D mGridU; ///< grid for U axis
  IrregularSpline1D mGridV; ///< grid for V axis
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

GPUdi() void IrregularSpline2D3D::getKnotUV(int iKnot, float& u, float& v) const
{
  /// Get u,v of i-th knot
  const IrregularSpline1D& gridU = getGridU();
  const IrregularSpline1D& gridV = getGridV();
  int nu = gridU.getNumberOfKnots();
  int iv = iKnot / nu;
  int iu = iKnot % nu;
  u = gridU.getKnot(iu).u;
  v = gridV.getKnot(iv).u;
}

template <typename T>
GPUd() void IrregularSpline2D3D::correctEdges(T* data) const
{
  const IrregularSpline1D& gridU = getGridU();
  const IrregularSpline1D& gridV = getGridV();
  int nu = gridU.getNumberOfKnots();
  int nv = gridV.getNumberOfKnots();

  { // left edge of U
    int iu = 0;
    const IrregularSpline1D::Knot* s = gridU.getKnots() + iu;
    double c0, c1, c2, c3;
    gridU.getEdgeCorrectionCoefficients(s[0].u, s[1].u, s[2].u, s[3].u, c0, c1, c2, c3);
    for (int iv = 0; iv < nv; iv++) {
      T* f0 = data + (nu * (iv) + iu) * 3;
      T* f1 = f0 + 3;
      T* f2 = f0 + 6;
      T* f3 = f0 + 9;
      for (int idim = 0; idim < 3; idim++) {
        f0[idim] = (T)(c0 * f0[idim] + c1 * f1[idim] + c2 * f2[idim] + c3 * f3[idim]);
      }
    }
  }

  { // right edge of U
    int iu = nu - 4;
    const IrregularSpline1D::Knot* s = gridU.getKnots() + iu;
    double c0, c1, c2, c3;
    gridU.getEdgeCorrectionCoefficients(s[3].u, s[2].u, s[1].u, s[0].u, c3, c2, c1, c0);
    for (int iv = 0; iv < nv; iv++) {
      T* f0 = data + (nu * (iv) + iu) * 3;
      T* f1 = f0 + 3;
      T* f2 = f0 + 6;
      T* f3 = f0 + 9;
      for (int idim = 0; idim < 3; idim++) {
        f3[idim] = (T)(c0 * f0[idim] + c1 * f1[idim] + c2 * f2[idim] + c3 * f3[idim]);
      }
    }
  }

  { // low edge of V
    int iv = 0;
    const IrregularSpline1D::Knot* s = gridV.getKnots() + iv;
    double c0, c1, c2, c3;
    gridV.getEdgeCorrectionCoefficients(s[0].u, s[1].u, s[2].u, s[3].u, c0, c1, c2, c3);
    for (int iu = 0; iu < nu; iu++) {
      T* f0 = data + (nu * iv + iu) * 3;
      T* f1 = f0 + nu * 3;
      T* f2 = f0 + nu * 6;
      T* f3 = f0 + nu * 9;
      for (int idim = 0; idim < 3; idim++) {
        f0[idim] = (T)(c0 * f0[idim] + c1 * f1[idim] + c2 * f2[idim] + c3 * f3[idim]);
      }
    }
  }

  { // high edge of V
    int iv = nv - 4;
    const IrregularSpline1D::Knot* s = gridV.getKnots() + iv;
    double c0, c1, c2, c3;
    gridV.getEdgeCorrectionCoefficients(s[3].u, s[2].u, s[1].u, s[0].u, c3, c2, c1, c0);
    for (int iu = 0; iu < nu; iu++) {
      T* f0 = data + (nu * iv + iu) * 3;
      T* f1 = f0 + nu * 3;
      T* f2 = f0 + nu * 6;
      T* f3 = f0 + nu * 9;
      for (int idim = 0; idim < 3; idim++) {
        f3[idim] = (T)(c0 * f0[idim] + c1 * f1[idim] + c2 * f2[idim] + c3 * f3[idim]);
      }
    }
  }
}

template <typename T>
GPUdi() void IrregularSpline2D3D::getSpline(GPUgeneric() const T* correctedData, float u, float v, GPUgeneric() T& x, GPUgeneric() T& y, GPUgeneric() T& z) const
{
  // Get interpolated value for f(u,v) using data array correctedData[getNumberOfKnots()] with corrected edges

  const IrregularSpline1D& gridU = getGridU();
  const IrregularSpline1D& gridV = getGridV();
  int nu = gridU.getNumberOfKnots();
  int iu = gridU.getKnotIndex(u);
  int iv = gridV.getKnotIndex(v);

  const IrregularSpline1D::Knot& knotU = gridU.getKnot(iu);
  const IrregularSpline1D::Knot& knotV = gridV.getKnot(iv);

  const T* dataV0 = correctedData + (nu * (iv - 1) + iu - 1) * 3;
  const T* dataV1 = dataV0 + 3 * nu;
  const T* dataV2 = dataV0 + 6 * nu;
  const T* dataV3 = dataV0 + 9 * nu;

  T dataV[12];
  for (int i = 0; i < 12; i++) {
    dataV[i] = gridV.getSpline(knotV, dataV0[i], dataV1[i], dataV2[i], dataV3[i], v);
  }

  T* dataU0 = dataV + 0;
  T* dataU1 = dataV + 3;
  T* dataU2 = dataV + 6;
  T* dataU3 = dataV + 9;

  T res[3];
  for (int i = 0; i < 3; i++) {
    res[i] = gridU.getSpline(knotU, dataU0[i], dataU1[i], dataU2[i], dataU3[i], u);
  }
  x = res[0];
  y = res[1];
  z = res[2];
}

GPUdi() void IrregularSpline2D3D::getSplineVec(const float* correctedData, float u, float v, float& x, float& y, float& z) const
{
  // Same as getSpline, but using vectorized calculation.
  // \param correctedData should be at least 128-bit aligned

#if !defined(__CINT__) && !defined(__ROOTCINT__) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_VC) && defined(__cplusplus) && __cplusplus >= 201703L
  const IrregularSpline1D& gridU = getGridU();
  const IrregularSpline1D& gridV = getGridV();
  int nu = gridU.getNumberOfKnots();
  int iu = gridU.getKnotIndex(u);
  int iv = gridV.getKnotIndex(v);
  const IrregularSpline1D::Knot& knotU = gridU.getKnot(iu);
  const IrregularSpline1D::Knot& knotV = gridV.getKnot(iv);

  const float* dataV0 = correctedData + (nu * (iv - 1) + iu - 1) * 3;
  const float* dataV1 = dataV0 + 3 * nu;
  const float* dataV2 = dataV0 + 6 * nu;
  const float* dataV3 = dataV0 + 9 * nu;

  // calculate F values at V==v and U == Ui of knots

  Vc::SimdArray<float, 12> dataV0vec(dataV0), dataV1vec(dataV1), dataV2vec(dataV2), dataV3vec(dataV3), dataVvec = gridV.getSpline(knotV, dataV0vec, dataV1vec, dataV2vec, dataV3vec, v);

  using V = std::conditional_t<(Vc::float_v::size() >= 4),
                               Vc::float_v,
                               Vc::SimdArray<float, 3>>;

  float dataV[9 + V::size()];
  // dataVvec.scatter( dataV, Vc::SimdArray<float,12>::IndexType::IndexesFromZero() );
  //dataVvec.scatter(dataV, Vc::SimdArray<uint, 12>(Vc::IndexesFromZero));
  dataVvec.store(dataV, Vc::Unaligned);

  for (unsigned int i = 12; i < 9 + V::size(); i++) // fill not used part of the vector with 0
    dataV[i] = 0.f;

  // calculate F values at V==v and U == u
  V dataU0vec(dataV), dataU1vec(dataV + 3), dataU2vec(dataV + 6), dataU3vec(dataV + 9);

  V dataUVvec = gridU.getSpline(knotU, dataU0vec, dataU1vec, dataU2vec, dataU3vec, u);

  x = dataUVvec[0];
  y = dataUVvec[1];
  z = dataUVvec[2];
#else
  getSpline(correctedData, u, v, x, y, z);
#endif
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
