// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  SemiregularSpline2D3D.h
/// \brief Definition of SemiregularSpline2D3D class
///
/// \author  Felix Lapp
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SEMIREGULARSPLINE2D3D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SEMIREGULARSPLINE2D3D_H

#include "GPUCommonDef.h"

#include "RegularSpline1D.h"
#include "FlatObject.h"

#if !defined(__CINT__) && !defined(__ROOTCINT__) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_VC)
#include <Vc/Vc>
#include <Vc/SimdArray>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The SemiregularSpline2D3D class represents twoo-dimensional spline interpolation on a semi-unifom grid.
///
/// The class is flat C structure. No virtual methods, no ROOT types are used.
/// It is designed for spline parameterisation of TPC transformation.
///
/// ---
/// The spline interpolates a generic function F:[u,v)->(x,y,z),
/// where u,v belong to [0,1]x[0,1]
///
/// It is a variation of IrregularSpline2D3D class, see IrregularSpline2D3D.h for more details.
///
/// Important:
///   -- The number of knots may change during initialisation
///   -- Don't forget to call correctEdges() for the array of F values (see IrregularSpline2D3D.h )
///
///
class SemiregularSpline2D3D : public FlatObject
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor. Creates an empty uninitialised object
  SemiregularSpline2D3D();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject() instead
  SemiregularSpline2D3D(const SemiregularSpline2D3D&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject() instead
  SemiregularSpline2D3D& operator=(const SemiregularSpline2D3D&) CON_DELETE;

  /// Destructor
  ~SemiregularSpline2D3D() CON_DEFAULT;

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Memory alignment

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

  /// Construction interface

  void cloneFromObject(const SemiregularSpline2D3D& obj, char* newFlatBufferPtr);
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
  void construct(const int numberOfRows, const int numbersOfKnots[]);

  /// _______________  Main functionality   ________________________

  /// Correction of data values at edge knots.
  ///
  /// It is needed for the fast spline mathematics to work correctly. See explanation in IrregularSpline1D.h header
  ///
  /// \param data array of function values. It has the size of getNumberOfKnots()
  template <typename T>
  void correctEdges(T* data) const;

  /// Get interpolated value for f(u,v) using data array correctedData[getNumberOfKnots()] with corrected edges
  template <typename T>
  void getSpline(const T* correctedData, float u, float v, T& x, T& y, T& z) const;

  /// Same as getSpline, but using vectorized calculation.
  /// \param correctedData should be at least 128-bit aligned
  void getSplineVec(const float* correctedData, float u, float v, float& x, float& y, float& z) const;

  /// Get number total of knots: UxV
  int getNumberOfKnots() const { return mNumberOfKnots; }

  /// Get number of rows. Always the same as gridV's number of knots
  int getNumberOfRows() const { return mNumberOfRows; }

  /// Get 1-D grid for U coordinate
  const RegularSpline1D& getGridV() const { return mGridV; }

  /// Get 1-D grid for V coordinate
  //const RegularSpline1D& getGridV() const { return mGridV; }
  const RegularSpline1D& getGridU(const int i) const { return getSplineArray()[i]; }

  /// Get u,v of i-th knot
  void getKnotUV(int iKnot, float& u, float& v) const;

  /// Get size of the mFlatBuffer data
  size_t getFlatBufferSize() const { return mFlatBufferSize; }

  ///Gets the knot index which is the i-th knot in v-space and the j-th knot in u-space
  int getDataIndex(int i, int j) const;
  int getDataIndex0(int i, int j) const;

  /// Gets the offset for the data index map inside the flat buffer
  int getDataIndexMapOffset() const { return mDataIndexMapOffset; }

  /// Get pointer to the flat buffer
  const char* getFlatBufferPtr() const { return mFlatBufferPtr; }

  /// Get minimal required alignment for the class
  static constexpr size_t getClassAlignmentBytes() { return 8; }

  /// Get minimal required alignment for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return 8; }

  /// Get minimal required alignment for the spline data
  static constexpr size_t getDataAlignmentBytes() { return 8; }

  // Gets the spline array for u-coordinates
  const RegularSpline1D* getSplineArray() const
  {
    return reinterpret_cast<const RegularSpline1D*>(mFlatBufferPtr);
  }

  const int* getDataIndexMap() const
  {
    return reinterpret_cast<const int*>(mFlatBufferPtr + mDataIndexMapOffset);
  }

 private:
  void relocateBufferPointers(const char* oldBuffer, char* newBuffer);

  RegularSpline1D* getSplineArrayNonConst()
  {
    return reinterpret_cast<RegularSpline1D*>(mFlatBufferPtr);
  }

  int* getDataIndexMapNonConst()
  {
    return reinterpret_cast<int*>(mFlatBufferPtr + mDataIndexMapOffset);
  }

  ///
  /// ====  Data members   ====
  ///

  RegularSpline1D mGridV; ///< grid for V axis
  int mNumberOfRows;
  int mNumberOfKnots;
  int mDataIndexMapOffset;
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

inline int SemiregularSpline2D3D::getDataIndex(int u, int v) const
{
  return (getDataIndexMap()[v] + u) * 3;
}

inline int SemiregularSpline2D3D::getDataIndex0(int u, int v) const
{
  return (getDataIndexMap()[v] + u);
}

inline void SemiregularSpline2D3D::getKnotUV(int iKnot, float& u, float& v) const
{

  // iterate through all RegularSpline1D's
  for (int i = 0; i < mNumberOfRows; i++) {
    const RegularSpline1D& gridU = getGridU(i);
    const int nk = gridU.getNumberOfKnots();

    // if the searched index is less or equal as the number of knots in the current spline
    // the searched u-v-coordinates have to be in this spline.
    if (iKnot <= nk - 1) {

      //in that case v is the current index
      v = mGridV.knotIndexToU(i);

      //and u the coordinate of the given index
      u = gridU.knotIndexToU(iKnot);
      break;
    }

    //if iKnot is greater than number of knots the searched u-v cannot be in the current gridU
    //so we search for nk less indizes and continue with the next v-coordinate
    iKnot -= nk;
  }
}

template <typename T>
inline void SemiregularSpline2D3D::correctEdges(T* data) const
{
  //Regular v-Grid (vertical)
  const RegularSpline1D& gridV = getGridV();

  int nv = mNumberOfRows;

  //EIGENTLICH V VOR U!!!
  //Wegen Splines aber U vor V

  { // ==== left edge of U ====
    //loop through all gridUs
    for (int iv = 1; iv < mNumberOfRows - 1; iv++) {
      T* f0 = data + getDataIndex(0, iv);
      T* f1 = f0 + 3;
      T* f2 = f0 + 6;
      T* f3 = f0 + 9;
      for (int idim = 0; idim < 3; idim++) {
        f0[idim] = (T)(0.5 * f0[idim] + 1.5 * f1[idim] - 1.5 * f2[idim] + 0.5 * f3[idim]);
      }
    }
  }

  { // ==== right edge of U ====
    //loop through all gridUs
    for (int iv = 1; iv < mNumberOfRows - 1; iv++) {
      const RegularSpline1D& gridU = getGridU(iv);
      int nu = gridU.getNumberOfKnots();
      T* f0 = data + getDataIndex(nu - 4, iv);
      T* f1 = f0 + 3;
      T* f2 = f0 + 6;
      T* f3 = f0 + 9;
      for (int idim = 0; idim < 3; idim++) {
        f3[idim] = (T)(0.5 * f0[idim] - 1.5 * f1[idim] + 1.5 * f2[idim] + 0.5 * f3[idim]);
      }
    }
  }

  { // ==== low edge of V ====
    const RegularSpline1D& gridU = getGridU(0);
    int nu = gridU.getNumberOfKnots();

    for (int iu = 0; iu < nu; iu++) {
      //f0 to f3 are the x,y,z values of 4 points in the grid along the v axis.
      //Since there are no knots because of the irregularity you can get this by using the getSplineMethod.
      T* f0 = data + getDataIndex(iu, 0);
      float u = gridU.knotIndexToU(iu);

      float x1 = 0, y1 = 0, z1 = 0, x2 = 0, y2 = 0, z2 = 0, x3 = 0, y3 = 0, z3 = 0;
      getSpline(data, u, gridV.knotIndexToU(1), x1, y1, z1);
      getSpline(data, u, gridV.knotIndexToU(2), x2, y2, z2);
      getSpline(data, u, gridV.knotIndexToU(3), x3, y3, z3);

      T f1[3] = { x1, y1, z1 };
      T f2[3] = { x2, y2, z2 };
      T f3[3] = { x3, y3, z3 };
      for (int idim = 0; idim < 3; idim++) {
        f0[idim] = (T)(0.5 * f0[idim] + 1.5 * f1[idim] - 1.5 * f2[idim] + 0.5 * f3[idim]);
      }
    }
  }

  { // ==== high edge of V ====
    const RegularSpline1D& gridU = getGridU(nv - 4);
    int nu = getGridU(nv - 1).getNumberOfKnots();

    for (int iu = 0; iu < nu; iu++) {

      float u = getGridU(nv - 1).knotIndexToU(iu);

      float x1 = 0, y1 = 0, z1 = 0, x2 = 0, y2 = 0, z2 = 0, x3 = 0, y3 = 0, z3 = 0;
      getSpline(data, u, gridV.knotIndexToU(nv - 4), x1, y1, z1);
      getSpline(data, u, gridV.knotIndexToU(nv - 3), x2, y2, z2);
      getSpline(data, u, gridV.knotIndexToU(nv - 2), x3, y3, z3);

      T f0[3] = { x1, y1, z1 };
      T f1[3] = { x2, y2, z2 };
      T f2[3] = { x3, y3, z3 };
      T* f3 = data + getDataIndex(iu, nv - 1);
      for (int idim = 0; idim < 3; idim++) {
        f3[idim] = (T)(0.5 * f0[idim] - 1.5 * f1[idim] + 1.5 * f2[idim] + 0.5 * f3[idim]);
      }
    }
  }

  // =============== CORRECT CORNERS ==============

  { // === Lower left corner with u-direction ===
    T* f0 = data;
    T* f1 = f0 + 3;
    T* f2 = f0 + 6;
    T* f3 = f0 + 9;
    for (int idim = 0; idim < 3; idim++) {
      f0[idim] = (T)(0.5 * f0[idim] + 1.5 * f1[idim] - 1.5 * f2[idim] + 0.5 * f3[idim]);
    }
  }

  { // ==== Lower right corner with u-direction ===

    const RegularSpline1D& gridU = getGridU(0);
    int nu = gridU.getNumberOfKnots();
    T* f0 = data + getDataIndex(nu - 4, 0);
    T* f1 = f0 + 3;
    T* f2 = f0 + 6;
    T* f3 = f0 + 9;
    for (int idim = 0; idim < 3; idim++) {
      f3[idim] = (T)(0.5 * f0[idim] - 1.5 * f1[idim] + 1.5 * f2[idim] + 0.5 * f3[idim]);
    }
  }

  { // === upper left corner with u-direction ===
    T* f0 = data + getDataIndex(0, nv - 1);
    T* f1 = f0 + 3;
    T* f2 = f0 + 6;
    T* f3 = f0 + 9;
    for (int idim = 0; idim < 3; idim++) {
      f0[idim] = (T)(0.5 * f0[idim] + 1.5 * f1[idim] - 1.5 * f2[idim] + 0.5 * f3[idim]);
    }
  }

  { // ==== upper right corner with u-direction ===
    const RegularSpline1D& gridU = getGridU(nv - 1);
    int nu = gridU.getNumberOfKnots();
    T* f0 = data + getDataIndex(nu - 4, nv - 1);
    T* f1 = f0 + 3;
    T* f2 = f0 + 6;
    T* f3 = f0 + 9;
    for (int idim = 0; idim < 3; idim++) {
      f3[idim] = (T)(0.5 * f0[idim] - 1.5 * f1[idim] + 1.5 * f2[idim] + 0.5 * f3[idim]);
    }
  }
}

template <typename T>
inline void SemiregularSpline2D3D::getSpline(const T* correctedData, float u, float v, T& x, T& y, T& z) const
{
  // Get interpolated value for f(u,v) using data array correctedData[getNumberOfKnots()] with corrected edges

  // find the v indizes of the u-splines that are needed.
  int iknotv = mGridV.getKnotIndex(v);

  // to save the index positions of u-coordinates we create an array
  T dataVx[12];
  //int dataOffset0 = getDataIndex0(0, iknotv-1); //index of the very left point in the vi-1-th gridU

  // we loop through the 4 needed u-Splines
  int vxIndex = 0;
  for (int vi = 0; vi < 4; vi++, vxIndex += 3) {

    const int vDelta = iknotv + vi - 1;
    const RegularSpline1D& gridU = getGridU(vDelta);
    // and find at which index in that specific spline the u-coordinate must lay.

    const int ui = gridU.getKnotIndex(u);
    const int dataOffset = getDataIndex(ui - 1, vDelta); //(dataOffset0 + (ui-1))*3;

    dataVx[vxIndex + 0] = gridU.getSpline(ui, correctedData[dataOffset], correctedData[dataOffset + 3], correctedData[dataOffset + 6], correctedData[dataOffset + 9], u);
    dataVx[vxIndex + 1] = gridU.getSpline(ui, correctedData[dataOffset + 1], correctedData[dataOffset + 4], correctedData[dataOffset + 7], correctedData[dataOffset + 10], u);
    dataVx[vxIndex + 2] = gridU.getSpline(ui, correctedData[dataOffset + 2], correctedData[dataOffset + 5], correctedData[dataOffset + 8], correctedData[dataOffset + 11], u);
  }

  //return results
  x = mGridV.getSpline(iknotv, dataVx[0], dataVx[3], dataVx[6], dataVx[9], v);
  y = mGridV.getSpline(iknotv, dataVx[1], dataVx[4], dataVx[7], dataVx[10], v);
  z = mGridV.getSpline(iknotv, dataVx[2], dataVx[5], dataVx[8], dataVx[11], v);
}

inline void SemiregularSpline2D3D::getSplineVec(const float* correctedData, float u, float v, float& x, float& y, float& z) const
{
  // Same as getSpline, but using vectorized calculation.
  // \param correctedData should be at least 128-bit aligned

#if !defined(__CINT__) && !defined(__ROOTCINT__) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_VC)
  //&& !defined(__CLING__)
  /*
	Idea: There are 16 knots important for (u, v).
	a,b,c,d := Knots in first u-grid
	e,f,g,h := Knots in second u-grid
	i,j,k,l := Knots in third u-grid
	m,n,o,p := Knots in fourth u-grid.
        It could be possible to calculate the spline in 3 dimentions for a,b,c,d at the same time as e,f,g,h etc.
	3 of the 4 parallel threads of the vector would calculate x,y,z for one row and the last task already calculates x for the next one.
	=> 4x faster

	Problem:To do this, we need vectors where every i-th element is used for the calculation. So what we need is:
	[a,e,i,m]
	[b,f,j,n]
	[c,g,k,o]
	[d,h,l,p]
	This is barely possible to do with a good performance because e.g. a,e,i,m do not lay beside each other in data.
	
	Work around 1:
	Don't calculate knots parrallel but the dimensions. But you can only be 3x faster this way because the 4th thread would be the x-dimension of the next point.

	Work around 2:
	Try to create a matrix as it was mentioned earlier ([a,e,i,m][b,f,..]...) by copying data.
	This may be less efficient than Work around 1 but needs to be measured.
	
      */

  //workaround 1:
  int vGridi = mGridV.getKnotIndex(v);

  float dataU[12];
  int vOffset = 0;
  for (int vi = 0; vi < 4; vi++, vOffset += 3) {
    const RegularSpline1D& gridU = getGridU(vi + vGridi - 1);

    // and find at which index in that specific spline the u-coordinate must lay.
    int ui = gridU.getKnotIndex(u);

    // using getDataIndex we know at which position knot (ui, vi) is saved.
    // dataU0 to U3 are 4 points along the u-spline surrounding the u coordinate.
    const float* dataU0 = correctedData + getDataIndex(ui - 1, vGridi + vi - 1);

    Vc::float_v dt0(dataU0 + 0);
    Vc::float_v dt1(dataU0 + 3);
    Vc::float_v dt2(dataU0 + 6);
    Vc::float_v dt3(dataU0 + 9);

    Vc::float_v resU = gridU.getSpline(ui, dt0, dt1, dt2, dt3, u);

    // save the results in dataVx array
    dataU[vOffset + 0] = resU[0];
    dataU[vOffset + 1] = resU[1];
    dataU[vOffset + 2] = resU[2];
  }

  Vc::float_v dataV0(dataU + 0);
  Vc::float_v dataV1(dataU + 3);
  Vc::float_v dataV2(dataU + 6);
  Vc::float_v dataV3(dataU + 9);

  Vc::float_v res = mGridV.getSpline(vGridi, dataV0, dataV1, dataV2, dataV3, v);
  x = res[0];
  y = res[1];
  z = res[2];

//getSpline( correctedData, u, v, x, y, z );
#else
  getSpline(correctedData, u, v, x, y, z);
#endif
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
