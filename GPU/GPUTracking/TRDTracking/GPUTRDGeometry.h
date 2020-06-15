// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDGeometry.h
/// \author David Rohr

#ifndef GPUTRDGEOMETRY_H
#define GPUTRDGEOMETRY_H

#include "GPUCommonDef.h"

#ifdef GPUCA_ALIROOT_LIB
#include "AliTRDgeometry.h"
#include "AliTRDpadPlane.h"
#include "AliGeomManager.h"
#include "TGeoMatrix.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

typedef AliTRDpadPlane GPUTRDpadPlane;

class GPUTRDGeometry : public AliTRDgeometry
{
 public:
  static bool CheckGeometryAvailable() { return AliGeomManager::GetGeometry(); }

  // Make sub-functionality available directly in GPUTRDGeometry
  double GetPadPlaneWidthIPad(int det) const { return GetPadPlane(det)->GetWidthIPad(); }
  double GetPadPlaneRowPos(int layer, int stack, int row) const { return GetPadPlane(layer, stack)->GetRowPos(row); }
  double GetPadPlaneRowSize(int layer, int stack, int row) const { return GetPadPlane(layer, stack)->GetRowSize(row); }
  int GetGeomManagerVolUID(int det, int modId) const { return AliGeomManager::LayerToVolUID(AliGeomManager::ELayerID(AliGeomManager::kTRD1 + GetLayer(det)), modId); }
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#elif defined(HAVE_O2HEADERS) //&& defined(GPUCA_GPUCODE)

class TObjArray;
#include "GPUDef.h"
#include "TRDBase/TRDGeometryFlat.h"
#include "TRDBase/TRDPadPlane.h"
#include "GPUCommonTransform3D.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTRDpadPlane : private o2::trd::TRDPadPlane
{
 public:
  GPUd() float GetTiltingAngle() const { return getTiltingAngle(); }
  GPUd() float GetRowSize(int row) const { return getRowSize(row); }
  GPUd() float GetRow0() const { return getRow0(); }
  GPUd() float GetRowEnd() const { return getRowEnd(); }
  GPUd() float GetColEnd() const { return getColEnd(); }
  GPUd() float GetRowPos(int row) const { return getRowPos(row); }
  GPUd() float GetColPos(int col) const { return getColPos(col); }
  GPUd() float GetNrows() const { return getNrows(); }
};

class GPUTRDGeometry : private o2::trd::TRDGeometryFlat
{
 public:
  GPUd() static bool CheckGeometryAvailable() { return true; }

  // Make sub-functionality available directly in GPUTRDGeometry
  GPUd() float GetPadPlaneWidthIPad(int det) const { return getPadPlane(det)->getWidthIPad(); }
  GPUd() float GetPadPlaneRowPos(int layer, int stack, int row) const { return getPadPlane(layer, stack)->getRowPos(row); }
  GPUd() float GetPadPlaneRowSize(int layer, int stack, int row) const { return getPadPlane(layer, stack)->getRowSize(row); }
  GPUd() int GetGeomManagerVolUID(int det, int modId) const { return 0; }

  // Base functionality of TRDGeometry
  GPUd() float GetTime0(int layer) const { return getTime0(layer); }
  GPUd() float GetCol0(int layer) const { return getCol0(layer); }
  GPUd() int GetLayer(int det) const { return getLayer(det); }
  GPUd() bool CreateClusterMatrixArray() const { return false; }
  GPUd() float AnodePos() const { return anodePos(); }
  GPUd() const Transform3D* GetClusterMatrix(int det) const { return getMatrixT2L(det); }
  GPUd() int GetDetector(int layer, int stack, int sector) const { return getDetector(layer, stack, sector); }
  GPUd() const GPUTRDpadPlane* GetPadPlane(int layer, int stack) const { return (GPUTRDpadPlane*)getPadPlane(layer, stack); }
  GPUd() const GPUTRDpadPlane* GetPadPlane(int detector) const { return (GPUTRDpadPlane*)getPadPlane(detector); }
  GPUd() int GetSector(int det) const { return getSector(det); }
  GPUd() int GetStack(int det) const { return getStack(det); }
  GPUd() int GetStack(float z, int layer) const { return getStack(z, layer); }
  GPUd() float GetAlpha() const { return getAlpha(); }
  GPUd() bool IsHole(int la, int st, int se) const { return isHole(la, st, se); }
  GPUd() int GetRowMax(int layer, int stack, int sector) const { return getRowMax(layer, stack, sector); }
  GPUd() bool ChamberInGeometry(int det) const { return chamberInGeometry(det); }

  static constexpr int kNstack = o2::trd::kNstack;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#else // below are dummy definitions to enable building the standalone version with AliRoot

#include "GPUDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class TGeoHMatrix
{
 public:
  template <class T>
  GPUd() void LocalToMaster(T*, T*) const
  {
  }
};

class GPUTRDpadPlane
{
 public:
  GPUd() float GetTiltingAngle() const { return 0; }
  GPUd() float GetRowSize(int row) const { return 0; }
  GPUd() float GetRowPos(int row) const { return 0; }
  GPUd() float GetRow0() const { return 0; }
  GPUd() float GetRowEnd() const { return 0; }
  GPUd() float GetCol0() const { return 0; }
  GPUd() float GetColEnd() const { return 0; }
  GPUd() float GetColPos(int col) const { return 0; }
  GPUd() float GetNrows() const { return 0; }
};

class GPUTRDGeometry
{
 public:
  GPUd() static bool CheckGeometryAvailable() { return false; }
  void clearInternalBufferUniquePtr() const {}

  // Make sub-functionality available directly in GPUTRDGeometry
  GPUd() float GetPadPlaneWidthIPad(int det) const { return 0; }
  GPUd() float GetPadPlaneRowPos(int layer, int stack, int row) const { return 0; }
  GPUd() float GetPadPlaneRowSize(int layer, int stack, int row) const { return 0; }
  GPUd() int GetGeomManagerVolUID(int det, int modId) const { return 0; }

  // Base functionality of TRDGeometry
  GPUd() float GetTime0(int layer) const { return 0; }
  GPUd() float GetCol0(int layer) const { return 0; }
  GPUd() int GetLayer(int det) const { return 0; }
  GPUd() bool CreateClusterMatrixArray() const { return false; }
  GPUd() float AnodePos() const { return 0; }
  GPUd() const TGeoHMatrix* GetClusterMatrix(int det) const { return nullptr; }
  GPUd() int GetDetector(int layer, int stack, int sector) const { return 0; }
  GPUd() const GPUTRDpadPlane* GetPadPlane(int layer, int stack) const { return nullptr; }
  GPUd() const GPUTRDpadPlane* GetPadPlane(int detector) const { return nullptr; }
  GPUd() int GetSector(int det) const { return 0; }
  GPUd() int GetStack(int det) const { return 0; }
  GPUd() int GetStack(float z, int layer) const { return 0; }
  GPUd() float GetAlpha() const { return 0; }
  GPUd() bool IsHole(int la, int st, int se) const { return false; }
  GPUd() int GetRowMax(int layer, int stack, int /* sector */) const { return 0; }
  GPUd() bool ChamberInGeometry(int det) const { return false; }

  static CONSTEXPR int kNstack = 0;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // !GPUCA_ALIROOT_LIB && !defined(HAVE_O2HEADERS)

#endif // GPUTRDGEOMETRY_H
