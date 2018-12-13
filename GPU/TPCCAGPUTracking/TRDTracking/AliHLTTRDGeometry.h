#ifndef ALIHLTTRDGEOMETRY_H
#define ALIHLTTRDGEOMETRY_H

#ifdef HLTCA_BUILD_ALIROOT_LIB
#include "AliTRDgeometry.h"
#include "AliTRDpadPlane.h"
#include "AliGeomManager.h"
#include "TGeoMatrix.h"

typedef AliTRDpadPlane AliHLTTRDpadPlane;

class AliHLTTRDGeometry : public AliTRDgeometry
{
public:
	static bool CheckGeometryAvailable() {return AliGeomManager::GetGeometry();}

	//Make sub-functionality available directly in AliHLTTRDGeometry
	double GetPadPlaneWidthIPad(int det) {return GetPadPlane(det)->GetWidthIPad();}
	double GetPadPlaneRowPos(int layer, int stack, int row) {return GetPadPlane(layer, stack)->GetRowPos(row);}
	double GetPadPlaneRowSize(int layer, int stack, int row) {return GetPadPlane(layer, stack)->GetRowSize(row);}
	int GetGeomManagerVolUID(int det, int modId) {return AliGeomManager::LayerToVolUID(AliGeomManager::ELayerID(AliGeomManager::kTRD1+GetLayer(det)), modId);}
};

#elif defined(HAVE_O2HEADERS)

class TObjArray;
#include "AliHLTTPCCADef.h"
#include "TRDBase/TRDGeometryFlat.h"
#include "TRDBase/TRDPadPlane.h"
#include "AliTPCCommonTransform3D.h"

class AliHLTTRDpadPlane : private o2::trd::TRDPadPlane
{
public:
	GPUd() float GetTiltingAngle() {return getTiltingAngle();}
	GPUd() float GetRowSize(int row) {return getRowSize(row);}
	GPUd() float GetRow0() {return getRow0();}
	GPUd() float GetRowEnd() {return getRowEnd();}
	GPUd() float GetColEnd() {return getColEnd();}
	GPUd() float GetRowPos(int row) {return getRowPos(row);}
	GPUd() float GetColPos(int col) {return getColPos(col);}
	GPUd() float GetNrows() {return getNrows();}
};

class AliHLTTRDGeometry : private o2::trd::TRDGeometryFlat
{
public:
	GPUd() static bool CheckGeometryAvailable() {return true;}

	//Make sub-functionality available directly in AliHLTTRDGeometry
	GPUd() float GetPadPlaneWidthIPad(int det) {return getPadPlane(det)->getWidthIPad();}
	GPUd() float GetPadPlaneRowPos(int layer, int stack, int row) {return getPadPlane(layer, stack)->getRowPos(row);}
	GPUd() float GetPadPlaneRowSize(int layer, int stack, int row) {return getPadPlane(layer, stack)->getRowSize(row);}
	GPUd() int GetGeomManagerVolUID(int det, int modId) {return 0;}
	
	//Base functionality of TRDGeometry
	GPUd() float GetTime0(int layer) {return getTime0(layer);}
	GPUd() float GetCol0(int layer) {return getCol0(layer);}
	GPUd() int GetLayer(int det) {return getLayer(det);}
	GPUd() bool CreateClusterMatrixArray() {return false;}
	GPUd() float AnodePos() {return anodePos();}
	GPUd() const ali_tpc_common::Transform3D* GetClusterMatrix(int det) {return getMatrixT2L(det);}
	GPUd() int GetDetector(int layer, int stack, int sector) {return getDetector(layer, stack, sector);}
	GPUd() AliHLTTRDpadPlane* GetPadPlane(int layer, int stack) {return (AliHLTTRDpadPlane*) getPadPlane(layer, stack);}
	GPUd() AliHLTTRDpadPlane* GetPadPlane(int detector) {return (AliHLTTRDpadPlane*) getPadPlane(detector);}
	GPUd() int GetSector(int det) {return getSector(det);}
	GPUd() int GetStack(int det) {return getStack(det);}
	GPUd() int GetStack(float z, int layer) {return getStack(z, layer);}
	GPUd() float GetAlpha() {return getAlpha();}
	GPUd() bool IsHole(int la, int st, int se) const {return isHole(la, st, se);}
	GPUd() int GetRowMax(int layer, int stack, int sector) {return getRowMax(layer, stack, sector);}
  GPUd() bool ChamberInGeometry(int det) {return chamberInGeometry(det);}

	static constexpr int kNstack = o2::trd::kNstack;
};

#else

#include "AliHLTTPCCADef.h"

class TGeoHMatrix
{
public:
	template <class T> GPUd() void LocalToMaster(T*, T*) {}
};

class AliHLTTRDpadPlane
{
public:
	GPUd() float GetTiltingAngle() {return 0;}
	GPUd() float GetRowSize(int row) {return 0;}
	GPUd() float GetRowPos(int row) {return 0;}
	GPUd() float GetColPos(int col) {return 0;}
	GPUd() float GetNrows() {return 0;}
};

class AliHLTTRDGeometry
{
public:
	GPUd() static bool CheckGeometryAvailable() {return false;}

	//Make sub-functionality available directly in AliHLTTRDGeometry
	GPUd() float GetPadPlaneWidthIPad(int det) {return 0;}
	GPUd() float GetPadPlaneRowPos(int layer, int stack, int row) {return 0;}
	GPUd() float GetPadPlaneRowSize(int layer, int stack, int row) {return 0;}
	GPUd() int GetGeomManagerVolUID(int det, int modId) {return 0;}
	
	//Base functionality of TRDGeometry
	GPUd() float GetTime0(int layer) {return 0;}
	GPUd() float GetCol0(int layer) {return 0;}
	GPUd() int GetLayer(int det) {return 0;}
	GPUd() bool CreateClusterMatrixArray() {return false;}
	GPUd() float AnodePos() {return 0;}
	GPUd() TGeoHMatrix* GetClusterMatrix(int det) {return nullptr;}
	GPUd() int GetDetector(int layer, int stack, int sector) {return 0;}
	GPUd() AliHLTTRDpadPlane* GetPadPlane(int layer, int stack) {return nullptr;}
	GPUd() AliHLTTRDpadPlane* GetPadPlane(int detector) {return nullptr;}
	GPUd() int GetSector(int det) {return 0;}
	GPUd() int GetStack(int det) {return 0;}
	GPUd() int GetStack(float z, int layer) {return 0;}
	GPUd() float GetAlpha() {return 0;}
	GPUd() bool IsHole(int la, int st, int se) const {return false;}
	GPUd() int GetRowMax(int layer, int stack, int /*sector*/) {return 0;}
  GPUd() bool ChamberInGeometry(int det) {return false;}

	static const int kNstack = 0;
};

#endif

#endif
