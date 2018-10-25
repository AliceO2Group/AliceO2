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
#include "TRDBase/TRDGeometry.h"

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

	static const int kNstack = 0;
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

	static const int kNstack = 0;
};

#endif

#endif
