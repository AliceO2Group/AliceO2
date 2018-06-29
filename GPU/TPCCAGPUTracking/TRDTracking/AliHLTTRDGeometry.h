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
	double GetPadPlaneWithIPad(int det) {return GetPadPlane(det)->GetWidthIPad();}
	double GetPadPlaneRowPos(int layer, int stack, int row) {return GetPadPlane(layer, stack)->GetRowPos(row);}
	double GetPadPlaneRowSize(int layer, int stack, int row) {return GetPadPlane(layer, stack)->GetRowSize(row);}
	int GetGeomManagerVolUID(int det, int modId) {return AliGeomManager::LayerToVolUID(AliGeomManager::ELayerID(AliGeomManager::kTRD1+GetLayer(det)), modId);}
};

#else

class TGeoHMatrix
{
public:
	template <class T> void LocalToMaster(T*, T*) {}
};

class AliHLTTRDpadPlane
{
public:
	float GetTiltingAngle() {return 0;}
	float GetRowSize(int row) {return 0;}
	float GetRowPos(int row) {return 0;}
	float GetColPos(int col) {return 0;}
	float GetNrows() {return 0;}
};

class AliHLTTRDGeometry
{
public:
	static bool CheckGeometryAvailable() {return false;}

	//Make sub-functionality available directly in AliHLTTRDGeometry
	float GetPadPlaneWithIPad(int det) {return 0;}
	float GetPadPlaneRowPos(int layer, int stack, int row) {return 0;}
	float GetPadPlaneRowSize(int layer, int stack, int row) {return 0;}
	int GetGeomManagerVolUID(int det, int modId) {return 0;}
	
	//Base functionality of TRDGeometry
	float GetTime0(int layer) {return 0;}
	float GetCol0(int layer) {return 0;}
	int GetLayer(int det) {return 0;}
	bool CreateClusterMatrixArray() {return false;}
	float AnodePos() {return 0;}
	TGeoHMatrix* GetClusterMatrix(int det) {return nullptr;}
	int GetDetector(int layer, int stack, int sector) {return 0;}
	AliHLTTRDpadPlane* GetPadPlane(int layer, int stack) {return nullptr;}
	AliHLTTRDpadPlane* GetPadPlane(int detector) {return nullptr;}
	int GetSector(int det) {return 0;}
	int GetStack(int det) {return 0;}
	int GetStack(float z, int layer) {return 0;}
	float GetAlpha() {return 0;}
	bool IsHole(int la, int st, int se) const {return false;}
	int GetRowMax(int layer, int stack, int /*sector*/) {return 0;}
	
	static const int kNstack = 0;
};

#endif

#endif
