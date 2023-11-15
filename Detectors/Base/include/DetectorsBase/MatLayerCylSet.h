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

/// \file MatLayerCylSet.h
/// \brief Declarations for the wrapper for the set of cylindrical material layers

#ifndef ALICEO2_MATLAYERCYLSET_H
#define ALICEO2_MATLAYERCYLSET_H

#include "GPUCommonDef.h"
#include "DetectorsBase/MatLayerCyl.h"
#include "DetectorsBase/Ray.h"
#include "FlatObject.h"

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
#include "MathUtils/Cartesian.h"
#endif // !GPUCA_ALIGPUCODE

/**********************************************************************
 *                                                                    *
 * Set of cylindrical material layer                                  *
 *                                                                    *
 **********************************************************************/
namespace o2
{
namespace base
{

struct MatLayerCylSetLayout {
  float mRMin;          ///< min radius
  float mRMax;          ///< max radius
  float mZMax;          ///< max Z span
  float mRMin2;         ///< precalculater rmin^2
  float mRMax2;         ///< precalculater rmax^2
  int mNLayers;         ///< number of layers
  int mNRIntervals;     ///< number of R interval boundaries (gaps are possible)
  MatLayerCyl* mLayers; //[mNLayers] set of cylinrical layers
  float* mR2Intervals;  //[mNRIntervals+1] limits of layers
  int* mInterval2LrID;  //[mNRIntervals] mapping from r2 interval to layer ID
};

class MatLayerCylSet : public o2::gpu::FlatObject
{

 public:
  MatLayerCylSet() CON_DEFAULT;
  ~MatLayerCylSet() CON_DEFAULT;
  MatLayerCylSet(const MatLayerCylSet& src) CON_DELETE;

  GPUd() const MatLayerCylSetLayout* get() const { return reinterpret_cast<const MatLayerCylSetLayout*>(mFlatBufferPtr); }
  GPUd() MatLayerCylSetLayout* get() { return reinterpret_cast<MatLayerCylSetLayout*>(mFlatBufferPtr); }

  GPUd() int getNLayers() const { return get() ? get()->mNLayers : 0; }
  GPUd() const MatLayerCyl& getLayer(int i) const { return get()->mLayers[i]; }

  GPUd() bool getLayersRange(const Ray& ray, short& lmin, short& lmax) const;
  GPUd() float getRMin() const { return get()->mRMin; }
  GPUd() float getRMax() const { return get()->mRMax; }
  GPUd() float getZMax() const { return get()->mZMax; }
  GPUd() float getRMin2() const { return get()->mRMin2; }
  GPUd() float getRMax2() const { return get()->mRMax2; }

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
  void print(bool data = false) const;
  void addLayer(float rmin, float rmax, float zmax, float dz, float drphi);
  void populateFromTGeo(int ntrPerCel = 10);
  void optimizePhiSlices(float maxRelDiff = 0.05);

  void dumpToTree(const std::string& outName = "matbudTree.root") const;
  void writeToFile(const std::string& outFName = "matbud.root");
  static MatLayerCylSet* loadFromFile(const std::string& inpFName = "matbud.root");
  static MatLayerCylSet* rectifyPtrFromFile(MatLayerCylSet* ptr);

  // initializes internal voxel lookup
  void initLayerVoxelLU();

  void flatten();

  MatLayerCyl& getLayer(int i) { return get()->mLayers[i]; }
  MatLayerCylSet* extractCopy(float rmin, float rmax, float tol = 1e-3) const;
  void finalizeStructures();

#endif // !GPUCA_ALIGPUCODE

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version
  MatBudget getMatBudget(const math_utils::Point3D<float>& point0, const math_utils::Point3D<float>& point1) const
  {
    // get material budget traversed on the line between point0 and point1
    return getMatBudget(point0.X(), point0.Y(), point0.Z(), point1.X(), point1.Y(), point1.Z());
  }
#endif // !GPUCA_ALIGPUCODE
  GPUd() MatBudget getMatBudget(float x0, float y0, float z0, float x1, float y1, float z1) const;

  GPUd() int searchSegment(float val, int low = -1, int high = -1) const;

  /// searches a layer based on r2 input, using a lookup table
  GPUd() int searchLayerFast(float r2, int low = -1, int high = -1) const;

#ifndef GPUCA_GPUCODE
  //-----------------------------------------------------------
  std::size_t estimateFlatBufferSize() const;
  void moveBufferTo(char* newFlatBufferPtr);

  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);
  void cloneFromObject(const MatLayerCylSet& obj, char* newFlatBufferPtr);
  using o2::gpu::FlatObject::adoptInternalBuffer;
  using o2::gpu::FlatObject::releaseInternalBuffer;

  void fixPointers(char* newPtr = nullptr);
  void fixPointers(char* oldPtr, char* newPtr, bool newPtrValid = true);

  /// Gives minimal alignment in bytes required for the class object
  static constexpr size_t getClassAlignmentBytes() { return 8; }
  /// Gives minimal alignment in bytes required for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return 8; }
#endif // !GPUCA_GPUCODE

  static constexpr float LayerRMax = 500;    // maximum value of R lookup (corresponds to last layer of MatLUT)
  static constexpr float VoxelRDelta = 0.05; // voxel spacing for layer lookup; seems a natural choice - corresponding ~ to smallest spacing
  static constexpr float InvVoxelRDelta = 1.f / VoxelRDelta;
  static constexpr int NumVoxels = int(LayerRMax / VoxelRDelta);

  uint16_t mLayerVoxelLU[2 * NumVoxels]; //! helper structure to lookup a layer based on known radius (static dimension for easy copy to GPU)
  bool mInitializedLayerVoxelLU = false; //! if the voxels have been initialized

  ClassDefNV(MatLayerCylSet, 1);
};

} // namespace base
} // namespace o2

#endif
