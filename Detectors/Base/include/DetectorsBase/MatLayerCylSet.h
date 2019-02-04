// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MatLayerCylSet.h
/// \brief Declarations for the wrapper for the set of cylindrical material layers

#ifndef ALICEO2_MATLAYERCYLSET_H
#define ALICEO2_MATLAYERCYLSET_H

#include "DetectorsBase/MatLayerCyl.h"
#include "DetectorsBase/Ray.h"
#include "DetectorsBase/flatObject.h"

#ifndef GPUCA_GPUCODE // this part is unvisible on GPU version
#include "MathUtils/Cartesian3D.h"
#endif // !GPUCA_GPUCODE

/**********************************************************************
 *                                                                    *
 * Set of cylindrical material layer                                  *
 *                                                                    *
 **********************************************************************/
namespace o2
{
namespace Base
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

class MatLayerCylSet : public flatObject
{

 public:
  MatLayerCylSet() CON_DEFAULT;
  ~MatLayerCylSet() CON_DEFAULT;
  MatLayerCylSet(const MatLayerCylSet& src) CON_DELETE;

  const MatLayerCylSetLayout* get() const { return reinterpret_cast<const MatLayerCylSetLayout*>(mFlatBufferPtr); }
  MatLayerCylSetLayout* get() { return reinterpret_cast<MatLayerCylSetLayout*>(mFlatBufferPtr); }

  int getNLayers() const { return get() ? get()->mNLayers : 0; }
  const MatLayerCyl& getLayer(int i) const { return get()->mLayers[i]; }

  bool getLayersRange(const Ray& ray, short& lmin, short& lmax) const;
  float getRMin() const { return get()->mRMin; }
  float getRMax() const { return get()->mRMax; }
  float getZMax() const { return get()->mZMax; }
  float getRMin2() const { return get()->mRMin2; }
  float getRMax2() const { return get()->mRMax2; }

#ifndef GPUCA_GPUCODE // this part is unvisible on GPU version

  void print(bool data = false) const;
  void addLayer(float rmin, float rmax, float zmax, float dz, float drphi);
  void populateFromTGeo(int ntrPerCel = 10);
  void optimizePhiSlices(float maxRelDiff = 0.05);

  void dumpToTree(const std::string outName = "matbudTree.root") const;
  void writeToFile(std::string outFName = "matbud.root", std::string name = "MatBud");
  static MatLayerCylSet* loadFromFile(std::string inpFName = "matbud.root", std::string name = "MatBud");
  void flatten();

#endif // !GPUCA_GPUCODE

  std::size_t estimateFlatBufferSize() const;

#ifndef GPUCA_GPUCODE // this part is unvisible on GPU version
  MatBudget getMatBudget(const Point3D<float>& point0, const Point3D<float>& point1) const
  {
    // get material budget traversed on the line between point0 and point1
    return getMatBudget(point0.X(), point0.Y(), point0.Z(), point1.X(), point1.Y(), point1.Z());
  }
#endif // !GPUCA_GPUCODE
  MatBudget getMatBudget(float x0, float y0, float z0, float x1, float y1, float z1) const;

  int searchSegment(float val, int low = -1, int high = -1) const;

  //-----------------------------------------------------------
  void moveBufferTo(char* newFlatBufferPtr);

  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);
  void cloneFromObject(const MatLayerCylSet& obj, char* newFlatBufferPtr);
  using flatObject::adoptInternalBuffer;
  using flatObject::releaseInternalBuffer;

  void fixPointers(char* newPtr = nullptr);
  void fixPointers(char* oldPtr, char* newPtr);

  /// Gives minimal alignment in bytes required for the class object
  static constexpr size_t getClassAlignmentBytes() { return 8; }
  /// Gives minimal alignment in bytes required for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return 8; }

  ClassDefNV(MatLayerCylSet, 1);
};

} // namespace Base
} // namespace o2

#endif
