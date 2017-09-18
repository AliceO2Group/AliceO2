// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_BASE_DETMATRIXCACHE_H
#define ALICEO2_BASE_DETMATRIXCACHE_H

#include <vector>
#include <array>
#include "MathUtils/Cartesian3D.h"
#include "DetectorsBase/DetID.h"
#include "Rtypes.h"
#include <FairLogger.h>

class TGeoHMatrix;

namespace o2
{
namespace Base
{

/// MatrixCache is a vector of cached transform matrices (per sensor) for specific Transformation type
 
template <typename T = o2::Base::Transform3D>
class MatrixCache {
  // matrices (per sensor) for specific transformation type

 public:

  MatrixCache() = default;
  ~MatrixCache() = default;
  MatrixCache(const MatrixCache &src) = delete;
  MatrixCache& operator=(const MatrixCache& src) = delete;
 
  /// set the size of the cache
  void setSize(int s)  {if (!mCache.size()) mCache.resize(s);}

  /// get the size of the cache
  int getSize() const {return mCache.size();}

  /// assign matrix to a slot
  void setMatrix(const T& mat, int sensID) {
    // assign matrix for given sensor. The cache must be booked in advance 
    if (sensID>=mCache.size()) {
      LOG(FATAL) << "SensID " << sensID << " exceeds cache size of " << mCache.size() << FairLogger::endl;
    }
    mCache[sensID] = mat;
  }

  const T& getMatrix(int sensID)  const {return mCache[sensID];}

  bool isFilled() const {return !mCache.empty();}
  
 private:
  std::vector<T> mCache;
   ClassDefNV(MatrixCache,1);
};


/// Set of MatrixCache vectors for transformations used by detector, to be overriden by
/// detector class (see ITS GeometryTGeo)

class DetMatrixCache
{
  
 public:
  typedef o2::Base::Transform3D Mat3D;
  typedef o2::Base::Rotation2D Rot2D;

  DetMatrixCache() = default;
  DetMatrixCache(const o2::Base::DetID &id) : mDetID(id) {}

  /// this may serve as a base class for detector interface to geometry, make it virtual
  virtual ~DetMatrixCache() = default;

  DetMatrixCache(const DetMatrixCache& src) = delete;
  DetMatrixCache& operator=(const DetMatrixCache& geom) = delete;
  
  const o2::Base::DetID& getDetID() const { return mDetID;}

  const char* getName() const { return mDetID.getName(); }

  const MatrixCache<Mat3D> & getCacheT2L() const {return mT2L;}
  const MatrixCache<Mat3D> & getCacheT2G() const {return mT2G;}
  const MatrixCache<Mat3D> & getCacheL2G() const {return mL2G;}
  const MatrixCache<Rot2D> & getCacheT2GRot() const {return mT2GRot;}
 
  const Mat3D& getMatrixT2L(int sensID) const {return mT2L.getMatrix(sensID);}
  const Mat3D& getMatrixT2G(int sensID) const {return mT2G.getMatrix(sensID);}
  const Mat3D& getMatrixL2G(int sensID) const {return mL2G.getMatrix(sensID);}
  const Rot2D& getMatrixT2GRot(int sensID) const {return mT2GRot.getMatrix(sensID);}
 
  bool isBuilt() const {return mSize!=0;}
  int  getSize() const {return mSize;}

  //  protected:

  // detector derived class must define its implementation for the method to populate the matrix cache, as an
  // example, see ITS implementation GeometryTGeo.
  // The method can be called multiple times to init caches for different transformations, i.e.
  // with differen mask as  o2::Base::Utils::bit2Mask(T2L), or bit2Mask(L2G,T2L), but for the consistency
  // check the nsens must be always the same.
  virtual void fillMatrixCache(int mask) = 0;
  
  // before calling fillMatrixCache, detector implementation should set the size of the matrix cache
  void setSize(int s);

  // acces to non-const caches for filling in the base classes only
  MatrixCache<Mat3D> & getCacheT2L()  {return mT2L;}
  MatrixCache<Mat3D> & getCacheT2G()  {return mT2G;}
  MatrixCache<Mat3D> & getCacheL2G()  {return mL2G;}
  MatrixCache<Rot2D> & getCacheT2GRot()  {return mT2GRot;}

  o2::Base::DetID mDetID;                      ///< detector ID
  int mSize=0;                                 ///< prebooked number of sensors

  MatrixCache<Mat3D> mL2G;                     ///< Local to Global matrices
  MatrixCache<Mat3D> mT2L;                     ///< Tracking to Local matrices
  MatrixCache<Mat3D> mT2G;                     ///< Tracking to Global matrices (general case)
  MatrixCache<Rot2D> mT2GRot;                  ///< Tracking to Global matrices in case of barrel (simple rotation)

  ClassDef(DetMatrixCache,1);
};

}
}
 
#endif
