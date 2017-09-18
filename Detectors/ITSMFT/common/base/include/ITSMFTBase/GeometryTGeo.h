// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryTGeo.h
/// \brief Definition of the GeometryTGeo class : common part for ITS and MFT
/// \brief collects ITS and MFT common methods to acces matrices and segmentations

/// \author ruben.shahoyan@cern.ch
#ifndef ALICEO2_ITSMFT_GEOMETRYTGEO_H_
#define ALICEO2_ITSMFT_GEOMETRYTGEO_H_

#include "DetectorsBase/DetMatrixCache.h"
#include <TObjArray.h>  // for TObjArray
#include <string>

namespace o2
{
namespace ITSMFT
{
class Segmentation;
 
class GeometryTGeo : public o2::Base::DetMatrixCache
{
 public:

  enum {
    kChipTypePix = 0,
    kNChipTypes,
    kMaxSegmPerChipType = 10
  }; // defined detector chip types (each one can have different segmentations)
  
  using DetMatrixCache::fillMatrixCache;
  GeometryTGeo() = default;
  GeometryTGeo(const GeometryTGeo& src) = delete;
  GeometryTGeo& operator=(const GeometryTGeo& geom) = delete;

 GeometryTGeo(const o2::Base::DetID& detid) : DetMatrixCache(detid) {}

  ~GeometryTGeo() override
    {
      mSegmentations.Delete();
    }

  
  TObjArray* getSegmentations() const { return (TObjArray*)&mSegmentations; }
  const Segmentation* getSegmentationById(Int_t id) const;

  Int_t getNumberOfChips() const { return mSize; }
  
  /// build detector layout data, must be overriden by detector
  virtual void Build(Bool_t loadSegmentations, int loadTrans) = 0;

  const char* getChipTypeName(Int_t i);
  void setChipTypeName(Int_t i, const char* nm);
  
 protected:

  TObjArray mSegmentations;;                                ///< segmentations
  static std::array<std::string,kNChipTypes> sChipTypeName; ///< upg detType Names

  ClassDefOverride(GeometryTGeo, 1); // ITSMFR geometry based on TGeo
};

inline const Segmentation* GeometryTGeo::getSegmentationById(Int_t id) const
{
  /// Get segmentation by ID
  return (Segmentation*)mSegmentations.At(id);
}

inline void GeometryTGeo::setChipTypeName(Int_t i, const char* nm)
{
  /// set the name of the chip type
  if (i >= kNChipTypes) {
    i /= kMaxSegmPerChipType; // full type is provided
  }
  sChipTypeName[i] = nm;
}

inline const char* GeometryTGeo::getChipTypeName(Int_t i)
{
  if (i >= kNChipTypes) {
    i /= kMaxSegmPerChipType; // full type is provided
  }
  return sChipTypeName[i].c_str();
}

 
}
}
#endif
