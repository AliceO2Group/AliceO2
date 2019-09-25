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
/// \brief collects ITS and MFT common methods to acces matrices

/// \author ruben.shahoyan@cern.ch
#ifndef ALICEO2_ITSMFT_GEOMETRYTGEO_H_
#define ALICEO2_ITSMFT_GEOMETRYTGEO_H_

#include <TObjArray.h> // for TObjArray
#include <string>
#include "DetectorsCommonDataFormats/DetMatrixCache.h"

namespace o2
{
namespace itsmft
{
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  using o2::detectors::DetMatrixCache::fillMatrixCache;
  GeometryTGeo() = default;
  GeometryTGeo(const GeometryTGeo& src) = delete;
  GeometryTGeo& operator=(const GeometryTGeo& geom) = delete;

  GeometryTGeo(const o2::detectors::DetID& detid) : o2::detectors::DetMatrixCache(detid) {}
  ~GeometryTGeo() override = default;

  Int_t getNumberOfChips() const { return mSize; }
  /// build detector layout data, must be overriden by detector
  virtual void Build(int loadTrans) = 0;

 protected:
  ClassDefOverride(GeometryTGeo, 1); // ITSMFR geometry based on TGeo
};
} // namespace itsmft
} // namespace o2
#endif
