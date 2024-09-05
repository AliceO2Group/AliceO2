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

#ifndef O2_TRD_GEOMETRYFLAT_H
#define O2_TRD_GEOMETRYFLAT_H

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstring>
#endif
#include "FlatObject.h"
#include "GPUCommonDef.h"
#include "GPUCommonTransform3D.h"
#include "TRDBase/GeometryBase.h"
#include "TRDBase/PadPlane.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{

class Geometry;

//Reduced flat version of TRD geometry class.
//Contains all entries required for tracking on GPUs.
class GeometryFlat : public o2::gpu::FlatObject, public GeometryBase
{
 public:
#ifndef GPUCA_GPUCODE
  GeometryFlat() = default;
  GeometryFlat(const GeometryFlat& v) : FlatObject(), GeometryBase()
  {
    memcpy((void*)this, (void*)&v, sizeof(*this));
  }
  GeometryFlat(const Geometry& geo);
  ~GeometryFlat() = default;
#endif
  GPUd() const o2::gpu::Transform3D* getMatrixT2L(int det) const
  {
    if (mMatrixIndirection[det] == -1) {
      return nullptr;
    }
    return &mMatrixCache[mMatrixIndirection[det]];
    ;
  }

  GPUd() bool chamberInGeometry(int det) const
  {
    return (mMatrixIndirection[det] >= 0);
  }

 private:
  o2::gpu::Transform3D mMatrixCache[constants::NCHAMBER];
  short mMatrixIndirection[constants::MAXCHAMBER];
};

} // namespace trd
} // namespace o2

#endif
