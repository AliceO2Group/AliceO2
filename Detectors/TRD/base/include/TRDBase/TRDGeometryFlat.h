// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDGEOMETRYFLAT_H
#define O2_TRDGEOMETRYFLAT_H

#include <cstring>
#include "FlatObject.h"
#include "AliTPCCommonDefGPU.h"
#include "AliTPCCommonTransform3D.h"
#include "TRDBase/TRDGeometryBase.h"
#include "TRDBase/TRDPadPlane.h"

using namespace o2::trd;

namespace o2
{
namespace trd
{

class TRDGeometry;

//Reduced flat version of TRD geometry class.
//Contains all entries required for tracking on GPUs.
class TRDGeometryFlat : public ali_tpc_common::tpc_fast_transformation::FlatObject, public TRDGeometryBase
{
 public:
  TRDGeometryFlat() = default;
  TRDGeometryFlat(const TRDGeometryFlat& v) : FlatObject(), TRDGeometryBase()
  {
    memcpy((void*)this, (void*)&v, sizeof(*this));
  }
  TRDGeometryFlat(const TRDGeometry& geo);
  ~TRDGeometryFlat() = default;

  GPUd() const ali_tpc_common::Transform3D* getMatrixT2L(int det) const
  {
    if (mMatrixIndirection[det] == -1)
      return nullptr;
    return &mMatrixCache[mMatrixIndirection[det]];
    ;
  }

  GPUd() bool chamberInGeometry(int det) const
  {
    return (mMatrixIndirection[det] >= 0);
  }

 private:
  ali_tpc_common::Transform3D mMatrixCache[MAXMATRICES];
  short mMatrixIndirection[kNdet];
};

} // namespace trd
} // namespace o2

#endif
