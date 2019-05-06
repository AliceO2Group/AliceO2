// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDBase/TRDGeometryFlat.h"
#include "TRDBase/TRDGeometry.h"

TRDGeometryFlat::TRDGeometryFlat(const TRDGeometry& geo)
{
  TRDGeometryBase& b1 = *this;
  const TRDGeometryBase& b2 = geo;
  memcpy((void*)&b1, (void*)&b2, sizeof(b1));
  int matrixCount = 0;
  for (int i = 0; i < kNdet; i++) {
    if (geo.chamberInGeometry(i)) {
      double v[12];
      geo.getMatrixT2L(i).GetComponents(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]);
      float f[12];
      for (int k = 0; k < 12; k++) {
        f[k] = v[k];
      }
      mMatrixCache[matrixCount] = o2::gpu::Transform3D(f);
      mMatrixIndirection[i] = matrixCount++;
    } else {
      mMatrixIndirection[i] = -1;
    }
  }
}
