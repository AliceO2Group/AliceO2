// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Sandro Wenzel <sandro.wenzel@cern.ch>
/// \since 2026-07

#ifndef ALICEO2_CADSUPPORT_O2SURFACESOLIDIO_
#define ALICEO2_CADSUPPORT_O2SURFACESOLIDIO_

#include <string>

namespace o2
{
namespace base
{
class O2Tessellated;
}
namespace cad
{

class O2BVHSurfaceSolid;
class O2FlatCSG;

/// Load an exact-surface sidecar (surfaces_*.bin, versions 1-3) into \a solid through its Add*Surface methods; call CloseShape() after.
/// False on an I/O or format error, when the solid may be partly filled and should be discarded.
bool LoadSurfaceSolid(const std::string& file, O2BVHSurfaceSolid& solid);

/// Load a facet sidecar (facets_*.bin: a uint32 triangle count, then nine float32 per triangle) into \a solid; call CloseShape() after.
/// False on an I/O or format error; degenerate facets are skipped and counted in a warning.
bool LoadFacetSolid(const std::string& file, o2::base::O2Tessellated& solid);

/// Load a flat-CSG sidecar (flatcsg_*.bin, version 1) into \a solid; call CloseShape() after. False on an I/O or format error.
bool LoadFlatCSG(const std::string& file, O2FlatCSG& solid);

/// Write \a solid in the same format. Used by the converter's tests and by the round-trip case;
/// the production writer is Detectors/CADSupport/tools/cadsupport/flat.py, and the two must agree byte for byte.
bool WriteFlatCSG(const std::string& file, const O2FlatCSG& solid);

} // namespace cad
} // namespace o2

#endif
