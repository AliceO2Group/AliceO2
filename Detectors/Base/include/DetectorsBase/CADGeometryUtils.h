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

/// \file CADGeometryUtils.h
/// \brief Helpers to inject CAD-derived (TGeo) geometry into O2 simulation
///
/// These utilities are shared between purely passive external modules
/// (o2::passive::ExternalModule) and sensitive external detectors
/// (o2::ext::ExternalDetector). They deal with the geometry produced by
/// scripts/geometry/O2_CADtoTGeo.py, which is emitted as a ROOT macro.

#ifndef ALICEO2_BASE_CADGEOMETRYUTILS_H
#define ALICEO2_BASE_CADGEOMETRYUTILS_H

#include <string>

class TGeoVolume;

namespace o2::base
{

/// JIT-compile a CAD-derived ROOT geometry macro (as produced by O2_CADtoTGeo.py)
/// and execute it to obtain the top TGeoVolume of the described module.
///
/// The macro body is wrapped into a unique namespace (derived from \a instanceTag)
/// so that several such macros — which all export identically named symbols
/// (build(), get_builder_hook_unchecked(), ...) — can coexist in the same Cling
/// session without colliding. Returns nullptr on failure.
///
/// \param macroFile path to the geometry macro (shell variables are expanded)
/// \param instanceTag a short tag used to build a unique, human-readable namespace
TGeoVolume* buildCADVolumeFromMacro(const std::string& macroFile, const std::string& instanceTag);

/// Re-register the TGeo media used in the volume tree rooted at \a top into the O2
/// MaterialManager under ownership of \a modulename, rewiring the volumes to the
/// newly created media. This brings the CAD-imported media under O2's media/cut
/// handling (so that e.g. tracking cuts apply consistently).
void remapCADMedia(TGeoVolume* top, const char* modulename);

} // namespace o2::base

#endif
