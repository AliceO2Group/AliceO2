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

#ifndef ALICEO2_ITSMFT_TRACKING_DETECTORDEFINITIONS_H_
#define ALICEO2_ITSMFT_TRACKING_DETECTORDEFINITIONS_H_

#include <array>
#include <cstddef>
#include <limits>

#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/Constants.h"

namespace o2::itsmft::tracking
{
namespace detail
{
constexpr NominalSurfaceMaterial siliconMaterial(float xOverX0) noexcept
{
  return {xOverX0, xOverX0 * o2::its::constants::Radl * o2::its::constants::Rho};
}

// Preserve the production prescription: 0.042/5 X/X0 for each sensor plane.
inline constexpr auto mftSurfaceMaterial = siliconMaterial(0.042f / 5.f);

} // namespace detail

// Canonical descriptors, using the exact production tracking values.
inline constexpr std::array<SurfaceDescriptor, ITSNLayers> kITSSurfaces{
  SurfaceDescriptor{0, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder, 0, 2.33959f, detail::siliconMaterial(5.e-3f), {-kITSLookupZHalfExtent[0], kITSLookupZHalfExtent[0]}},
  SurfaceDescriptor{1, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder, 0, 3.14076f, detail::siliconMaterial(5.e-3f), {-kITSLookupZHalfExtent[1], kITSLookupZHalfExtent[1]}},
  SurfaceDescriptor{2, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder, 0, 3.91924f, detail::siliconMaterial(5.e-3f), {-kITSLookupZHalfExtent[2], kITSLookupZHalfExtent[2]}},
  SurfaceDescriptor{3, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder, 0, 19.6213f, detail::siliconMaterial(1.e-2f), {-kITSLookupZHalfExtent[3], kITSLookupZHalfExtent[3]}},
  SurfaceDescriptor{4, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder, 0, 24.5597f, detail::siliconMaterial(1.e-2f), {-kITSLookupZHalfExtent[4], kITSLookupZHalfExtent[4]}},
  SurfaceDescriptor{5, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder, 0, 34.388f, detail::siliconMaterial(1.e-2f), {-kITSLookupZHalfExtent[5], kITSLookupZHalfExtent[5]}},
  SurfaceDescriptor{6, static_cast<uint8_t>(o2::detectors::DetID::ITS), SurfaceKind::Cylinder, 0, 39.3329f, detail::siliconMaterial(1.e-2f), {-kITSLookupZHalfExtent[6], kITSLookupZHalfExtent[6]}},
};

inline constexpr std::array<SurfaceDescriptor, MFTNLayers> kMFTSurfaces{
  SurfaceDescriptor{0, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -45.2889f, detail::mftSurfaceMaterial, {2.1f, 12.5f}},
  SurfaceDescriptor{1, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -46.7111f, detail::mftSurfaceMaterial, {2.1f, 12.5f}},
  SurfaceDescriptor{2, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -48.5889f, detail::mftSurfaceMaterial, {2.1f, 12.5f}},
  SurfaceDescriptor{3, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -50.0111f, detail::mftSurfaceMaterial, {2.1f, 12.5f}},
  SurfaceDescriptor{4, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -52.3889f, detail::mftSurfaceMaterial, {2.1f, 14.f}},
  SurfaceDescriptor{5, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -53.8111f, detail::mftSurfaceMaterial, {2.1f, 14.f}},
  SurfaceDescriptor{6, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -67.6889f, detail::mftSurfaceMaterial, {3.1f, 17.f}},
  SurfaceDescriptor{7, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -69.1111f, detail::mftSurfaceMaterial, {3.1f, 17.f}},
  SurfaceDescriptor{8, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -76.0889f, detail::mftSurfaceMaterial, {3.5f, 17.5f}},
  SurfaceDescriptor{9, static_cast<uint8_t>(o2::detectors::DetID::MFT), SurfaceKind::Disk, 0, -77.5111f, detail::mftSurfaceMaterial, {3.5f, 17.5f}},
};

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_DETECTORDEFINITIONS_H_ */
