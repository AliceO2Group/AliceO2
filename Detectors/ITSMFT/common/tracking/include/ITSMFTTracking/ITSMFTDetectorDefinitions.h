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

#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/SurfaceSpec.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/Constants.h"

namespace o2::itsmft::tracking
{

static_assert(MFTNLayers % 2 == 0);
inline constexpr int MFTDisks = MFTNLayers / 2;
inline constexpr std::array<float, ITSNLayers> kNominalITSLayerX0{
  5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f};
inline constexpr float kMFTNominalRadLength = 0.042f;
inline constexpr std::array<float, MFTNLayers> kMFTLookupRMin{
  2.1f, 2.1f, 2.1f, 2.1f, 2.1f, 2.1f, 3.1f, 3.1f, 3.5f, 3.5f};
inline constexpr std::array<float, MFTNLayers> kMFTLookupRMax{
  12.5f, 12.5f, 12.5f, 12.5f, 14.f, 14.f, 17.f, 17.f, 17.5f, 17.5f};

constexpr std::array<float, MFTNLayers> makeNominalMFTLayerX0()
{
  std::array<float, MFTNLayers> values{};
  // Each disk's budget is shared by its two sensor planes: the refit applies
  // the nominal material once per attached surface.
  for (auto& value : values) {
    value = kMFTNominalRadLength / static_cast<float>(MFTNLayers);
  }
  return values;
}

inline constexpr std::array<float, MFTNLayers> kNominalMFTLayerX0 = makeNominalMFTLayerX0();

constexpr NominalSurfaceMaterial itsLayerMaterial(std::size_t layer) noexcept
{
  const float x0 = kNominalITSLayerX0[layer];
  return {x0, x0 * o2::its::constants::Radl * o2::its::constants::Rho};
}

struct ITSSurfaceSpec {
  inline static constexpr std::array<StaticSurfaceDescriptor, ITSNLayers> surfaces{
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::ITS), 0}, SurfaceKind::Cylinder, 2.3259652f, itsLayerMaterial(0), {-kITSLookupZHalfExtent[0], kITSLookupZHalfExtent[0]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::ITS), 1}, SurfaceKind::Cylinder, 3.1353536f, itsLayerMaterial(1), {-kITSLookupZHalfExtent[1], kITSLookupZHalfExtent[1]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::ITS), 2}, SurfaceKind::Cylinder, 3.9162421f, itsLayerMaterial(2), {-kITSLookupZHalfExtent[2], kITSLookupZHalfExtent[2]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::ITS), 3}, SurfaceKind::Cylinder, 19.58824f, itsLayerMaterial(3), {-kITSLookupZHalfExtent[3], kITSLookupZHalfExtent[3]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::ITS), 4}, SurfaceKind::Cylinder, 24.527159f, itsLayerMaterial(4), {-kITSLookupZHalfExtent[4], kITSLookupZHalfExtent[4]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::ITS), 5}, SurfaceKind::Cylinder, 34.354595f, itsLayerMaterial(5), {-kITSLookupZHalfExtent[5], kITSLookupZHalfExtent[5]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::ITS), 6}, SurfaceKind::Cylinder, 39.310642f, itsLayerMaterial(6), {-kITSLookupZHalfExtent[6], kITSLookupZHalfExtent[6]}},
  };
};

static_assert(SurfaceSpec<ITSSurfaceSpec>);
static_assert(SurfaceCount<ITSSurfaceSpec> == ITSNLayers);

constexpr NominalSurfaceMaterial mftLayerMaterial(std::size_t layer) noexcept
{
  const float x0 = kNominalMFTLayerX0[layer];
  return {x0, x0 * o2::its::constants::Radl * o2::its::constants::Rho};
}

struct MFTSurfaceSpec {
  inline static constexpr std::array<StaticSurfaceDescriptor, MFTNLayers> surfaces{
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 0}, SurfaceKind::Disk, -45.2889f, mftLayerMaterial(0), {kMFTLookupRMin[0], kMFTLookupRMax[0]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 1}, SurfaceKind::Disk, -46.7111f, mftLayerMaterial(1), {kMFTLookupRMin[1], kMFTLookupRMax[1]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 2}, SurfaceKind::Disk, -48.5889f, mftLayerMaterial(2), {kMFTLookupRMin[2], kMFTLookupRMax[2]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 3}, SurfaceKind::Disk, -50.0111f, mftLayerMaterial(3), {kMFTLookupRMin[3], kMFTLookupRMax[3]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 4}, SurfaceKind::Disk, -52.3889f, mftLayerMaterial(4), {kMFTLookupRMin[4], kMFTLookupRMax[4]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 5}, SurfaceKind::Disk, -53.8111f, mftLayerMaterial(5), {kMFTLookupRMin[5], kMFTLookupRMax[5]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 6}, SurfaceKind::Disk, -67.6889f, mftLayerMaterial(6), {kMFTLookupRMin[6], kMFTLookupRMax[6]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 7}, SurfaceKind::Disk, -69.1111f, mftLayerMaterial(7), {kMFTLookupRMin[7], kMFTLookupRMax[7]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 8}, SurfaceKind::Disk, -76.0889f, mftLayerMaterial(8), {kMFTLookupRMin[8], kMFTLookupRMax[8]}},
    StaticSurfaceDescriptor{{static_cast<uint8_t>(o2::detectors::DetID::MFT), 9}, SurfaceKind::Disk, -77.5111f, mftLayerMaterial(9), {kMFTLookupRMin[9], kMFTLookupRMax[9]}},
  };
};

static_assert(SurfaceSpec<MFTSurfaceSpec>);
static_assert(SurfaceCount<MFTSurfaceSpec> == MFTNLayers);

template <SurfaceSpec Spec>
consteval std::array<SurfaceDescriptor, SurfaceCount<Spec>> projectStaticSurfaceCatalog() noexcept
{
  std::array<SurfaceDescriptor, SurfaceCount<Spec>> result{};
  for (std::size_t i = 0; i < SurfaceCount<Spec>; ++i) {
    result[i] = toRuntimeSurfaceDescriptor(Spec::surfaces[i]);
  }
  return result;
}

inline constexpr auto kITSStaticSurfaceCatalog = projectStaticSurfaceCatalog<ITSSurfaceSpec>();
inline constexpr auto kMFTStaticSurfaceCatalog = projectStaticSurfaceCatalog<MFTSurfaceSpec>();

static_assert(kITSStaticSurfaceCatalog.size() == ITSNLayers);
static_assert(kMFTStaticSurfaceCatalog.size() == MFTNLayers);

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_DETECTORDEFINITIONS_H_ */
