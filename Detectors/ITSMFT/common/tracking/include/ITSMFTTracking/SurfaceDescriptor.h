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

#ifndef ALICEO2_ITSMFT_TRACKING_SURFACEDESCRIPTOR_H_
#define ALICEO2_ITSMFT_TRACKING_SURFACEDESCRIPTOR_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "GPUCommonDef.h"
#include "ITSMFTTracking/IdTypes.h"

namespace o2::itsmft::tracking
{

// Nominal normal-incidence material; zero denotes material not configured.
struct NominalSurfaceMaterial {
  float xOverX0{0.f};
  float arealDensityGPerCm2{0.f};
};

struct SurfaceChartRange {
  float min{0.f};
  float max{0.f};

  GPUhdi() constexpr bool isValid() const noexcept { return min < max; }
};

static_assert(std::is_standard_layout_v<NominalSurfaceMaterial>);
static_assert(std::is_trivially_copyable_v<NominalSurfaceMaterial>);
static_assert(sizeof(NominalSurfaceMaterial) == 8);
static_assert(alignof(NominalSurfaceMaterial) == 4);
static_assert(offsetof(NominalSurfaceMaterial, xOverX0) == 0);
static_assert(offsetof(NominalSurfaceMaterial, arealDensityGPerCm2) == 4);
static_assert(std::is_standard_layout_v<SurfaceChartRange>);
static_assert(std::is_trivially_copyable_v<SurfaceChartRange>);
static_assert(sizeof(SurfaceChartRange) == 8);

// Immutable surface geometry and nominal material. Its LayerId is the dense
// position of this descriptor in DetectorLayout and is intentionally not
// duplicated here.
struct SurfaceDescriptor {
  uint16_t detectorSurfaceIndex{0};
  uint8_t detectorId{0};
  SurfaceKind kind{SurfaceKind::Undefined};
  uint16_t flags{0};
  float referenceCoordinate{0.f}; // nominal radius for cylinders, z for disks
  NominalSurfaceMaterial material{};
  SurfaceChartRange chartRange{};
};

static_assert(std::is_standard_layout_v<SurfaceDescriptor>);
static_assert(std::is_trivially_copyable_v<SurfaceDescriptor>);
static_assert(sizeof(SurfaceDescriptor) == 28);
static_assert(alignof(SurfaceDescriptor) == 4);
static_assert(offsetof(SurfaceDescriptor, detectorSurfaceIndex) == 0);
static_assert(offsetof(SurfaceDescriptor, detectorId) == 2);
static_assert(offsetof(SurfaceDescriptor, kind) == 3);
static_assert(offsetof(SurfaceDescriptor, flags) == 4);
static_assert(offsetof(SurfaceDescriptor, referenceCoordinate) == 8);
static_assert(offsetof(SurfaceDescriptor, material) == 12);
static_assert(offsetof(SurfaceDescriptor, chartRange) == 20);

// Non-owning surface-catalog view. Topology, timing and measurements stay
// outside this POD so loading and propagation do not depend on them.
struct SurfaceCatalogView {
  const SurfaceDescriptor* surfaces{nullptr};
  uint32_t nSurfaces{0};

  GPUhdi() uint32_t getSurfaceIndex(LayerId id) const
  {
    return id.isValid() && id.value() < nSurfaces ? id.value() : nSurfaces;
  }

  GPUhdi() bool hasSurface(LayerId id) const { return getSurfaceIndex(id) < nSurfaces; }
  GPUhdi() const SurfaceDescriptor& getSurface(LayerId id) const { return surfaces[getSurfaceIndex(id)]; }
};

static_assert(std::is_standard_layout_v<SurfaceCatalogView>);
static_assert(std::is_trivially_copyable_v<SurfaceCatalogView>);

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_SURFACEDESCRIPTOR_H_ */
