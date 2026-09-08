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

#ifndef ALICEO2_ITSMFT_TRACKING_SURFACESPEC_H_
#define ALICEO2_ITSMFT_TRACKING_SURFACESPEC_H_

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"

namespace o2::itsmft::tracking
{

// Detector-qualified identity of a logical tracking surface. The detector ID
// remains open to every detector representable by the existing uint8_t field.
struct DetectorLayerIdentity {
  uint8_t detectorId{0};
  uint16_t detectorSurfaceIndex{0};

  GPUhdi() friend constexpr bool operator==(DetectorLayerIdentity lhs, DetectorLayerIdentity rhs) noexcept
  {
    return lhs.detectorId == rhs.detectorId && lhs.detectorSurfaceIndex == rhs.detectorSurfaceIndex;
  }
  GPUhdi() friend constexpr bool operator!=(DetectorLayerIdentity lhs, DetectorLayerIdentity rhs) noexcept { return !(lhs == rhs); }
};

struct StaticSurfaceDescriptor {
  DetectorLayerIdentity identity{};
  SurfaceKind kind{SurfaceKind::Undefined};
  float nominalReferenceCoordinate{0.f};
  NominalSurfaceMaterial material{};
  SurfaceChartRange chartRange{};
};

// Precondition: source belongs to a validated SurfaceSpec. This is an
// ideal/static layout projection only; it neither validates nor repairs an
// arbitrary descriptor. Runtime geometry observations do not define another
// catalogue.
GPUhdi() constexpr SurfaceDescriptor toRuntimeSurfaceDescriptor(const StaticSurfaceDescriptor& source) noexcept
{
  return SurfaceDescriptor{source.identity.detectorSurfaceIndex,
                           source.identity.detectorId,
                           source.kind,
                           0,
                           source.nominalReferenceCoordinate,
                           source.material,
                           source.chartRange};
}

#define O2_ITSMFT_ASSERT_STATIC_SURFACE_TYPE(Type, Size, Alignment) \
  static_assert(std::is_standard_layout_v<Type>);                   \
  static_assert(std::is_trivially_copyable_v<Type>);                \
  static_assert(sizeof(Type) == Size);                              \
  static_assert(alignof(Type) == Alignment)

O2_ITSMFT_ASSERT_STATIC_SURFACE_TYPE(DetectorLayerIdentity, 4, 2);
O2_ITSMFT_ASSERT_STATIC_SURFACE_TYPE(StaticSurfaceDescriptor, 28, 4);

static_assert(offsetof(DetectorLayerIdentity, detectorId) == 0);
static_assert(offsetof(DetectorLayerIdentity, detectorSurfaceIndex) == 2);
static_assert(offsetof(StaticSurfaceDescriptor, identity) == 0);
static_assert(offsetof(StaticSurfaceDescriptor, kind) == 4);
static_assert(offsetof(StaticSurfaceDescriptor, nominalReferenceCoordinate) == 8);
static_assert(offsetof(StaticSurfaceDescriptor, material) == 12);
static_assert(offsetof(StaticSurfaceDescriptor, chartRange) == 20);

#undef O2_ITSMFT_ASSERT_STATIC_SURFACE_TYPE

namespace detail
{
template <typename T>
struct IsStaticSurfaceArray : std::false_type {
};

template <std::size_t N>
struct IsStaticSurfaceArray<std::array<StaticSurfaceDescriptor, N>> : std::true_type {
};

constexpr bool isEnabled(SurfaceKind kind) noexcept
{
  return isRecognizedSurfaceKind(kind);
}

template <typename Spec>
consteval bool hasStaticSurfaceArray()
{
  if constexpr (!requires { Spec::surfaces; }) {
    return false;
  } else {
    using Array = std::remove_cv_t<decltype(Spec::surfaces)>;
    if constexpr (!IsStaticSurfaceArray<Array>::value) {
      return false;
    } else {
      // Being usable as a non-type template argument proves static lifetime
      // and constant initialization of the canonical array.
      return requires { typename std::integral_constant<const StaticSurfaceDescriptor*, Spec::surfaces.data()>; };
    }
  }
}

template <std::size_t N>
consteval bool validateSurfaceArray(const std::array<StaticSurfaceDescriptor, N>& surfaces)
{
  if constexpr (N > MaxLayoutSurfaces) {
    return false;
  }

  for (std::size_t i = 0; i < N; ++i) {
    const auto& surface = surfaces[i];
    if (!isEnabled(surface.kind) || !o2::gpu::GPUCommonMath::Finite(surface.nominalReferenceCoordinate) ||
        (surface.kind == SurfaceKind::Cylinder && surface.nominalReferenceCoordinate <= 0.f)) {
      return false;
    }

    if (!o2::gpu::GPUCommonMath::Finite(surface.material.xOverX0) || surface.material.xOverX0 < 0.f ||
        !o2::gpu::GPUCommonMath::Finite(surface.material.arealDensityGPerCm2) || surface.material.arealDensityGPerCm2 < 0.f) {
      return false;
    }

    for (std::size_t other = i + 1; other < N; ++other) {
      if (surface.identity == surfaces[other].identity) {
        return false;
      }
    }
  }

  // Detector-local indices form an independent dense [0, N) range for every
  // arbitrary detector ID represented in the catalogue.
  for (std::size_t i = 0; i < N; ++i) {
    const auto detectorId = surfaces[i].identity.detectorId;
    std::size_t detectorCount = 0;
    for (const auto& surface : surfaces) {
      detectorCount += surface.identity.detectorId == detectorId;
    }
    for (std::size_t expected = 0; expected < detectorCount; ++expected) {
      bool found = false;
      for (const auto& surface : surfaces) {
        found = found ||
                (surface.identity.detectorId == detectorId && surface.identity.detectorSurfaceIndex == expected);
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}

template <typename Spec>
consteval bool validateSurfaceSpecDefinition()
{
  return validateSurfaceArray(Spec::surfaces);
}
} // namespace detail

template <typename Spec>
// Structural requirement only: canonical inline-static-array shape and
// lifetime. Catalogue contents are deliberately not accepted by this concept.
concept SurfaceSpecDefinition = detail::hasStaticSurfaceArray<Spec>();

template <typename Spec>
// A consumer-facing SurfaceSpec has both the required definition shape and a
// fully validated catalogue.
concept SurfaceSpec = SurfaceSpecDefinition<Spec> && detail::validateSurfaceSpecDefinition<Spec>();

template <SurfaceSpec Spec>
inline constexpr std::size_t SurfaceCount = std::tuple_size_v<std::remove_cv_t<decltype(Spec::surfaces)>>;

template <typename Spec>
consteval bool validateSurfaceSpec()
{
  if constexpr (!SurfaceSpecDefinition<Spec>) {
    return false;
  } else {
    return detail::validateSurfaceSpecDefinition<Spec>();
  }
}

namespace detail
{
template <SurfaceSpecDefinition A, SurfaceSpecDefinition B>
consteval auto concatenate()
{
  constexpr auto countA = std::tuple_size_v<std::remove_cv_t<decltype(A::surfaces)>>;
  constexpr auto countB = std::tuple_size_v<std::remove_cv_t<decltype(B::surfaces)>>;
  std::array<StaticSurfaceDescriptor, countA + countB> result{};
  std::size_t output = 0;
  for (const auto& surface : A::surfaces) {
    result[output++] = surface;
  }
  for (const auto& surface : B::surfaces) {
    result[output++] = surface;
  }
  return result;
}

template <SurfaceSpecDefinition A, SurfaceSpecDefinition B>
consteval bool surfaceSpecsCanBeConcatenated()
{
  if constexpr (!SurfaceSpec<A> || !SurfaceSpec<B>) {
    return false;
  } else if constexpr (SurfaceCount<A> + SurfaceCount<B> > MaxLayoutSurfaces) {
    return false;
  } else {
    return validateSurfaceArray(concatenate<A, B>());
  }
}
} // namespace detail

template <SurfaceSpecDefinition A, SurfaceSpecDefinition B>
inline constexpr bool SurfaceSpecsCanBeConcatenated = detail::surfaceSpecsCanBeConcatenated<A, B>();

template <SurfaceSpec A, SurfaceSpec B>
struct ConcatenatedSurfaceSpec {
  static_assert(SurfaceSpecsCanBeConcatenated<A, B>, "SurfaceSpecs cannot be concatenated into a valid catalogue");
  inline static constexpr auto surfaces = detail::concatenate<A, B>();
};

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_SURFACESPEC_H_ */
