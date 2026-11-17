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
#define BOOST_TEST_MODULE Test EMCAL Base
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <fmt/format.h>
#include "EMCALBase/Geometry.h"
#include "EMCALBase/ShishKebabTrd1Module.h"
#include <array>
#include <cmath>
#include <iostream>
#include <fstream>

std::tuple<int, int, int, int> GetRefCellIndex(int CellId);

/// \macro Test implementation of the EMCAL Geometry
///
/// Test coverage:
/// - GetCellIndex (get #sm, #mod, phi index and eta index): all cells (0-17663)
/// - Invalid CellId: exception test for cell -1 and 17664
BOOST_AUTO_TEST_CASE(Geometry_test)
{
  auto testgeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);

  // Check GetCellIndex function for all valid cells by comparing to GetRefCellIndex function
  for (int iCell = 0; iCell < 17664; iCell++) {
    auto [smod, mod, iphi, ieta] = testgeometry->GetCellIndex(iCell);
    auto [smod_ref, mod_ref, iphi_ref, ieta_ref] = GetRefCellIndex(iCell);
    BOOST_CHECK_EQUAL(smod, smod_ref);
    BOOST_CHECK_EQUAL(mod, mod_ref);
    BOOST_CHECK_EQUAL(iphi, iphi_ref);
    BOOST_CHECK_EQUAL(ieta, ieta_ref);
  } // And then check the exeptions of -1 and 17664
  BOOST_CHECK_EXCEPTION(testgeometry->GetCellIndex(-1), o2::emcal::InvalidCellIDException, [](o2::emcal::InvalidCellIDException const& mCellID) { return -1; });
  BOOST_CHECK_EXCEPTION(testgeometry->GetCellIndex(17664), o2::emcal::InvalidCellIDException, [](o2::emcal::InvalidCellIDException const& mCellID) { return 17664; });
}

std::tuple<int, int, int, int> GetRefCellIndex(int CellId)
{
  // Four cells per module:
  int ieta = CellId % 2;                                   // cells 0 and 2 (in each module) have eta index 0
  int iphi = (CellId % 4 == 2 || CellId % 4 == 3) ? 1 : 0; // cells 0 and 1 (in each module) have phi index 0

  int smod = 0, mod = 0;                          // Super module number and module number
  if (CellId >= 0 && CellId < 11520) {            // The first 10 super modules are full modules
    smod = CellId / 1152;                         // Their number is their cell number divided by the cells per sm (rounded down)
    mod = (CellId % 1152) / 4;                    // And the module is the cell number within the sm (%) divided by four (four cells in one module)
  } else if (CellId >= 11520 && CellId < 12288) { // First two one thirds
    smod = 10 + (CellId - 11520) / 384;           // +10 to account for the - 11520
    mod = ((CellId - 11520) % 384) / 4;           // -11520 to subtract all cells in full super modules
  } else if (CellId >= 12288 && CellId < 16896) { // Six two third modules
    smod = 12 + (CellId - 12288) / 768;
    mod = ((CellId - 12288) % 768) / 4;
  } else if (CellId >= 16896 && CellId < 17664) { // Second two one third modules
    smod = 18 + (CellId - 16896) / 384;
    mod = ((CellId - 16896) % 384) / 4;
  }

  return std::make_tuple(smod, mod, iphi, ieta);
}

/// \macro Test that neighbouring EMCAL modules touch exactly
///
/// Test coverage:
/// - the eta side faces of adjacent ShishKebabTrd1Modules coincide
/// - the first module's inner face starts on the inner radius and its left face lies on z = 0
BOOST_AUTO_TEST_CASE(ModuleFaces_test)
{
  auto geo = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
  const auto& modules = geo->GetShishKebabTrd1Modules();
  const double dx1 = geo->GetEtaModuleSize() / 2., dx2 = geo->Get2Trd1Dx2() / 2., dz = geo->GetLongModuleSize() / 2.;
  const double r0 = geo->GetIPDistance() + geo->GetSteelFrontThickness();
  // corner of a module in (z, r): s = -1 inner / +1 outer end, t = -1 small z / +1 large z side
  auto corner = [&](const o2::emcal::ShishKebabTrd1Module& m, int s, int t) {
    const double th = m.GetTheta(), dx = s < 0 ? dx1 : dx2;
    const auto& c = m.GetCenterOfModule();
    return std::array<double, 2>{c.X() + s * dz * std::cos(th) + t * dx * std::sin(th),
                                 c.Y() + s * dz * std::sin(th) - t * dx * std::cos(th)};
  };
  const double tolerance = 1e-10; // cm
  BOOST_CHECK_SMALL(corner(modules[0], -1, +1)[1] - r0, tolerance);
  BOOST_CHECK_SMALL(corner(modules[0], -1, -1)[0], tolerance);
  BOOST_CHECK_SMALL(corner(modules[0], +1, -1)[0], tolerance);
  for (std::size_t i = 0; i + 1 < modules.size(); ++i) {
    // distance of the next module's small-z face corners from this module's large-z face
    const auto a = corner(modules[i], -1, +1), b = corner(modules[i], +1, +1);
    const double nz = b[1] - a[1], nr = -(b[0] - a[0]), norm = std::hypot(nz, nr);
    for (int s : {-1, +1}) {
      const auto p = corner(modules[i + 1], s, -1);
      BOOST_CHECK_SMALL(((p[0] - a[0]) * nz + (p[1] - a[1]) * nr) / norm, tolerance);
    }
  }
}
