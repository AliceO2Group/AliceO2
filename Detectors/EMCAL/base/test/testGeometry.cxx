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