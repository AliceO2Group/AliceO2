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

///
/// @author  Diego Stocco

#define BOOST_TEST_MODULE Test MID CrateMapper
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <boost/test/data/test_case.hpp>
#include <iostream>
#include "DataFormatsMID/ROBoard.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDRaw/CrateMapper.h"
#include "MIDRaw/CrateParameters.h"

BOOST_AUTO_TEST_SUITE(o2_mid_CrateMapper)

BOOST_AUTO_TEST_CASE(FEEBoardToDE)
{
  o2::mid::CrateMapper crateMapper;
  for (uint8_t icrate = 0; icrate < o2::mid::crateparams::sNCrates; ++icrate) {
    int crateIdInSide = icrate % (o2::mid::crateparams::sNCrates / 2);
    for (uint8_t iboard = 0; iboard < o2::mid::crateparams::sMaxNBoardsInCrate; ++iboard) {
      int boardInSide = iboard % o2::mid::crateparams::sMaxNBoardsInLink;
      bool exceptionExpected = false;
      if ((crateIdInSide == 1 || crateIdInSide == 3) && iboard == 15) {
        exceptionExpected = true;
      } else if (crateIdInSide == 2 && boardInSide == 7) {
        exceptionExpected = true;
      } else if (crateIdInSide == 7 && iboard > 8) {
        exceptionExpected = true;
      }
      bool exceptionReceived = false;
      try {
        crateMapper.roLocalBoardToDE(o2::mid::raw::makeUniqueLocID(icrate, iboard));
      } catch (std::runtime_error except) {
        exceptionReceived = true;
      }
      std::stringstream ss;
      ss << "CrateId: " << static_cast<int>(icrate) << "  boardId: " << static_cast<int>(iboard) << "  exception expected: " << exceptionExpected << "  received: " << exceptionReceived;
      BOOST_TEST(exceptionExpected == exceptionReceived, ss.str());
    }
  }
}

BOOST_AUTO_TEST_CASE(DEBoardToFEE)
{
  o2::mid::Mapping mapping;
  o2::mid::CrateMapper crateMapper;
  for (int ide = 0; ide < o2::mid::detparams::NDetectionElements; ++ide) {
    for (int icol = mapping.getFirstColumn(ide); icol < 7; ++icol) {
      for (int iline = mapping.getFirstBoardBP(icol, ide); iline <= mapping.getLastBoardBP(icol, ide); ++iline) {
        try {
          crateMapper.deLocalBoardToRO(ide, icol, iline);
        } catch (std::runtime_error except) {
          std::stringstream ss;
          ss << "DEId: " << static_cast<int>(ide) << "  colId: " << static_cast<int>(icol) << "  lineId: " << static_cast<int>(iline) << "  exception received!";
          BOOST_TEST(false, ss.str());
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(Consistency)
{
  o2::mid::Mapping mapping;
  o2::mid::CrateMapper crateMapper;
  for (int ide = 0; ide < o2::mid::detparams::NDetectionElements; ++ide) {
    for (int icol = mapping.getFirstColumn(ide); icol < 7; ++icol) {
      for (int iline = mapping.getFirstBoardBP(icol, ide); iline <= mapping.getLastBoardBP(icol, ide); ++iline) {
        auto uniqueLocId = crateMapper.deLocalBoardToRO(ide, icol, iline);
        auto crateId = o2::mid::raw::getCrateId(uniqueLocId);
        auto deBoardId = crateMapper.roLocalBoardToDE(uniqueLocId);
        BOOST_TEST(static_cast<int>(o2::mid::detparams::getColumnIdFromFEEId(deBoardId)) == icol);
        BOOST_TEST(static_cast<int>(o2::mid::detparams::getLineIdFromFEEId(deBoardId)) == iline);
        int rpcLineId = o2::mid::detparams::getDEIdFromFEEId(deBoardId);
        BOOST_TEST(rpcLineId == o2::mid::detparams::getRPCLine(ide));
        int ich = o2::mid::detparams::getChamber(ide);
        BOOST_TEST(o2::mid::detparams::getDEId(o2::mid::crateparams::isRightSide(crateId), ich, rpcLineId) == ide);
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
