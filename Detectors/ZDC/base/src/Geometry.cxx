// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCBase/Geometry.h"
#include "FairLogger.h"

ClassImp(o2::zdc::Geometry);

using namespace o2::zdc;

constexpr double Geometry::ZNDIMENSION[3];  //;
constexpr double Geometry::ZNFIBRE[3];      //;
constexpr double Geometry::ZNSECTORS[2];    //;
constexpr double Geometry::ZNDIVISION[2];   //;
constexpr double Geometry::ZNGROOVES[3];    // = {0.03, 0.03, 50.};
constexpr double Geometry::ZNCPOSITION[3];  //; // = {0., 0., -11261.3};
constexpr double Geometry::ZNAPOSITION[3];  //; // = {0., 0., 11259.8};
constexpr double Geometry::ZNFIBREDIAMETER; // 18;
                                            //
constexpr double Geometry::ZPDIMENSION[3];  // = {11.2, 6., 75.};
constexpr double Geometry::ZPFIBRE[3];      // = {0., 0.0275, 75.};
constexpr double Geometry::ZPSECTORS[2];    // = {4, 1};
constexpr double Geometry::ZPDIVISION[2];   // = {7, 15};
constexpr double Geometry::ZPGROOVES[3];    // = {0.04, 0.04, 75.};
constexpr double Geometry::ZPCPOSITION[3];  // = {24.35, 0., -11253.3};
constexpr double Geometry::ZPAPOSITION[3];  // = {24.35, 0., 11251.8};
constexpr double Geometry::ZPFIBREDIAMETER; // = 28;

//
constexpr double Geometry::ZEMDIMENSION[6]; // = {20.62/2., 3.5, 3.5, 45., 0., 0.};
constexpr double Geometry::ZEMFIBRE[3];     // = {0., 0.0275, 7.};
constexpr double Geometry::ZEMDIVISION[3];  // = {92, 0, 20};
constexpr double Geometry::ZEMPOSITION[3];  // = {9.69, 0., 760.};
