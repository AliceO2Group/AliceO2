// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ZDC_GEOMETRY_H
#define ALICEO2_ZDC_GEOMETRY_H

#include "TMath.h"
#include <cmath>

namespace o2
{
namespace zdc
{

class Geometry
{
 public:
  static constexpr double ZNDIMENSION[3] = {3.52, 3.52, 50.};
  static constexpr double ZNFIBRE[3] = {0., 0.01825, 50.};
  static constexpr double ZNSECTORS[2] = {2, 2};
  static constexpr double ZNDIVISION[2] = {11, 11};
  static constexpr double ZNGROOVES[3] = {0.03, 0.03, 50.};
  static constexpr double ZNCPOSITION[3] = {0., 0., -11261.3};
  static constexpr double ZNAPOSITION[3] = {0., 0., 11259.8};
  static constexpr double ZNFIBREDIAMETER = 18;
  //
  static constexpr double ZPDIMENSION[3] = {11.2, 6., 75.};
  static constexpr double ZPFIBRE[3] = {0., 0.0275, 75.};
  static constexpr double ZPSECTORS[2] = {4, 1};
  static constexpr double ZPDIVISION[2] = {7, 15};
  static constexpr double ZPGROOVES[3] = {0.04, 0.04, 75.};
  static constexpr double ZPCPOSITION[3] = {24.35, 0., -11253.3};
  static constexpr double ZPAPOSITION[3] = {24.35, 0., 11251.8};
  static constexpr double ZPFIBREDIAMETER = 28;

  //
  static constexpr double ZEMDIMENSION[6] = {20.62 / 2., 3.5, 3.5, 45., 0., 0.};
  static constexpr double ZEMFIBRE[3] = {0., 0.0275, 7.};
  static constexpr double ZEMDIVISION[3] = {92, 0, 20};
  //
  // using 1.4142135 instead of sqrt(2.): check!
  static constexpr double ZEMPOSITION[3] = {9.69, 0., 760.};

 private:
  static Int_t getDetector(const Int_t det);
  static Int_t getSector(const Int_t tow);
  static Int_t getVolumeId(const Int_t* vol);

  ClassDefNV(Geometry, 1);
};
} // namespace zdc
} // namespace o2

#endif
