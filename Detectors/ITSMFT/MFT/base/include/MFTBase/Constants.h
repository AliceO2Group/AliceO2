// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Constants.h
/// \brief Constants for the MFT; distance unit is cm
/// \author Antonio Uras <antonio.uras@cern.ch>

#ifndef ALICEO2_MFT_CONSTANTS_H_
#define ALICEO2_MFT_CONSTANTS_H_

#include <Rtypes.h>

namespace o2
{
namespace mft
{
namespace constants
{
constexpr Int_t DisksNumber{ 5 };
constexpr Int_t LayersNumber{ 2 * DisksNumber };
constexpr Int_t HalvesNumber{ 2 };

/// layer Z position to the middle of the CMOS sensor
constexpr Double_t LayerZPosition[] = { -45.3, -46.7, -48.6, -50.0, -52.4, -53.8, -67.7, -69.1, -76.1, -77.5 };

/// disk thickness in X0
constexpr Double_t DiskThicknessInX0[DisksNumber] = { 0.008, 0.008, 0.008, 0.008, 0.008 };

inline const Double_t diskThicknessInX0(Int_t id) { return (id >= 0 && id < DisksNumber) ? DiskThicknessInX0[id] : 0.; }
inline const Double_t layerZPosition(Int_t id)
{
  return (id >= 0 && id < LayersNumber) ? LayerZPosition[id] + (-(id % 2) * 2 - 1) * 0.0025 : 0.;
}
} // namespace Constants
} // namespace mft
} // namespace o2

#endif
