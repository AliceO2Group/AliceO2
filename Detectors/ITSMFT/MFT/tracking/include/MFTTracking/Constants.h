// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Constants.h
/// \brief Some constants, fixed parameters and look-up-table functions
///

#ifndef O2_MFT_CONSTANTS_H_
#define O2_MFT_CONSTANTS_H_

#include <climits>
#include <vector>

#include <Rtypes.h>

namespace o2
{
namespace MFT
{

namespace Constants
{

namespace Math
{
constexpr Float_t Pi{ 3.14159265359f };
constexpr Float_t TwoPi{ 2.0f * Pi };
constexpr Float_t FloatMinThreshold{ 1e-20f };
} // namespace Math

namespace MFT
{
constexpr Int_t LayersNumber{ 10 };
constexpr Int_t TrackletsPerRoad{ LayersNumber - 1 };
constexpr Int_t CellsPerRoad{ LayersNumber - 2 };
constexpr Int_t ClustersPerCell{ 2 };
constexpr Int_t UnusedIndex{ -1 };
constexpr Float_t Resolution{ 0.0005f };
constexpr std::array<Float_t, LayersNumber> LayerZCoordinate()
{
  return std::array<Float_t, LayersNumber>{ -45.3, -46.7, -48.6, -50.0, -52.4, -53.8, -68.0, -69.4, -76.1, -77.5 };
}
constexpr Int_t MinTrackPoints{ 4 };
constexpr Int_t MaxTrackPoints{ 20 };
constexpr Float_t LTFclsRCut{ 0.0100 };
constexpr Float_t ROADclsRCut{ 0.0400 };
} // namespace MFT

namespace IndexTable
{
constexpr Float_t RMin{ 2.0 }; // [cm]
constexpr Float_t RMax{ 16.0 };

constexpr Float_t PhiMin{ 0. };
constexpr Float_t PhiMax{ Constants::Math::TwoPi }; // [rad]

constexpr Int_t RBins{ 50 };
constexpr Int_t PhiBins{ 50 };

constexpr Float_t InversePhiBinSize{ PhiBins / (PhiMax - PhiMin) };
constexpr Float_t InverseRBinSize{ RBins / (RMax - RMin) };

constexpr UChar_t LTFseed2BinWin{ 3 };
constexpr UChar_t LTFinterBinWin{ 3 };

constexpr Int_t getRBinIndex(const Float_t r)
{
  return (Int_t)((r - RMin) * InverseRBinSize);
}

constexpr Int_t getPhiBinIndex(const Float_t phi)
{
  return (Int_t)((phi - PhiMin) * InversePhiBinSize);
}

constexpr Int_t getBinIndex(const Int_t rIndex, const Int_t phiIndex)
{
  if (0 <= rIndex && rIndex < RBins &&
      0 <= phiIndex && phiIndex < PhiBins) {
    return (phiIndex * RBins + rIndex);
  }
  return -1;
}
} // namespace IndexTable

} // namespace Constants
} // namespace MFT
} // namespace o2

#endif /* O2_MFT_CONSTANTS_H_ */
