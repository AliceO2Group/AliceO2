// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Ladder.h
/// \brief Class building the Ladder geometry
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_LADDER_H_
#define ALICEO2_MFT_LADDER_H_

#include "TNamed.h"
#include "TGeoVolume.h"

namespace o2
{
namespace mft
{
class LadderSegmentation;
}
} // namespace o2
namespace o2
{
namespace mft
{
class Flex;
}
} // namespace o2

namespace o2
{
namespace mft
{

class Ladder : public TNamed
{

 public:
  Ladder();
  Ladder(LadderSegmentation* segmentation);

  ~Ladder() override;

  TGeoVolume* createVolume();
  void createSensors();

 private:
  const static Double_t sLadderDeltaY; ///< \brief Ladder size along Y direction (height)
  const static Double_t sLadderDeltaZ; ///< \brief Ladder size along Z direction (thickness)
  LadderSegmentation* mSegmentation;   ///< \brief Virtual Segmentation object of the ladder
  Flex* mFlex;                         ///< \brief Flex object (\todo to be removed ?)
  TGeoVolumeAssembly* mLadderVolume;   ///< \brief Pointer to the Volume holding the ladder geometry

  ClassDefOverride(Ladder, 1);
};
} // namespace mft
} // namespace o2

#endif
