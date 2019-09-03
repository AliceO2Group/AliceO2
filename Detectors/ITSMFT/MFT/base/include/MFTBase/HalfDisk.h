// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HalfDisk.h
/// \brief Class building geometry of one half of an MFT disk
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFDISK_H_
#define ALICEO2_MFT_HALFDISK_H_

#include "TNamed.h"

class TGeoVolumeAssembly;

namespace o2
{
namespace mft
{
class HalfDiskSegmentation;
}
} // namespace o2
namespace o2
{
namespace mft
{
class Support;
}
} // namespace o2
namespace o2
{
namespace mft
{
class HeatExchanger;
}
} // namespace o2

namespace o2
{
namespace mft
{

class HalfDisk : public TNamed
{

 public:
  HalfDisk();
  HalfDisk(HalfDiskSegmentation* segmentation);

  TGeoVolumeAssembly* createHeatExchanger();
  void createLadders();

  ~HalfDisk() override;

  /// \brief Returns a pointer to the Volume Assembly describing the entire half-disk
  TGeoVolumeAssembly* getVolume() { return mHalfDiskVolume; };

 private:
  Support* mSupport;                   ///< \brief Disk Support
  HeatExchanger* mHeatExchanger;       ///< \brief Heat Exchanger
  TGeoVolumeAssembly* mHalfDiskVolume; ///< \brief Half-Disk Volume
  HalfDiskSegmentation* mSegmentation; ///< \brief Virtual Segmentation of the half-disk

  ClassDefOverride(HalfDisk, 1);
};
} // namespace mft
} // namespace o2

#endif
