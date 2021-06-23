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

/// \file HalfDetector.h
/// \brief Class describing geometry of one half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFDETECTOR_H_
#define ALICEO2_MFT_HALFDETECTOR_H_

#include "TNamed.h"
#include "TGeoVolume.h"

namespace o2
{
namespace mft
{
class HalfSegmentation;
}
} // namespace o2

namespace o2
{
namespace mft
{

class HalfDetector : public TNamed
{

 public:
  HalfDetector();
  HalfDetector(HalfSegmentation* segmentation);

  ~HalfDetector() override;

  /// \brief Returns the Volume holding the Half-MFT
  TGeoVolumeAssembly* getVolume() { return mHalfVolume; };

 protected:
  TGeoVolumeAssembly* mHalfVolume;

 private:
  HalfSegmentation* mSegmentation; ///< \brief Pointer to the half-MFT segmentation

  void createHalfDisks();

  ClassDefOverride(HalfDetector, 1);
};
} // namespace mft
} // namespace o2

#endif
