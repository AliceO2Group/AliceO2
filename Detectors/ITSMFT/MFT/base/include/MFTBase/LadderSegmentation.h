// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file LadderSegmentation.h
/// \brief Description of the virtual segmentation of a ladder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_LADDERSEGMENTATION_H_
#define ALICEO2_MFT_LADDERSEGMENTATION_H_

#include "TClonesArray.h"
#include "TXMLEngine.h"

#include "MFTBase/VSegmentation.h"
#include "MFTBase/ChipSegmentation.h"

namespace o2
{
namespace mft
{

class LadderSegmentation : public VSegmentation
{

 public:
  LadderSegmentation();
  LadderSegmentation(UInt_t uniqueID);
  LadderSegmentation(const LadderSegmentation& ladder);

  ~LadderSegmentation() override
  {
    if (mChips) {
      mChips->Delete();
      delete mChips;
      mChips = nullptr;
    }
  }
  void print(Option_t* opt = "");
  void Clear(const Option_t* /*opt*/) override
  {
    if (mChips) {
      mChips->Clear();
    }
  }

  ChipSegmentation* getSensor(Int_t sensor) const;

  void createSensors(TXMLEngine* xml, XMLNodePointer_t node);

  /// \brief Returns number of Sensor on the ladder
  Int_t getNSensors() const { return mNSensors; };
  /// \brief Set number of Sensor on the ladder
  void setNSensors(Int_t val) { mNSensors = val; };

  ChipSegmentation* getChip(Int_t chipNumber) const { return getSensor(chipNumber); };

 private:
  Int_t mNSensors;      ///< \brief Number of Sensors holded by the ladder
  TClonesArray* mChips; ///< \brief Array of pointer to ChipSegmentation

  ClassDefOverride(LadderSegmentation, 1);
};
} // namespace mft
} // namespace o2

#endif
