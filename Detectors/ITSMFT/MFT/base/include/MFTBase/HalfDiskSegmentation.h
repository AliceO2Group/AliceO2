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

/// \file HalfDiskSegmentation.h
/// \brief Class for the description of the structure of a half-disk
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFDISKSEGMENTATION_H_
#define ALICEO2_MFT_HALFDISKSEGMENTATION_H_

#include "TXMLEngine.h"

#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/VSegmentation.h"

class TClonesArray;

namespace o2
{
namespace mft
{

class HalfDiskSegmentation : public VSegmentation
{

 public:
  HalfDiskSegmentation();
  HalfDiskSegmentation(UInt_t uniqueID);
  HalfDiskSegmentation(const HalfDiskSegmentation& pt);

  ~HalfDiskSegmentation() override;

  void Clear(const Option_t* /*opt*/) override;

  void print(Option_t* opt = "");

  void createLadders(TXMLEngine* xml, XMLNodePointer_t node);

  /// \brief Get the number of Ladder on the Half-Disk really constructed
  Int_t getNLaddersBuild() const { return mLadders->GetEntriesFast(); };

  /// \brief Get the number of Ladder on the Half-Disk
  Int_t getNLadders() const { return mNLadders; };

  /// \brief Set the number of Ladder on the Half-Disk
  void setNLadders(Int_t val) { mNLadders = val; };

  /// \brief Returns pointer to the ladder segmentation object
  /// \param iLadder Int_t : ladder number on the Half-Disk
  LadderSegmentation* getLadder(Int_t iLadder)
  {
    return ((iLadder >= 0 && iLadder < getNLadders()) ? (LadderSegmentation*)mLadders->At(iLadder) : nullptr);
  }

  /// \brief Returns the Z position of the half-disk
  Double_t getZ() const
  {
    const Double_t* pos = getTransformation()->GetTranslation();
    return pos[2];
  };

  Int_t getNChips();

 private:
  Int_t mNLadders; ///< \brief Number of ladder holded by the half-disk

  TClonesArray* mLadders; ///< \brief Array of pointer to LadderSegmentation

  ClassDefOverride(HalfDiskSegmentation, 1);
};
} // namespace mft
} // namespace o2

#endif
