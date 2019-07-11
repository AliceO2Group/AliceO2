// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HalfSegmentation.h
/// \brief Segmentation class for each half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFSEGMENTATION_H_
#define ALICEO2_MFT_HALFSEGMENTATION_H_

#include "TNamed.h"
#include "TXMLEngine.h"

#include "MFTBase/Segmentation.h"
#include "MFTBase/VSegmentation.h"

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

class HalfSegmentation : public VSegmentation
{

 public:
  HalfSegmentation();
  HalfSegmentation(const Char_t* initFile, const Short_t id);
  HalfSegmentation(const HalfSegmentation& source);

  ~HalfSegmentation() override;
  void Clear(const Option_t* /*opt*/) override;

  // Bool_t getID() const { return (GetUniqueID() >> 12); };

  Int_t getNHalfDisks() const { return mHalfDisks->GetEntries(); }

  HalfDiskSegmentation* getHalfDisk(Int_t iDisk) const
  {
    if (iDisk >= 0 && iDisk < mHalfDisks->GetEntries())
      return (HalfDiskSegmentation*)mHalfDisks->At(iDisk);
    else
      return nullptr;
  }

 private:
  void findHalf(TXMLEngine* xml, XMLNodePointer_t node, XMLNodePointer_t& retnode);
  void createHalfDisks(TXMLEngine* xml, XMLNodePointer_t node);

  TClonesArray* mHalfDisks; ///< \brief Array of pointer to HalfDiskSegmentation

  ClassDefOverride(HalfSegmentation, 1);
};
} // namespace mft
} // namespace o2

#endif
