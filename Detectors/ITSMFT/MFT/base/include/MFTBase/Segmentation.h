// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Segmentation.h
/// \brief Class for the virtual segmentation of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_SEGMENTATION_H_
#define ALICEO2_MFT_SEGMENTATION_H_

#include "TNamed.h"
#include "TClonesArray.h"

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

class Segmentation
{

 public:
  enum { Bottom,
         Top };

  Segmentation();
  Segmentation(const Char_t* nameGeomFile);

  ~Segmentation();
  void Clear(const Option_t* /*opt*/);

  /// \brief Returns pointer to the segmentation of the half-MFT
  /// \param iHalf Integer : 0 = Bottom; 1 = Top
  /// \return Pointer to a HalfSegmentation
  HalfSegmentation* getHalf(Int_t iHalf) const;

  Int_t getDetElemLocalID(Int_t half, Int_t disk, Int_t ladder, Int_t sensor) const;

  Bool_t hitToPixelID(Double_t xHit, Double_t yHit, Double_t zHit, Int_t half, Int_t disk, Int_t ladder, Int_t sensor,
                      Int_t& xPixel, Int_t& yPixel);

  static constexpr Int_t NumberOfHalves = 2; ///< \brief Number of detector halves

 private:
  TClonesArray* mHalves; ///< \brief Array of pointer to HalfSegmentation

  ClassDef(Segmentation, 1);
};
} // namespace mft
} // namespace o2

#endif
