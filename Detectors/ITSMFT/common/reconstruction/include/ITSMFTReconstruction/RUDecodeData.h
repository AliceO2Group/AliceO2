// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RUDecodeData.h
/// \brief Declaration of the Readout Unite decoder class
#ifndef ALICEO2_ITSMFT_RUDECODEDATA_H_
#define ALICEO2_ITSMFT_RUDECODEDATA_H_

#include <array>
#include <memory>
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/AlpideCoder.h"

namespace o2
{
namespace itsmft
{
struct RUInfo;
struct GBTLink;

struct RUDecodeData {

  static constexpr int MaxCablesPerRU = 28; // max number of cables RU can readout
  static constexpr int MaxChipsPerRU = 196; // max number of chips the RU can readout
  static constexpr int MaxLinksPerRU = 3;   // max number of GBT links per RU

  std::array<PayLoadCont, MaxCablesPerRU> cableData;       // cable data in compressed ALPIDE format
  std::vector<o2::itsmft::ChipPixelData> chipsData;        // fully decoded data in 1st nChipsFired chips
  std::array<int, MaxLinksPerRU> links;                    // link entry RSTODO: consider removing this and using pointer
  std::array<uint8_t, MaxCablesPerRU> cableHWID;           // HW ID of cable whose data is in the corresponding slot of cableData
  std::array<uint8_t, MaxCablesPerRU> cableLinkID;         // ID of the GBT link transmitting this cable data
  std::array<const GBTLink*, MaxCablesPerRU> cableLinkPtr; // Ptr of the GBT link transmitting this cable data

  int ruSWID = -1;         // SW (stave) ID
  int nCables = 0;         // total number of cables decoded for single trigger
  int nChipsFired = 0;     // number of chips with data or with errors
  int lastChipChecked = 0; // last chips checked among nChipsFired
  const RUInfo* ruInfo = nullptr;

  RUDecodeData()
  {
    memset(&links[0], -1, MaxLinksPerRU * sizeof(int));
  }

  int decodeROF();
  void clear();
  void setROFInfo(ChipPixelData* chipData, const GBTLink* lnk);
  template <class Mapping>
  int decodeROF(const Mapping& mp);

  ClassDefNV(RUDecodeData, 1);
};

///_________________________________________________________________
/// decode single readout frame, the cable's data must be filled in advance via GBTLink::collectROFCableData
template <class Mapping>
int RUDecodeData::decodeROF(const Mapping& mp)
{
  nChipsFired = 0;
  lastChipChecked = 0;
  int ntot = 0;
  auto* chipData = &chipsData[0];
  for (int icab = 0; icab < nCables; icab++) {
    if (!cableData[icab].getSize()) {
      continue;
    }
    int nhits = 0;
    while ((nhits = AlpideCoder::decodeChip(*chipData, cableData[icab]))) { // we register only chips with hits or errors flags set
      if (nhits > 0) {
        // convert HW chip id within the module to absolute chip id
        chipData->setChipID(mp.getGlobalChipID(chipData->getChipID(), cableHWID[icab], *ruInfo));
        setROFInfo(chipData, cableLinkPtr[icab]);
        ntot += nhits;
        if (++nChipsFired < chipsData.size()) { // fetch next free chip
          chipData = &chipsData[nChipsFired];
        } else {
          break; // last chip decoded
        }
      }
    }
  }

  return ntot;
}

} // namespace itsmft
} // namespace o2

#endif
