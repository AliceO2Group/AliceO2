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

/// \file RUDecodeData.h
/// \brief Declaration of the Readout Unite decoder class
#ifndef ALICEO2_ITSMFT_RUDECODEDATA_H_
#define ALICEO2_ITSMFT_RUDECODEDATA_H_

#include <array>
#include <memory>
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/RUInfo.h"
#include "ITSMFTReconstruction/AlpideCoder.h"
#include "DataFormatsITSMFT/GBTCalibData.h"

namespace o2
{
namespace itsmft
{
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
  std::array<GBTLink*, MaxCablesPerRU> cableLinkPtr;       // Ptr of the GBT link transmitting this cable data
  std::unordered_map<uint64_t, uint32_t> linkHBFToDump;    // FEEID<<32+hbfEntry to dump in case of error
  int ruSWID = -1;                                         // SW (stave) ID
  int nChipsFired = 0;     // number of chips with data or with errors
  int lastChipChecked = 0; // last chips checked among nChipsFired
  int nNonEmptyLinks = 0;  // number of non-empty links for current ROF
  int nLinks = 0;          // number of links seen for this TF
  int nLinksDone = 0;      // number of links finished for this TF
  int verbosity = 0;       // verbosity level, for -1,0 print only summary data, for 1: print once every error
  GBTCalibData calibData{}; // calibration info from GBT calibration word
  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> chipErrorsTF; // vector of chip decoding errors seen in the given TF
  const RUInfo* ruInfo = nullptr;

  RUDecodeData()
  {
    memset(&links[0], -1, MaxLinksPerRU * sizeof(int));
  }
  void clear();
  void setROFInfo(ChipPixelData* chipData, const GBTLink* lnk);
  template <class Mapping>
  int decodeROF(const Mapping& mp, const o2::InteractionRecord ir);
  void fillChipStatistics(int icab, const ChipPixelData* chipData);
  void dumpcabledata(int icab);
  bool checkLinkInSync(int icab, const o2::InteractionRecord ir);
  ClassDefNV(RUDecodeData, 2);
};

///_________________________________________________________________
/// decode single readout frame, the cable's data must be filled in advance via GBTLink::collectROFCableData
template <class Mapping>
int RUDecodeData::decodeROF(const Mapping& mp, const o2::InteractionRecord ir)
{
  nChipsFired = 0;
  lastChipChecked = 0;
  int ntot = 0;
  // poll majority ROF IR to detect desynchronization between the liks

  std::array<bool, Mapping::getNChips()> doneChips{};
  auto* chipData = &chipsData[0];
  for (int icab = 0; icab < ruInfo->nCables; icab++) { // cableData is ordered in such a way to have chipIDs in increasing order
    if (!cableData[icab].getSize()) {
      continue;
    }
    if (!checkLinkInSync(icab, ir)) { // apparently there was desynchronization
      continue;
    }
    auto cabHW = cableHWID[icab];
    auto chIdGetter = [this, &mp, cabHW](int cid) {
      //return mp.getGlobalChipID(cid, cabHW, *this->ruInfo);
      auto chip = mp.getGlobalChipID(cid, cabHW, *this->ruInfo);
      return chip;
    };
    int ret = 0;
    // dumpcabledata(icab);

    while ((ret = AlpideCoder::decodeChip(*chipData, cableData[icab], chIdGetter)) || chipData->isErrorSet()) { // we register only chips with hits or errors flags set
      setROFInfo(chipData, cableLinkPtr[icab]);
      auto nhits = chipData->getData().size();
      if (nhits && doneChips[chipData->getChipID()]) {
        if (chipData->getChipID() == chipsData[nChipsFired - 1].getChipID()) {
          LOGP(debug, "re-entry into the data of the chip {} after previously detector error", chipData->getChipID());
        }
#ifdef ALPIDE_DECODING_STAT
        else {
          chipData->setError(ChipStat::InterleavedChipData);
        }
#endif
        ret = -1; // discard decoded data
        nhits = 0;
      }
#ifdef ALPIDE_DECODING_STAT
      fillChipStatistics(icab, chipData);
#endif
      if (nhits && chipData->getChipID() < Mapping::getNChips()) {
        doneChips[chipData->getChipID()] = true;
        ntot += nhits;
        if (++nChipsFired < chipsData.size()) { // fetch next free chip
          chipData = &chipsData[nChipsFired];
        } else {
          break; // last chip decoded
        }
      }
      if (ret < 0) {
        break; // negative code was returned by decoder: abandon cable data
      }
    }
    cableData[icab].clear();
  }

  return ntot;
}

} // namespace itsmft
} // namespace o2

#endif
