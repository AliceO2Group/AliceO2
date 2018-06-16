// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RawPixelReader.h
/// \brief Definition of the Alpide pixel reader for raw data processing
#ifndef ALICEO2_ITSMFT_RAWPIXELREADER_H
#define ALICEO2_ITSMFT_RAWPIXELREADER_H

#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/AlpideCoder.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTBase/Digit.h"
#include "CommonUtils/RootChain.h"
#include <TTree.h>
#include <TStopwatch.h>
#include <FairLogger.h>
#include <vector>
#include <limits>
#include <memory>
#include <algorithm>
#include <cassert>

//#define _RAW_READER_DEBUG_

namespace o2
{
namespace ITSMFT
{
/// Used both for encoding to and decoding from the alpide raw data format
/// Requires as a template parameter a helper class for detector-specific
/// mapping between the software global chip ID and HW module ID and chip ID
/// within the module, see for example ChipMappingITS class.
/// Similar helper class must be provided for the MFT

template <class Mapping = o2::ITSMFT::ChipMappingITS>
class RawPixelReader : public PixelReader
{
  using Coder = o2::ITSMFT::AlpideCoder;

 public:
  RawPixelReader() = default;
  ~RawPixelReader() override = default;

  /// module IDs to process
  int getModuleMin() const { return mModuleMin; }

  int getModuleMax() const { return mModuleMax; }

  /// set min/max modules to process
  void setModuleMinMax(int mn = 0, int mx = -1)
  {
    mModuleMin = mn < 0 ? 0 : (mn < mMapping.getNModules() ? mn : mMapping.getNModules() - 1);
    mModuleMax = mx < 0 ? mMapping.getNModules() - 1 : (mx < mMapping.getNModules() ? mx : mMapping.getNModules() - 1);
    assert(mModuleMin <= mModuleMax);
  }

  /// open raw data input file
  void openInput(const std::string rawInput)
  {
    mCoder.openInput(rawInput);
  }

  /// read single chip data and convert to row/col format in the provided array
  bool getNextChipData(ChipPixelData& chipData) override
  {
    chipData.clear();
    int res = 0;
    // get Alpide-compressed data for next non-empty chip
    while (!(res = mCoder.readChipData(mHitsRecordChip, mCurrChipInModule, mCurrModule, mCurrROF))) {
    }
    if (res < 0) { // EOF reached
      return false;
    }
    return decodeAlpideHitsFormat(mHitsRecordChip, chipData);
  }

  /// read/decode data to ChipPixelData slot of the vector with index of the chip
  ChipPixelData* getNextChipData(std::vector<ChipPixelData>& chipDataVec) override
  {
    int res = 0;
    while (!(res = mCoder.readChipData(mHitsRecordChip, mCurrChipInModule, mCurrModule, mCurrROF))) {
    }
    if (res < 0) { // EOF reached
      return nullptr;
    }
    ChipPixelData& chipData = chipDataVec[mMapping.module2ChipID(mCurrModule, mCurrChipInModule)];
    chipData.clear();
    decodeAlpideHitsFormat(mHitsRecordChip, chipData);

    //chipData.print();

    return &chipData;
  }

  /// convert digits from the Digits branch of digTree to raw ALPIDE data
  void convertDigits2Raw(std::string outName, std::string inpName, std::string digTreeName, std::string digBranchName,
                         std::size_t evFirst = 0, std::size_t evLast = std::numeric_limits<std::size_t>::max())
  {
    TStopwatch swTot;

    std::unique_ptr<TTree> inpTree = o2::utils::RootChain::load(digTreeName, inpName);
    if (!inpTree) {
      LOG(FATAL) << "Failed to find the tree " << digTreeName << " in the input " << inpName << FairLogger::endl;
    }
    std::vector<o2::ITSMFT::Digit>* digiVec = nullptr;

    inpTree->SetBranchAddress(digBranchName.data(), &digiVec);
    if (!digiVec) {
      LOG(FATAL) << "Failed to load digits branch " << digBranchName << " from the tree " << digTreeName
                 << " of the input " << inpName << FairLogger::endl;
    }
    if (outName.empty()) {
      outName = "raw" + digBranchName + ".raw";
      LOG(INFO) << "Output file name is not provided, set to " << outName << FairLogger::endl;
    }

    if (inpTree->GetEntries() < evLast) {
      evLast = inpTree->GetEntries() - 1;
    }

    mCoder.openOutput(outName);
    o2::ITSMFT::ChipPixelData chipData;
    chipData.setROFrame(0);
    chipData.setChipID(0);
    int counterROF = 0, counterTot = 0, shiftROF = 0;
    PrevChipDone prev; // previously processed data info

    for (auto ev = evFirst; ev <= evLast; ev++) {
      inpTree->GetEntry(ev);
      for (const auto& dig : *digiVec) {
        int modID = mMapping.chipID2Module(dig.getChipIndex());
        if (modID < mModuleMin || modID > mModuleMax) { // skip unwanted modules
          continue;
        }
        int rof = dig.getROFrame() + shiftROF;
        if (rof != chipData.getROFrame() || dig.getChipIndex() != chipData.getChipID()) {
          if (rof != chipData.getROFrame()) { // new frame
            counterTot += counterROF;
            LOG(INFO) << "Converted " << counterROF << " digits in ROFrame " << chipData.getROFrame()
                      << " (" << counterTot << " in total)" << FairLogger::endl;
            if (rof < chipData.getROFrame()) { // ROFs should be in the increasing order
              // this situation is possible if we work with the chian of the digits, each element of the
              // chain starting from 0. We will introduce a shift in the ROFs to avoid ups and downs
              shiftROF = chipData.getROFrame() + 1;
              LOG(WARNING) << "New ROF=" << dig.getROFrame() << " is inferior to last ROF=" << chipData.getROFrame() << " following ROFs will be shifter by " << shiftROF << FairLogger::endl;
            }
            counterROF = 0;
          }
          convertChipDigits(&chipData, prev);
          chipData.clear();
          chipData.setROFrame(dig.getROFrame() + shiftROF);
          chipData.setChipID(dig.getChipIndex());
        }
        o2::ITSMFT::PixelData pix(&dig);
        chipData.getData().push_back(pix);
        counterROF++;
      }
      //
      mCoder.flushBuffer(); // flushing is expensive, don't do this too often
    }

    if (chipData.getData().size()) {
      convertChipDigits(&chipData, prev);
      counterTot += counterROF;
      LOG(INFO) << "Converted " << counterROF << " digits in ROFrame " << chipData.getROFrame()
                << " (last ROF, total hits: " << counterTot << ")" << FairLogger::endl;
    }
    convertChipDigits(nullptr, prev); // to close the streem
    mCoder.closeIO();
    swTot.Stop();
    swTot.Print();
    delete digiVec;
  }

  // dummy method to comply with base class requirements
  void init() override
  {
  }

 private:
  Coder mCoder;                                   /// ALPIDE data format coder/decoder
  std::vector<Coder::HitsRecord> mHitsRecordChip; // chip data in Alpide format

  std::uint16_t mCurrModule = 0;       // currently read module
  std::uint16_t mCurrChipInModule = 0; // currently read chip in the module
  std::uint16_t mCurrROF = 0;          // currently read ROF

  Mapping mMapping;                            /// utility class for global_chip_id <-> module&local_chip_id
  int mModuleMin = 0;                          /// 1st module to account
  int mModuleMax = mMapping.getNModules() - 1; /// last module to account

  /// these container is used only for the per-pixel reading, not recommended
  ChipPixelData mCacheSinglePixelMode; // local buffer for special case of per pixel readout
  int mCacheSinglePixelPointer = 0;    // index of last return hit in the mCacheSinglePixelMode

  struct PrevChipDone { // info about previously processed chip
    int rof = -1;
    int chipID = -1;
    int modID = -1;
    int chipInMod = -1;
    int nChipsInMod = -1;
    bool moduleClosed = true;
  };

  void addEmptyROF(int rof)
  {
    ///< Add empty ROFrame
    for (int md = 0; md < mMapping.getNModules(); md++) {
      addEmptyModule(md, rof);
    }
  }

  void addEmptyModule(int modID, int rof)
  {
    ///< Add empty module data
    if (modID < mModuleMin || modID > mModuleMax) { // process only requested modules
      return;
    }
    int nch = mMapping.getNChipsInModule(modID);
    addModuleHeader(modID, rof);
    for (int ich = 0; ich < nch; ich++) {
      addEmptyChip(mMapping.module2ChipID(modID, ich), rof);
    }
    addModuleTrailer(modID, rof);
  }

  void addModuleTrailer(int modID, int rof)
  {
    ///< Finish module data: add trailer
    mCoder.addModuleTrailer(modID, rof);
  }

  void addModuleHeader(int modID, int rof)
  {
    ///< Start new module data: add header
    mCoder.addModuleHeader(modID, rof);
  }

  void addEmptyChip(int chipID, int rof)
  {
    ///< Add empty chip record
    int chipInMod, modID = mMapping.chipID2Module(chipID, chipInMod);
    mCoder.addEmptyChip(chipInMod, rof);
  }

  void convertChipDigits(o2::ITSMFT::ChipPixelData* pixData, PrevChipDone& prev)
  {
    ///< Convert data of single chip. When called with nullptr, will write empty`
    ///< chip records till the end of the current ROF`
    ///< Note: the order of pixels in pixData is not preserved

    int chipInMod = 0, rof = 0, chipID = 0, modID = 0, nChipsInMod = 0;
    if (pixData) {
      rof = pixData->getROFrame();
      chipID = pixData->getChipID();
      modID = mMapping.chipID2Module(chipID, chipInMod);
      nChipsInMod = mMapping.getNChipsInModule(modID);
    } else {              // finalize last ROF
      rof = prev.rof + 1; // fake ROF to close the previous
    }

    if (prev.chipID != -1 && (rof != prev.rof || modID != prev.modID)) {
      // new readout frame or chip started, finish the current one

      // if new module is started, finish the current one
      if (modID != prev.modID || rof != prev.rof) {
        int goToChip = mMapping.module2ChipID(prev.modID, prev.nChipsInMod); // maxChip_of_module + 1
        for (++prev.chipID; prev.chipID < goToChip; prev.chipID++) {
          addEmptyChip(prev.chipID, prev.rof);
        }
        addModuleTrailer(prev.modID, prev.rof);
        prev.moduleClosed = true;
        prev.chipInMod = -1;
      }
      if (rof > prev.rof) { // close current ROF filling empty modules
        for (++prev.modID; prev.modID < mMapping.getNModules(); prev.modID++) {
          addEmptyModule(prev.modID, prev.rof);
        }
        // fill empty ROFs till new ROF
        for (++prev.rof; prev.rof < rof; prev.rof++) {
          addEmptyROF(prev.rof);
        }
        prev.modID = -1;
      }

      // close empty modules of current ROF up to the new module
      for (++prev.modID; prev.modID < modID; prev.modID++) {
        addEmptyModule(prev.modID, rof);
      }

      // start new chip/module only if there is some data
      if (pixData) {
        if (prev.moduleClosed) { // open new module if there is an input
          addModuleHeader(modID, rof);
          prev.moduleClosed = false;
        }
        // fill empty chips of the module till current one
        for (++prev.chipInMod; prev.chipInMod < chipInMod; prev.chipInMod++) {
          prev.chipID = mMapping.module2ChipID(prev.modID, prev.chipInMod);
          addEmptyChip(prev.chipID, rof);
        }
      }
    }
    //
    if (pixData) {
      if (prev.moduleClosed) {
        addModuleHeader(modID, rof);
        prev.moduleClosed = false;
      }
      addDigits(pixData);
    }
    // sync:
    prev.chipID = chipID;
    prev.modID = modID;
    prev.rof = rof;
    prev.chipInMod = chipInMod;
    prev.nChipsInMod = nChipsInMod;
  }

  /// Add digits for the single chip to the converter
  void addDigits(o2::ITSMFT::ChipPixelData* chipData)
  {
    // sort data in row/col
    auto& pixels = chipData->getData();
    std::sort(pixels.begin(), pixels.end(),
              [](auto lhs, auto rhs) {
                if (lhs.getRow() < rhs.getRow())
                  return true;
                if (lhs.getRow() > rhs.getRow())
                  return false;
                return lhs.getCol() < rhs.getCol();
              });
    int chipInMod;
    int modID = mMapping.chipID2Module(chipData->getChipID(), chipInMod);
    for (const auto& pix : pixels) {
      mCoder.addPixel(pix.getRow(), pix.getCol());
    }
    static int cnt = 0;
    mCoder.finishChipEncoding(chipInMod, chipData->getROFrame());
  }

  /// decode packed ALPIDE data to standard row/col format sorted in col/row
  bool decodeAlpideHitsFormat(const std::vector<Coder::HitsRecord>& records, ChipPixelData& chipData)
  {
    chipData.setROFrame(mCurrROF);
    chipData.setChipID(mMapping.module2ChipID(mCurrModule, mCurrChipInModule));
    //
    int nhitsChip = 0;                              // total hits counter
    int nRightCHits = 0;                            // counter for the hits in the right column of the current double column
    std::uint16_t rightColHits[AlpideCoder::NRows]; // buffer for the accumulation of hits in the right column
    std::uint16_t colDPrev = 0xffff;                // previously processed double column (to dected change of the double column)
    int nrec = records.size();
    for (int ih = 0; ih < nrec; ih++) {
      const auto& hitr = records[ih];

      std::uint16_t row = hitr.address >> 1;
      // abs id of left column in double column
      std::uint16_t colD = (std::uint16_t(hitr.region) * Coder::NDColInReg + std::uint16_t(hitr.dcolumn)) << 1;

      // if we start new double column, transfer the hits accumulated in the right column buffer of prev. double column
      if (colD != colDPrev) {
        colDPrev++;
        for (int ihr = 0; ihr < nRightCHits; ihr++) {
          chipData.getData().emplace_back(rightColHits[ihr], colDPrev);
        }
        colDPrev = colD;
        nRightCHits = 0; // reset the buffer
      }

      bool rightC = ((row & 0x1) ? 1 - (hitr.address & 0x1) : (hitr.address & 0x1)); // true for right column / lalse for left
      int col = colD + rightC;
#ifdef _RAW_READER_DEBUG_
      printf("%04d/%03d ", col, row);
#endif
      // we want to have hits sorted in column/row, so the hits in right column of given double column
      // are first collected in the temporary buffer
      if (rightC) {
        rightColHits[nRightCHits++] = row;
      } else {
        chipData.getData().emplace_back(row, col); // left column hits are added directly to the container
      }
      nhitsChip++;
      if (hitr.hitmap) { // extra hits
#ifdef _RAW_READER_DEBUG_
        printf(" [ ");
#endif
        for (int ip = 0; ip < Coder::HitMapSize; ip++) {
          if (hitr.hitmap & (0x1 << ip)) {
            int addr = hitr.address + ip + 1;
            int rowE = addr >> 1;
            rightC = ((rowE & 0x1) ? 1 - (addr & 0x1) : (addr & 0x1)); // true for right column / lalse for left
            int colE = colD + rightC;
            if (rightC) { // same as above
              rightColHits[nRightCHits++] = rowE;
            } else {
              chipData.getData().emplace_back(rowE, colE); // left column hits are added directly to the container
            }
#ifdef _RAW_READER_DEBUG_
            printf("%04d/%03d ", colE, rowE);
#endif
            nhitsChip++;
          }
        }
#ifdef _RAW_READER_DEBUG_
        printf("]");
#endif
      }
#ifdef _RAW_READER_DEBUG_
      printf("\n");
#endif
    }

    // transfer last right-column buffer
    colDPrev++;
    for (int ihr = 0; ihr < nRightCHits; ihr++) {
      chipData.getData().emplace_back(rightColHits[ihr], colDPrev);
    }

#ifdef _RAW_READER_DEBUG_
    printf("decoded %d hits in %d records of chip %d(%d/%d) at ROF %d\n", nhitsChip, nrec,
           chipData.getChipID(), mCurrModule, mCurrChipInModule, mCurrROF);
#endif

    return true;
  }

  ClassDefOverride(RawPixelReader, 1);
};

} // namespace ITSMFT
} // namespace o2

#endif /* ALICEO2_ITS_RAWPIXELREADER_H */
