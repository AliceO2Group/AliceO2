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

///
/// \file   HmpidCoder.cxx
/// \author Antonio Franco - INFN Bari
/// \brief Base Class for coding HMPID Raw Data File
/// \version 1.0
/// \date 24 feb 2021

#include <vector>
#include <iostream>
#include <memory>

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DataFormatsParameters/GRPObject.h"

#include "DataFormatsHMP/Digit.h"
#include "HMPIDSimulation/HmpidCoder2.h"

using namespace o2::raw;
using namespace o2::hmpid;
using namespace o2::header;

///  HMPID Raw Coder Constructor
/// @param[in] numOfEquipments : number of Equipments
HmpidCoder2::HmpidCoder2(int numOfEquipments)
{
  mPadsCoded = 0;
  mNumberOfEquipments = numOfEquipments;
  mVerbose = 0;
  mSkipEmptyEvents = true;
  mPayloadBufferDimPerEquipment = ((Geo::N_SEGMENTS * (Geo::N_COLXSEGMENT * (Geo::N_DILOGICS * (Geo::N_CHANNELS + 1) + 1) + 1)) + 10);
  mUPayloadBufferPtr = std::make_unique<uint32_t[]>(mNumberOfEquipments * mPayloadBufferDimPerEquipment);
  mUPadMap = std::make_unique<uint32_t[]>(Geo::N_HMPIDTOTALPADS);
  mPayloadBufferPtr = mUPayloadBufferPtr.get();
  mPadMap = mUPadMap.get();
  std::memset(mPadMap, 0, sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS); // Zero the map for the first event
  mBusyTime = 20000; // 1 milli sec
  mHmpidErrorFlag = 0;
  mHmpidFrwVersion = 9;
}

/// setDetectorSpecificFields() : sets the HMPID parameters for the next
/// raw file writes
/// @param[in] BusyTime : busy time in milliseconds
/// @param[in] Error : the Error field
/// @param[in] Version : the Firmware Version  [def. 9]
void HmpidCoder2::setDetectorSpecificFields(float BusyTime, int Error, int Version)
{
  uint32_t busy = (uint32_t)(BusyTime / 0.00000005);
  mBusyTime = busy;
  mHmpidErrorFlag = Error;
  mHmpidFrwVersion = Version;
  return;
}

/// setRDHFields() : sets the HMPID RDH Field for the next
/// raw file writes
/// @param[in] eq : the HMPID Equipment ID [0..13] if == -1 -> all
void HmpidCoder2::setRDHFields(int eq)
{
  int st, en;
  uint32_t wr = (mBusyTime << 9) | ((mHmpidErrorFlag & 0x01F) << 4) | (mHmpidFrwVersion & 0x0F);
  st = (eq < 0 || eq >= Geo::MAXEQUIPMENTS) ? 0 : eq;
  en = (eq < 0 || eq >= Geo::MAXEQUIPMENTS) ? Geo::MAXEQUIPMENTS : eq + 1;
  for (int l = st; l < en; l++) {
    o2::raw::RawFileWriter::LinkData& link = mWriter.getLinkWithSubSpec(mTheRFWLinks[l]);
    RDHAny* RDHptr = link.getLastRDH();
    if (RDHptr != nullptr) {
      o2::raw::RDHUtils::setDetectorField(RDHptr, wr);
    }
  }
  return;
}

/// constexpr to accelerate the coordinates changing
constexpr int p1() { return (Geo::N_SEGMENTS * Geo::N_COLXSEGMENT * Geo::N_DILOGICS * Geo::N_CHANNELS); }
constexpr int p2() { return (Geo::N_DILOGICS * Geo::N_CHANNELS); }

/// getEquipmentPadIndex() : converts the (Equipment, Column, Dilogic, Channel)
/// coordinate into a unique PadIndex value used to address the PADs array
/// @param[in] eq : the HMPID Equipment ID [0..13]
/// @param[in] col : the Equipment Column [0..23]
/// @param[in] dil : the Dilogic [0..9]
/// @param[in] cha : the Channel [0..47]
/// @returns The PAD index value [0..161279]
int HmpidCoder2::getEquipmentPadIndex(int eq, int col, int dil, int cha)
{
  return (eq * p1() + col * p2() + dil * Geo::N_CHANNELS + cha);
}

/// Scans the PADs array and fill the Output buffer with the RawFile structure
/// a two step algorithm...
/// @param[in] padMap : poiter to the PADs map array
void HmpidCoder2::fillTheOutputBuffer(uint32_t* padMap)
{
  uint32_t rowMarker, segMarker, eoeMarker, padWord;
  uint32_t rowSize;
  uint32_t ptr = 0;
  int pads[Geo::MAXEQUIPMENTS];
  int padsCount;
  int segSize;

  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++) {
    mEventSizePerEquipment[i] = 0;
  }

  for (int eq = 0; eq < mNumberOfEquipments; eq++) {
    int startPtr = ptr;
    padsCount = 0;
    for (int s = 1; s <= Geo::N_SEGMENTS; s++) {
      segSize = 0;
      for (int c = 1; c <= Geo::N_COLXSEGMENT; c++) {
        // ---- Pre-calculate the size of each column
        for (int j = 0; j < Geo::N_DILOGICS; j++) {
          pads[j] = 0;
        }
        rowSize = 0;
        for (int j = 0; j < Geo::N_DILOGICS; j++) {
          for (int k = 0; k < Geo::N_CHANNELS; k++) {
            int idx = getEquipmentPadIndex(eq, ((s - 1) * Geo::N_COLXSEGMENT + (c - 1)), j, k);
            if (padMap[idx] > 0) {
              pads[j]++;
              rowSize++;
              padsCount++;
            }
          }
        }
        rowSize += Geo::N_DILOGICS;
        segSize += (rowSize + 1);
        rowMarker = 0x000036A8 | ((rowSize << 16) & 0x03ff0000);

        // ---- fills the Payload Buffer
        mPayloadBufferPtr[ptr++] = rowMarker;
        int col = (s - 1) * Geo::N_COLXSEGMENT + c;
        for (int d = 1; d <= Geo::N_DILOGICS; d++) {
          for (int p = 0; p < Geo::N_CHANNELS; p++) {
            int idx = getEquipmentPadIndex(eq, ((s - 1) * Geo::N_COLXSEGMENT + (c - 1)), (d - 1), p);
            if (padMap[idx] > 0) {
              padWord = ((col << 22) & 0x07c00000) | ((d << 18) & 0x003C0000) | ((p << 12) & 0x0003F000) | (padMap[idx] & 0x00000FFF);
              mPayloadBufferPtr[ptr++] = padWord;
            }
          }
          eoeMarker = 0x08000080 | ((col << 22) & 0x07c00000) | (d << 18 & 0x003C0000) | (pads[d - 1] & 0x0000007F);
          mPayloadBufferPtr[ptr++] = eoeMarker;
        }
      }
      segSize += 1;
      segMarker = 0xAB000000 | ((segSize << 8) & 0x000fff00) | (s & 0x0000000F);
      mPayloadBufferPtr[ptr++] = segMarker;
    }
    mPadsCoded += padsCount;
    mEventPadsPerEquipment[eq] = padsCount;
    mEventSizePerEquipment[eq] = ptr - startPtr;
  }
  return;
}

/// Add a chunk of data in the Output buffer to the RawWriter
/// setting the CRU,Link coordinates and the Trigger Info
/// One or more Pages will be created for each equipment
///
/// @param[in] orbit : the Trigger ORBIT value
/// @param[in] bc : the Trigger BC value
void HmpidCoder2::writePaginatedEvent(uint32_t orbit, uint16_t bc)
{
  uint32_t* ptrStartEquipment = mPayloadBufferPtr;
  for (int eq = 0; eq < mNumberOfEquipments; eq++) {
    int EventSize = mEventSizePerEquipment[eq];
    LOG(debug) << "writePaginatedEvent()  Eq=" << eq << " Size:" << EventSize << " Pads:" << mEventPadsPerEquipment[eq] << " Orbit:" << orbit << " BC:" << bc;
    if (mEventPadsPerEquipment[eq] > 0 || !mSkipEmptyEvents) { // Skips the Events with 0 Pads
      mWriter.addData(ReadOut::FeeId(eq),
                      ReadOut::CruId(eq),
                      ReadOut::LnkId(eq),
                      0,
                      {bc, orbit},
                      gsl::span<char>(reinterpret_cast<char*>(ptrStartEquipment),
                                      EventSize * sizeof(uint32_t)),
                      false,
                      0,
                      (uint32_t)((mBusyTime << 9) | ((mHmpidErrorFlag & 0x01F) << 4) | (mHmpidFrwVersion & 0x0F)));
      // We fill the fields !
      // TODO: we can fill the detector field with Simulated Data
      setDetectorSpecificFields(0.000001 * EventSize);
      setRDHFields(eq);
    }
    ptrStartEquipment += EventSize;
  }
  return;
}

/// Analyze a Digits Vector and setup the PADs array
/// with the charge value, then fills the output buffer
/// and forward it to the RawWriter object
///
/// NOTE: the vector could be empty!
/// @param[in] digits : the vector of Digit structures
/// @param[in] ir : the Interaction Record structure
void HmpidCoder2::codeEventChunkDigits(std::vector<o2::hmpid::Digit>& digits, InteractionRecord ir)
{
  int eq, col, dil, cha, mo, x, y, idx;
  uint32_t orbit = ir.orbit;
  uint16_t bc = ir.bc;

  int padsCount = 0;
  LOG(debug) << "Manage chunk Orbit :" << orbit << " BC:" << bc << "  Digits size:" << digits.size();
  for (o2::hmpid::Digit d : digits) {
    Digit::pad2Equipment(d.getPadID(), &eq, &col, &dil, &cha); // From Digit to Hardware coords
    eq = ReadOut::FeeId(eq);                                   // converts the Equipment Id in Cru/Link position ref
    idx = getEquipmentPadIndex(eq, col, dil, cha);             // finally to the unique padmap index
    if (mPadMap[idx] != 0) {                                   // We already have the pad set
      LOG(warning) << "Duplicated DIGIT =" << d << " (" << eq << "," << col << "," << dil << "," << cha << ")" << idx;
    } else {
      mPadMap[idx] = d.getCharge();
      padsCount++;
    }
  }
  fillTheOutputBuffer(mPadMap); // Fill the Buffer for all Equipments per Event
  writePaginatedEvent(orbit, bc);
  std::memset(mPadMap, 0, sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS); // Update for the new event
  return;
}

/// Create the Raw File/Files for the output.
/// Also registers the links in the RawWriter object
///
/// @param[in] OutputFileName : the Path/Prefix name for the raw files
/// @param[in] perFlpFile : if true a couple of files will be created, one for each
///                         HMPID FLPs
void HmpidCoder2::openOutputStream(const std::string& outputFileName, const std::string& fileFor)
{
  RAWDataHeader rdh; // by default, v6 is used currently.
  for (int eq = 0; eq < mNumberOfEquipments; eq++) {
    rdh.feeId = ReadOut::FeeId(eq);
    rdh.cruID = ReadOut::CruId(eq);
    rdh.linkID = ReadOut::LnkId(eq);
    rdh.endPointID = 0;
    std::string outfname;
    if (fileFor == "link") {
      outfname = fmt::format("{}_{}_feeid{}.raw", outputFileName, ReadOut::FlpHostName(eq), int(rdh.feeId));
    } else if (fileFor == "flp") {
      outfname = fmt::format("{}_{}.raw", outputFileName, ReadOut::FlpHostName(eq));
    } else if (fileFor == "all") {
      outfname = fmt::format("{}.raw", outputFileName);
    } else if (fileFor == "crorcendpoint") {
      outfname = fmt::format("{}_{}_crorc{}_{}.raw", outputFileName, ReadOut::FlpHostName(eq), int(rdh.cruID), int(rdh.linkID));
    } else {
      throw std::runtime_error(fmt::format("unknown raw file grouping option {}", fileFor));
    }

    mWriter.registerLink(rdh, outfname); // register the link
    LinkSubSpec_t ap = RDHUtils::getSubSpec(ReadOut::CruId(eq), ReadOut::LnkId(eq), 0, ReadOut::FeeId(eq));
    mTheRFWLinks[eq] = ap; // Store the RawFileWriter Link ID
  }
  return;
}

/// Close and flush the output streams.
void HmpidCoder2::closeOutputStream()
{
  mWriter.close();
  return;
}

/// Dumps the results of the last coding
void HmpidCoder2::dumpResults(const std::string& outputFileName)
{
  std::cout << " ****  HMPID RawFile Coder : results ****" << std::endl;
  std::cout << " Created files : " << outputFileName << std::endl;
  std::cout << " Number of Pads coded : " << mPadsCoded << std::endl;
  std::cout << " ----------------------------------------" << std::endl;
}
