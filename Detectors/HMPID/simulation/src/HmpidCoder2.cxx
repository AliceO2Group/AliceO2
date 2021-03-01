// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"

#include "HMPIDBase/Digit.h"
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
  mPailoadBufferDimPerEquipment = ((Geo::N_SEGMENTS * (Geo::N_COLXSEGMENT * (Geo::N_DILOGICS * (Geo::N_CHANNELS + 1) + 1) + 1)) + 10);
  mPayloadBufferPtr = (uint32_t*)std::malloc(mNumberOfEquipments * sizeof(uint32_t) * mPailoadBufferDimPerEquipment);
  mPadMap = (uint32_t*)std::malloc(sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS);
  // TODO: Add memory allocation error check

}

///  HMPID Raw Coder
HmpidCoder2::~HmpidCoder2()
{
  // TODO Auto-generated destructor stub
  std::free(mPayloadBufferPtr);
  std::free(mPadMap);
}

///  getEquipCoord() : converts the EquipmentID in CRU,Link couple
/// @param[in] Equi : the HMPID Equipment ID [0..13]
/// @param[out] CruId : the FLP CRU number [0..3]
/// @param[ou] LinkId : the FLP Linkk number [0..3]
void HmpidCoder2::getEquipCoord(int Equi, uint32_t* CruId, uint32_t* LinkId)
{
  for (int i = 0; i < Geo::MAXEQUIPMENTS; i++) {
    if (mEqIds[i] == Equi) {
      *CruId = mCruIds[i];
      *LinkId = mLinkIds[i];
      return;
    }
  }
  *CruId = mCruIds[0];
  *LinkId = mLinkIds[0];
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
    LOG(DEBUG) << "writePaginatedEvent()  Eq=" << eq << " Size:" << EventSize << " Pads:" << mEventPadsPerEquipment[eq] << " Orbit:" << orbit << " BC:" << bc;
    if (EventSize == 0 && mSkipEmptyEvents) {
      continue; // Skips the Events sized with 0
    }
    mWriter.addData(mEqIds[eq], mCruIds[eq], mLinkIds[eq], 0, {bc, orbit}, gsl::span<char>(reinterpret_cast<char*>(ptrStartEquipment), EventSize * sizeof(uint32_t)));
    ptrStartEquipment += EventSize;
  }
  return;
}

/// Analyze a Digits Vector and setup the PADs array
/// with the charge value, then fills the output buffer
/// and forward it to the RawWriter object
///
/// NOTE: this version take the Trigger info from the first
///       digit in the vector. We ASSUME that the vector contains
///       one and only one event !!!!
/// @param[in] digits : the vector of Digit structures
void HmpidCoder2::codeEventChunkDigits(std::vector<Digit>& digits)
{
  int eq, col, dil, cha, mo, x, y, idx;
  uint32_t orbit = 0;
  uint16_t bc = 0;

  int padsCount = 0;
  if(digits.size() == 0) return; // the vector is empty !

  orbit = digits[0].getOrbit();
  bc = digits[0].getBC();
  LOG(INFO) << "Manage chunk Orbit :" << orbit << " BC:" << bc;
  for (o2::hmpid::Digit d : digits) {
    Digit::Pad2Equipment(d.getPadID(), &eq, &col, &dil, &cha); // From Digit to Hardware coords
    eq = mEqIds[eq];                                           // converts the Equipment Id in Cru/Link position ref
    idx = getEquipmentPadIndex(eq, col, dil, cha);             // finally to the unique padmap index
    if (mPadMap[idx] != 0) { // We already have the pad set
      std::cerr << "HmpidCoder [ERROR] : Duplicated DIGIT =" << d << " (" << eq << "," << col << "," << dil << "," << cha << ")" << std::endl;
    } else {
      mPadMap[idx] = d.getCharge();
      padsCount++;
    }
  }
  fillTheOutputBuffer(mPadMap); // Fill the Buffer for all Equipments per Event
  writePaginatedEvent(orbit, bc);
  memset(mPadMap, 0, sizeof(uint32_t) * Geo::N_HMPIDTOTALPADS); // Update for the new event
  return;
}

/// Create the Raw File/Files for the output.
/// Also registers the links in the RawWriter object
///
/// @param[in] OutputFileName : the Path/Prefix name for the raw files
/// @param[in] perFlpFile : if true a couple of files will be created, one for each
///                         HMPID FLPs
void HmpidCoder2::openOutputStream(const char* OutputFileName, bool perFlpFile)
{
  if (perFlpFile) {
    sprintf(mFileName160, "%s_%d%s", OutputFileName, 160, ".raw");
    sprintf(mFileName161, "%s_%d%s", OutputFileName, 161, ".raw");
  } else {
    sprintf(mFileName160, "%s%s", OutputFileName, ".raw");
    sprintf(mFileName161, "%s%s", OutputFileName, ".raw");
  }
  RAWDataHeader rdh; // by default, v6 is used currently.
  for (int eq = 0; eq < mNumberOfEquipments; eq++) {
    rdh.feeId = mEqIds[eq];
    rdh.cruID = mCruIds[eq];
    rdh.linkID = mLinkIds[eq];
    rdh.endPointID = 0;
    if(mFlpIds[eq] == 160) {
      mWriter.registerLink(rdh, mFileName160);
    } else {
      mWriter.registerLink(rdh, mFileName161);
    }
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
void HmpidCoder2::dumpResults()
{
  std::cout << " ****  HMPID RawFile Coder : results ****" << std::endl;
  std::cout << " Created files : " << mFileName160 << " ," << mFileName161 << std::endl;
  std::cout << " Number of Pads coded : " << mPadsCoded << std::endl;
  std::cout << " ----------------------------------------" << std::endl;
}
