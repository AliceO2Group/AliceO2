// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// @brief Class for mapping between DAQ sourceIDs and O2 DataOrigins
// @author ruben.shahoyan@cern.ch

#ifndef DETECTOR_BASE_RAWDAQID_H
#define DETECTOR_BASE_RAWDAQID_H

#include "Headers/DataHeader.h"

namespace o2
{
namespace header
{
/// Source IDs used by DAQ

class DAQID
{

 public:
  typedef std::uint8_t ID;

  static constexpr ID TPC = 3;
  static constexpr ID TRD = 4;
  static constexpr ID TOF = 5;
  static constexpr ID HMP = 6;
  static constexpr ID PHS = 7;
  static constexpr ID CPV = 8;
  static constexpr ID INVALID = 9; // 1st invalid slot starting from meaningful values
  static constexpr ID MCH = 10;
  static constexpr ID ZDC = 15;
  static constexpr ID TRG = 17;
  static constexpr ID EMC = 18;
  static constexpr ID TST = 19;
  static constexpr ID ITS = 32;
  static constexpr ID FDD = 33;
  static constexpr ID FT0 = 34;
  static constexpr ID FV0 = 35;
  static constexpr ID MFT = 36;
  static constexpr ID MID = 37;
  static constexpr ID DCS = 38;
  static constexpr ID FOC = 39;
  static constexpr ID MINDAQ = TPC;
  static constexpr ID MAXDAQ = FOC;

  DAQID() : mID(INVALID) {}
  DAQID(ID i) : mID(i) {}
  operator ID() const { return static_cast<ID>(mID); }

  ID getID() const { return mID; }
  constexpr o2::header::DataOrigin getO2Origin() { return DAQtoO2(mID); }

  static constexpr o2::header::DataOrigin DAQtoO2(ID daq)
  {
    return (daq < MAXDAQ + 1) ? MAP_DAQtoO2[daq] : MAP_DAQtoO2[INVALID];
  }

  static constexpr ID O2toDAQ(o2::header::DataOrigin o2orig)
  {
    return or2daq(o2orig, MINDAQ);
  }

 private:
  ID mID = INVALID;

  static constexpr o2::header::DataOrigin MAP_DAQtoO2[] = {
    "NIL", "NIL", "NIL",
    "TPC", "TRD", "TOF", "HMP", "PHS", "CPV",
    "NIL",
    "MCH",
    "NIL", "NIL", "NIL", "NIL",
    "ZDC",
    "NIL",
    "TRG", "EMC", "TST",
    "NIL", "NIL", "NIL", "NIL", "NIL", "NIL", "NIL", "NIL", "NIL", "NIL", "NIL", "NIL",
    "ITS", "FDD", "FT0", "FV0", "MFT", "MID", "DCS", "FOC"};

  static constexpr ID or2daq(o2::header::DataOrigin origin, ID id)
  {
    return id > MAXDAQ ? INVALID : (origin == MAP_DAQtoO2[id] ? id : or2daq(origin, id + 1));
  }
};

} // namespace header
} // namespace o2

#endif
