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
/// \file   Geo.h
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 15/02/2021

#ifndef ALICEO2_HMPID_GEO_H
#define ALICEO2_HMPID_GEO_H

#include <TMath.h>
<<<<<<< HEAD

namespace o2
{
namespace hmpid
{

/*  ------------------ HMPID Detector Coordinate definition -------------------

  143 .-------------.             .-------------.          0  ------.------   23
      |             |             |.-----.-----.|            |      |  <-- |
      |             |             ||     |     ||            |      | 9  0 |
      |             |             ||  4  |  5  ||            |      | Dil. |
    ^ |             |             ||     |     ||          | |      |      |  ^
    | |             |             ||_____|_____||          | |      |      |  |
    | |             |             |.-----.-----.|          | |      |      |  |
    Y |             |             ||     |     ||            |      |      |
    | |             |             ||  2  |  3  ||     Column |  n   |  n+1 | Column
    | |             |             ||     |     ||            |      |      |
      |             |             ||_____|_____||          | |      |      |  |
      |             |       47  ^ |.-----.-----.|          | |      |      |  |
      |             |           | ||     |     ||          V |      |      |  |
      |             |           Y ||  0  |  1  ||            | Dil. |      |
      |             |           | ||     |     ||            | 9  0 |      |
      |             |        0    ||_____|_____||            |  <-- |      |
    0 .-------------.              -------------          23  ------.------   0
      0  --X-->   159             0 -X->79

     Pad(Module,x,y)         Pad(Chamber,PhotoCat,x,y)  Pad(Equipment,Column,Dilogic,Channel)

            Equipment n

 143
  ^
  |    46 40 34 28 22 16 10 04
  |    44 38 32 26 20 14 08 02
       40 36 30 24 18 12 06 00
  y    43 37 31 25 19 13 07 01
       45 39 33 27 21 15 09 03
  |    47 41 35 29 23 17 11 05
  |

  0


   0 --------- x ------> 79

     For Equipment n :    x = 79 - (Dilo * 8 + Chan / 8)
                          y = 143 - (Column * 6 + Chan % 6)

    --------------------------------------------------------------------------- */
/// \class Geo
/// \brief HMPID  detector geometry (only statics)
class Geo
{
 public:
  // From AliTOFGeometry
  //  static void translate(Float_t* xyz, Float_t translationVector[3]);
  //  enum {
  //    // DAQ characteristics
  //    kNDDL = 4,    // Number of DDL (Detector Data Link) per sector
  //    kNTRM = 12,   // Number of TRM ( Readout Module) per DDL
  //    kNTdc = 15,   // Number of Tdc (Time to Digital Converter) per TRM
  //    kNChain = 2,  // Number of chains per TRM
  //    kNCrate = 72, // Number of Crates
  //    kNCh = 8      // Number of channels per Tdc
  //  };

  // ---- HMPID geometry -------
  static constexpr int MAXEQUIPMENTS = 14;
  static constexpr int N_SEGMENTS = 3;
  static constexpr int N_COLXSEGMENT = 8;
  static constexpr int N_COLUMNS = 24;
  static constexpr int N_DILOGICS = 10;
  static constexpr int N_CHANNELS = 48;
  static constexpr int N_DILOCHANNELS = 64;

  static constexpr int N_MODULES = 7;
  static constexpr int N_XROWS = 160;
  static constexpr int N_YCOLS = 144;

  static constexpr int MAXYCOLS = 143;
  static constexpr int MAXHALFXROWS = 79;
  static constexpr int HALFXROWS = 80;

  static constexpr int DILOPADSCOLS = 6;
  static constexpr int DILOPADSROWS = 8;

  static constexpr int EQUIPMENTSPERMODULE = 2;

  static constexpr int N_EQUIPMENTTOTALPADS = N_SEGMENTS * N_COLXSEGMENT * N_DILOGICS * N_CHANNELS;
  static constexpr int N_HMPIDTOTALPADS = MAXEQUIPMENTS * N_SEGMENTS * N_COLXSEGMENT * N_DILOGICS * N_CHANNELS;

  static constexpr int N_PHOTOCATODS = 6;
  static constexpr int N_PHOTOCATODSX = 80;
  static constexpr int N_PHOTOCATODSY = 48;
  static constexpr int MAXXPHOTO = 79;
  static constexpr int MAXYPHOTO = 47;

  static void Module2Equipment(int Mod, int Row, int Col, int* Equi, int* Colu, int* Dilo, int* Chan);
  static void Equipment2Module(int Equi, int Colu, int Dilo, int Chan, int* Mod, int* Row, int* Col);

  // from
  //static constexpr Bool_t FEAWITHMASKS[NSECTORS] =
  //  // TOF sectors with Nino masks: 0, 8, 9, 10, 16
  //  {kTRUE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kTRUE,
  //   kTRUE, kTRUE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kTRUE, kFALSE};
  //; // Selecting TOF sectors containing FEA cooling masks

  //  static Float_t getCableLength(Int_t icrate, Int_t islot, Int_t ichain, Int_t itdc) { return CABLELENGTH[icrate][islot - 3][ichain][itdc / 3]; }

 private:
  static void Init();

  ClassDefNV(Geo, 1);
};

class ReadOut
{
 public:
  struct LinkAddr {
    uint8_t Fee;
    uint8_t Cru;
    uint8_t Lnk;
    uint8_t Flp;
  };
  union Lnk {
    LinkAddr Id;
    uint32_t LinkUId;
  };
  static constexpr Lnk mEq[Geo::MAXEQUIPMENTS] = {{0, 0, 0, 160},
                                                  {1, 0, 1, 160},
                                                  {2, 0, 2, 160},
                                                  {3, 0, 3, 160},
                                                  {4, 1, 0, 160},
                                                  {5, 1, 1, 160},
                                                  {8, 1, 2, 160},
                                                  {9, 1, 3, 160},
                                                  {6, 2, 0, 161},
                                                  {7, 2, 1, 161},
                                                  {10, 2, 2, 161},
                                                  {11, 3, 0, 161},
                                                  {12, 3, 1, 161},
                                                  {13, 3, 2, 161}};

  static inline int FeeId(unsigned int idx) { return (idx > Geo::MAXEQUIPMENTS) ? -1 : mEq[idx].Id.Fee; };
  static inline int CruId(unsigned int idx) { return (idx > Geo::MAXEQUIPMENTS) ? -1 : mEq[idx].Id.Cru; };
  static inline int LnkId(unsigned int idx) { return (idx > Geo::MAXEQUIPMENTS) ? -1 : mEq[idx].Id.Lnk; };
  static inline int FlpId(unsigned int idx) { return (idx > Geo::MAXEQUIPMENTS) ? -1 : mEq[idx].Id.Flp; };
  static inline uint32_t UniqueId(unsigned int idx) { return (idx > Geo::MAXEQUIPMENTS) ? -1 : mEq[idx].LinkUId; };

  static unsigned int searchIdx(int FeeId)
  {
    for (int i = 0; i < Geo::MAXEQUIPMENTS; i++) {
      if (FeeId == mEq[i].Id.Fee) {
        return (i);
      }
    }
    return (-1);
  }
  static unsigned int searchIdx(int CruId, int LnkId)
  {
    for (int i = 0; i < Geo::MAXEQUIPMENTS; i++) {
      if (CruId == mEq[i].Id.Cru && LnkId == mEq[i].Id.Lnk) {
        return (i);
      }
    }
    return (-1);
  }
  static void getInfo(unsigned int idx, int& Fee, int& Cru, int& Lnk, int& Flp)
  {
    Cru = mEq[idx].Id.Cru;
    Lnk = mEq[idx].Id.Lnk;
    Fee = mEq[idx].Id.Fee;
    Flp = mEq[idx].Id.Flp;
    return;
  }

  ClassDefNV(ReadOut, 1);
};
/*
void ReadOut::Init()
{
  mEq[0].Id  = { 0, 0, 0, 160};
  mEq[1].Id  = { 1, 0, 1, 160};
  mEq[2].Id  = { 2, 0, 2, 160};
  mEq[3].Id  = { 3, 0, 3, 160};
  mEq[4].Id  = { 4, 1, 0, 160};
  mEq[5].Id  = { 5, 1, 1, 160};
  mEq[6].Id  = { 8, 1, 2, 160};
  mEq[7].Id  = { 9, 1, 3, 160};
  mEq[8].Id  = { 6, 2, 0, 161};
  mEq[9].Id  = { 7, 2, 1, 161};
  mEq[10].Id = {10, 2, 2, 161};
  mEq[11].Id = {11, 3, 0, 161};
  mEq[12].Id = {12, 3, 1, 161};
  mEq[13].Id = {13, 3, 2, 161};
};
*/

=======
#include <Rtypes.h>

namespace o2
{
namespace hmpid
{

/*  ------------------ HMPID Detector Coordinate definition -------------------

  143 .-------------.             .-------------.          0  ------.------   23
      |             |             |.-----.-----.|            |      |  <-- |
      |             |             ||     |     ||            |      | 9  0 |
      |             |             ||  4  |  5  ||            |      | Dil. |
    ^ |             |             ||     |     ||          | |      |      |  ^
    | |             |             ||_____|_____||          | |      |      |  |
    | |             |             |.-----.-----.|          | |      |      |  |
    Y |             |             ||     |     ||            |      |      |
    | |             |             ||  2  |  3  ||     Column |  n   |  n+1 | Column
    | |             |             ||     |     ||            |      |      |
      |             |             ||_____|_____||          | |      |      |  |
      |             |       47  ^ |.-----.-----.|          | |      |      |  |
      |             |           | ||     |     ||          V |      |      |  |
      |             |           Y ||  0  |  1  ||            | Dil. |      |
      |             |           | ||     |     ||            | 9  0 |      |
      |             |        0    ||_____|_____||            |  <-- |      |
    0 .-------------.              -------------          23  ------.------   0
      0  --X-->   159             0 -X->79

     Pad(Module,x,y)         Pad(Chamber,PhotoCat,x,y)  Pad(Equipment,Column,Dilogic,Channel)

            Equipment n

 143
  ^
  |    46 40 34 28 22 16 10 04
  |    44 38 32 26 20 14 08 02
       40 36 30 24 18 12 06 00
  y    43 37 31 25 19 13 07 01
       45 39 33 27 21 15 09 03
  |    47 41 35 29 23 17 11 05
  |

  0


   0 --------- x ------> 79

     For Equipment n :    x = 79 - (Dilo * 8 + Chan / 8)
                          y = 143 - (Column * 6 + Chan % 6)

    --------------------------------------------------------------------------- */
/// \class Geo
/// \brief HMPID  detector geometry (only statics)
class Geo
{
 public:
  // From AliTOFGeometry
  //  static void translate(Float_t* xyz, Float_t translationVector[3]);
  //  enum {
  //    // DAQ characteristics
  //    kNDDL = 4,    // Number of DDL (Detector Data Link) per sector
  //    kNTRM = 12,   // Number of TRM ( Readout Module) per DDL
  //    kNTdc = 15,   // Number of Tdc (Time to Digital Converter) per TRM
  //    kNChain = 2,  // Number of chains per TRM
  //    kNCrate = 72, // Number of Crates
  //    kNCh = 8      // Number of channels per Tdc
  //  };

  // ---- HMPID geometry -------
  static constexpr int MAXEQUIPMENTS = 14;
  static constexpr int N_SEGMENTS = 3;
  static constexpr int N_COLXSEGMENT = 8;
  static constexpr int N_COLUMNS = 24;
  static constexpr int N_DILOGICS = 10;
  static constexpr int N_CHANNELS = 48;
  static constexpr int N_DILOCHANNELS = 64;

  static constexpr int N_MODULES = 7;
  static constexpr int N_XROWS = 160;
  static constexpr int N_YCOLS = 144;

  static constexpr int MAXYCOLS = 143;
  static constexpr int MAXHALFXROWS = 79;
  static constexpr int HALFXROWS = 80;

  static constexpr int DILOPADSCOLS = 6;
  static constexpr int DILOPADSROWS = 8;

  static constexpr int EQUIPMENTSPERMODULE = 2;

  static constexpr int N_EQUIPMENTTOTALPADS = N_SEGMENTS * N_COLXSEGMENT * N_DILOGICS * N_CHANNELS;
  static constexpr int N_HMPIDTOTALPADS = MAXEQUIPMENTS * N_SEGMENTS * N_COLXSEGMENT * N_DILOGICS * N_CHANNELS;

  static constexpr int N_PHOTOCATODS = 6;
  static constexpr int N_PHOTOCATODSX = 80;
  static constexpr int N_PHOTOCATODSY = 48;
  static constexpr int MAXXPHOTO = 79;
  static constexpr int MAXYPHOTO = 47;

  static void Module2Equipment(int Mod, int Row, int Col, int* Equi, int* Colu, int* Dilo, int* Chan);
  static void Equipment2Module(int Equi, int Colu, int Dilo, int Chan, int* Mod, int* Row, int* Col);

  // from
  //static constexpr Bool_t FEAWITHMASKS[NSECTORS] =
  //  // TOF sectors with Nino masks: 0, 8, 9, 10, 16
  //  {kTRUE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kTRUE,
  //   kTRUE, kTRUE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kTRUE, kFALSE};
  //; // Selecting TOF sectors containing FEA cooling masks

  //  static Float_t getCableLength(Int_t icrate, Int_t islot, Int_t ichain, Int_t itdc) { return CABLELENGTH[icrate][islot - 3][ichain][itdc / 3]; }

 private:
  static void Init();

  ClassDefNV(Geo, 1);
};
>>>>>>> refs/heads/dev
} // namespace hmpid
} // namespace o2

#endif
