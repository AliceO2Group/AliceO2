// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_CALPADSTATUS_H
#define O2_TRD_CALPADSTATUS_H

#include "DataFormatsTRD/Constants.h"

#include "TH1F.h"
#include "TH2F.h"
#include <string>

namespace o2
{
namespace trd
{

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for the single pad status                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

class CalSingleChamberStatus;

class CalPadStatus
{

 public:
  enum { kMasked = 2,
         kPadBridgedLeft = 4,
         kPadBridgedRight = 8,
         kReadSecond = 16,
         kNotConnected = 32 };

  CalPadStatus();
  CalPadStatus(const Text_t* name, const Text_t* title);
  CalPadStatus(const CalPadStatus& c);
  ~CalPadStatus();
  CalPadStatus& operator=(const CalPadStatus& c);

  void Copy(CalPadStatus& c) const;

  Bool_t isMasked(Int_t d, Int_t col, Int_t row) const
  {
    return checkStatus(d, col, row, kMasked);
  };
  Bool_t isBridgedLeft(Int_t d, Int_t col, Int_t row) const
  {
    return checkStatus(d, col, row, kPadBridgedLeft);
  };
  Bool_t isBridgedRight(Int_t d, Int_t col, Int_t row) const
  {
    return checkStatus(d, col, row, kPadBridgedRight);
  };
  Bool_t isReadSecond(Int_t d, Int_t col, Int_t row) const
  {
    return checkStatus(d, col, row, kReadSecond);
  };
  Bool_t isNotConnected(Int_t d, Int_t col, Int_t row) const
  {
    return checkStatus(d, col, row, kNotConnected);
  };
  Bool_t checkStatus(Int_t d, Int_t col, Int_t row, Int_t bitMask) const;

  CalSingleChamberStatus* getCalROC(Int_t d) const { return mROC[d]; };
  CalSingleChamberStatus* getCalROC(Int_t p, Int_t c, Int_t s) const;

  // Plot functions
  TH1F* makeHisto1D();
  TH2F* makeHisto2DSmPl(Int_t sm, Int_t pl);
  void plotHistos2DSm(Int_t sm, const Char_t* name);

  std::string getTitle() { return mTitle; };
  std::string getName() { return mName; };
  void setTitle(const std::string newTitle) { mTitle = newTitle; };
  void setName(const std::string newName) { mName = newName; };

 protected:
  CalSingleChamberStatus* mROC[constants::MAXCHAMBER]; //  Array of ROC objects which contain the values per pad

 private:
  std::string mName;
  std::string mTitle;
  ClassDefNV(CalPadStatus, 1); //  TRD calibration class for the single pad status
};
} //namespace trd
} //namespace o2
#endif
