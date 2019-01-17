
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDCALPADSTATUS_H
#define O2_TRDCALPADSTATUS_H

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

class TRDCalSingleChamberStatus;

class TRDCalPadStatus
{

 public:
  enum { kNplan = 6,
         kNcham = 5,
         kNsect = 18,
         kNdet = 540 };
  enum { kMasked = 2,
         kPadBridgedLeft = 4,
         kPadBridgedRight = 8,
         kReadSecond = 16,
         kNotConnected = 32 };

  TRDCalPadStatus();
  TRDCalPadStatus(const Text_t* name, const Text_t* title);
  TRDCalPadStatus(const TRDCalPadStatus& c);
  virtual ~TRDCalPadStatus();
  TRDCalPadStatus& operator=(const TRDCalPadStatus& c);

  void Copy(TRDCalPadStatus& c) const;

  Bool_t IsMasked(Int_t d, Int_t col, Int_t row) const
  {
    return CheckStatus(d, col, row, kMasked);
  };
  Bool_t IsBridgedLeft(Int_t d, Int_t col, Int_t row) const
  {
    return CheckStatus(d, col, row, kPadBridgedLeft);
  };
  Bool_t IsBridgedRight(Int_t d, Int_t col, Int_t row) const
  {
    return CheckStatus(d, col, row, kPadBridgedRight);
  };
  Bool_t IsReadSecond(Int_t d, Int_t col, Int_t row) const
  {
    return CheckStatus(d, col, row, kReadSecond);
  };
  Bool_t IsNotConnected(Int_t d, Int_t col, Int_t row) const
  {
    return CheckStatus(d, col, row, kNotConnected);
  };
  Bool_t CheckStatus(Int_t d, Int_t col, Int_t row, Int_t bitMask) const;

  TRDCalSingleChamberStatus* getCalROC(Int_t d) const { return mROC[d]; };
  TRDCalSingleChamberStatus* getCalROC(Int_t p, Int_t c, Int_t s) const;

  // Plot functions
  TH1F* MakeHisto1D();
  TH2F* MakeHisto2DSmPl(Int_t sm, Int_t pl);
  void PlotHistos2DSm(Int_t sm, const Char_t* name);

  std::string getTitle() { return mTitle; };
  std::string getName() { return mName; };
  void setTitle(const std::string newTitle) { mTitle = newTitle; };
  void setName(const std::string newName) { mName = newName; };

 protected:
  TRDCalSingleChamberStatus* mROC[kNdet]; //  Array of ROC objects which contain the values per pad

 private:
  std::string mName;
  std::string mTitle;
  ClassDefNV(TRDCalPadStatus, 1) //  TRD calibration class for the single pad status
};
} //namespace trd
} //namespace o2
#endif
