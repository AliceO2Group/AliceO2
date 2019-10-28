// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_PADSTATUS_H
#define O2_TRD_PADSTATUS_H

///////////////////////////////////////////////////////////////////////////////
// Store for the pad status across the entire TRD.
// PadCalibrations stores the roc, and PadParameters the actual content of
//  statuses.
//
// Note to run2 correlation:
// This then matches similarly to the old AliTRDCalPadStatus which held an
//  array of AliTRDCalSingleChamberStatus.
// Only real change to the interface is that one now needs to explicitly
//  state the roc in question when querying a pads status, as the method is
//  higher up in the heirachy now.
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/PadParameters.h"
#include "TRDBase/PadCalibrations.h"

namespace o2
{
namespace trd
{

class PadStatus : public PadCalibrations<char>
{
 public:
  using PadCalibrations<char>::PadCalibrations;
  ~PadStatus() = default;

  enum { kMasked = 2,
         kPadBridgedLeft = 4,
         kPadBridgedRight = 8,
         kReadSecond = 16,
         kNotConnected = 32 };

  bool isMasked(int roc, int col, int row) const { return ((getStatus(roc, col, row) & kMasked) ? true : false); };
  bool isBridgedLeft(int roc, int col, int row) const { return ((getStatus(roc, col, row) & kPadBridgedLeft) ? true : false); };
  bool isBridgedRight(int roc, int col, int row) const { return ((getStatus(roc, col, row) & kPadBridgedRight) ? true : false); };
  bool isNotConnected(int roc, int col, int row) const { return ((getStatus(roc, col, row) & kNotConnected) ? true : false); };
  int getNrows(int roc) const { return mreadOutChamber[roc].getNrows(); };
  int getNcols(int roc) const { return mreadOutChamber[roc].getNcols(); };

  int getChannel(int roc, int col, int row) const { return row + col * mreadOutChamber[roc].getNrows(); }
  int getNChannels(int roc) const { return getNChannels(roc); };
  char getStatus(int roc, int ich) const { return mreadOutChamber[roc].getValue(ich); };
  char getStatus(int roc, int col, int row) const { return mreadOutChamber[roc].getValue(getChannel(roc, col, row)); };

  void setStatus(int roc, int ich, char vd) { mreadOutChamber[roc].setValue(ich, vd); };
  void setStatus(int roc, int col, int row, char vd) { mreadOutChamber[roc].setValue(getChannel(roc, col, row), vd); };
};
} // namespace trd
} // namespace o2

#endif /* !PADSTATUS_H */
