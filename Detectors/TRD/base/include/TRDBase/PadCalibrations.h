// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_PADCALIBRATIONS_H
#define O2_TRD_PADCALIBRATIONS_H

////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//  TRD calibration class for parameters which are stored pad wise (1.2M entries) //
//  2019 - Ported from various bits of AliRoot (SHTM)                             //
//  Similar CalPad                                                                //
////////////////////////////////////////////////////////////////////////////////////

#include <array>

#include "TRDBase/PadParameters.h"

class TRDGeometry;

namespace o2
{
namespace trd
{

template <class T>
class PadCalibrations
{
 public:
  enum { kNplan = 6,
         kNcham = 5,
         kNsect = 18,
         kNdet = 540 };
  enum { kVdrift = 0,
         kGainFactor = 1,
         kT0 = 2,
         kExB = 3,
         kLocalGainFactor = 4 };
  PadCalibrations(int p, int c);
  ~PadCalibrations() = default;
  //
  // various functions that I *think* should be moved to higher classes, I do not understand the getmeanrms etc. as it is used in MakeHisto
  int getNChannels(int roc) { return mreadOutChamber[roc].getNChannels(); }
  PadParameters<T>& getChamberPads(int roc) { return mreadOutChamber[roc]; }
  T getValue(int roc, int col, int row) { return (mreadOutChamber.at(roc)).getValue(col, row); }

 protected:
  std::array<PadParameters<T>, kNdet> mreadOutChamber;
};
} // namespace trd
} // namespace o2
#endif
