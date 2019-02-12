// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibTimeSlewingParamTOF.h
/// \brief Class to store the output of the matching to TOF for calibration

#ifndef ALICEO2_CALIBTIMESLEWINGPARAMTOF_H
#define ALICEO2_CALIBTIMESLEWINGPARAMTOF_H

#include <vector>

namespace o2
{
namespace dataformats
{
class CalibTimeSlewingParamTOF
{
 public:
  static const int NCHANNELS=157248;
  static const int NSECTORS=18;
  static const int NCHANNELXSECTORSECTOR=NCHANNELS/NSECTORS;

  CalibTimeSlewingParamTOF();

  float evalTimeSlewing(int channel, float tot) const;

  void addTimeSlewingInfo(int channel, float tot, float time);

 private:
  // TOF channel calibrations
  int mChannelStart[NSECTORS][NCHANNELXSECTORSECTOR]; ///< output LHC phase in ps
  std::vector<std::pair<float,float>> mTimeSlewing[NSECTORS];           ///< timeslweing correction >tot,time>

  //  ClassDefNV(CalibTimeSlewingParamTOF, 1);
};
}
}
#endif
