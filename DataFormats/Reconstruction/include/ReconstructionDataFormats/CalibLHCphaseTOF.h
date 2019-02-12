// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibLHCphaseTOF.h
/// \brief Class to store the output of the matching to TOF for calibration

#ifndef ALICEO2_CALIBLHCPHASETOF_H
#define ALICEO2_CALIBLHCPHASETOF_H

#include <vector>

namespace o2
{
namespace dataformats
{
class CalibLHCphaseTOF
{
 public:
  CalibLHCphaseTOF();

  float getLHCphase(int timestamp) const;

  void addLHCphase(int timestamp, float phaseLHC);

 private:
  // LHCphase calibration
  std::vector<std::pair <int,float>> mLHCphase;                     ///< <timestamp,LHCphase> from which the LHCphase measurement is valid

  //  ClassDefNV(CalibLHCphaseTOF, 1);
};
}
}
#endif
