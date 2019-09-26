// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_PadNoise_H
#define O2_TRD_PadNoise_H

////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//  TRD calibration class for parameters which are stored pad wise (1.2M entries) //
//  2019 - Ported from various bits of AliRoot (SHTM)                             //
//  originally in TRD run2 parlance, this would be a AliTRDCalPad instantiated    //
//  as a PadNoise                                                                 //
////////////////////////////////////////////////////////////////////////////////////

#include "TRDBase/PadParameters.h"

namespace o2
{
namespace trd
{

class PadNoise : private PadCalibrations<unsigned int>
{
 public:
  using PadCalibrations<unsigned int>::PadCalibrations;
  ~PadNoise() = default;
};
} // namespace trd
} // namespace o2
#endif
