// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MCHChannelCalibrator.h
/// \brief Transitional compatibility header 
///
/// \author Andrea Ferrero

#ifndef ALICEO2_MCH_CALIBRATION_CHANNELCALIBRATOR_H
#define ALICEO2_MCH_CALIBRATION_CHANNELCALIBRATOR_H

#include "MCHCalibration/PedestalCalibrator.h"

namespace o2
{
namespace mch
{
namespace calibration
{

using MCHChannelCalibrator = PedestalCalibrator;

} // end namespace calibration
} // end namespace mch
} // end namespace o2

#endif