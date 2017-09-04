// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.h
/// \brief Defines unique identifier for all AliceO2 detector systems, needed for stack filtring

#ifndef ALICEO2_DATA_DETECTORLIST_H_
#define ALICEO2_DATA_DETECTORLIST_H_

// kSTOPHERE is needed for iteration over the enum. All detectors have to be put before.
enum DetectorId
{
    kAliIts, kAliTpc, kAliMft, kSTOPHERE
};

#endif
