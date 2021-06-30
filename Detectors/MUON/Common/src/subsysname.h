// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MUON_COMMON_SUBSYSNAME_H
#define O2_MUON_COMMON_SUBSYSNAME_H

namespace o2::muon
{
const char* subsysname()
{
#if defined(MUON_SUBSYSTEM_MCH) && !defined(MUON_SUBSYSTEM_MID)
  return "MCH";
#elif defined(MUON_SUBSYSTEM_MID) && !defined(MUON_SUBSYSTEM_MCH)
  return "MID";
#else
#error "Must define one and only one of MUON_SUBSYSTEM_MCH or MUON_SUBSYSTEM_MID"
#endif
}
} // namespace o2::muon

#endif
