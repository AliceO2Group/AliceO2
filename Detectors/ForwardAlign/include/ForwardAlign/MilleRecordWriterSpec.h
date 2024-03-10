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

/// \file MilleRecordWriterSpec.h
/// \brief Implementation of a data processor to write MillePede record in a root file
///
/// \author Chi Zhang, CEA-Saclay, chi.zhang@cern.ch

#ifndef ALICEO2_FWDALIGN_MILLERECORDWRITERSPEC_H
#define ALICEO2_FWDALIGN_MILLERECORDWRITERSPEC_H

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace fwdalign
{

framework::DataProcessorSpec getMilleRecordWriterSpec(bool useMC, const char* specName = "fwdalign-millerecord-writer",
                                                      const char* fileName = "millerecords.root");

} // namespace fwdalign
} // namespace o2

#endif // ALICEO2_FWDALIGN_MILLERECORDWRITERSPEC_H