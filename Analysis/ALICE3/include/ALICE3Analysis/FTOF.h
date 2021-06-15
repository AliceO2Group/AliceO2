// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   FTOF.h
/// \author Nicolo' Jacazio
/// \since  27/05/2021
/// \brief  Set of tables for the ALICE3 FTOF information
///

#ifndef O2_ANALYSIS_ALICE3_FTOF_H_
#define O2_ANALYSIS_ALICE3_FTOF_H_

// O2 includes
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace alice3ftof
{
DECLARE_SOA_INDEX_COLUMN(Track, track);                //!
DECLARE_SOA_COLUMN(FTOFLength, ftofLength, float);     //!
DECLARE_SOA_COLUMN(FTOFSignal, ftofSignal, float);     //!
DECLARE_SOA_COLUMN(FTOFDeltaEl, ftofDeltaEl, float);   //!
DECLARE_SOA_COLUMN(FTOFDeltaMu, ftofDeltaMu, float);   //!
DECLARE_SOA_COLUMN(FTOFDeltaPi, ftofDeltaPi, float);   //!
DECLARE_SOA_COLUMN(FTOFDeltaKa, ftofDeltaKa, float);   //!
DECLARE_SOA_COLUMN(FTOFDeltaPr, ftofDeltaPr, float);   //!
DECLARE_SOA_COLUMN(FTOFNsigmaEl, ftofNsigmaEl, float); //!
DECLARE_SOA_COLUMN(FTOFNsigmaMu, ftofNsigmaMu, float); //!
DECLARE_SOA_COLUMN(FTOFNsigmaPi, ftofNsigmaPi, float); //!
DECLARE_SOA_COLUMN(FTOFNsigmaKa, ftofNsigmaKa, float); //!
DECLARE_SOA_COLUMN(FTOFNsigmaPr, ftofNsigmaPr, float); //!
} // namespace alice3ftof

DECLARE_SOA_TABLE(FTOFs, "AOD", "FTOF", //!
                  o2::soa::Index<>,
                  alice3ftof::TrackId,
                  alice3ftof::FTOFLength,
                  alice3ftof::FTOFSignal,
                  alice3ftof::FTOFDeltaEl,
                  alice3ftof::FTOFDeltaMu,
                  alice3ftof::FTOFDeltaPi,
                  alice3ftof::FTOFDeltaKa,
                  alice3ftof::FTOFDeltaPr,
                  alice3ftof::FTOFNsigmaEl,
                  alice3ftof::FTOFNsigmaMu,
                  alice3ftof::FTOFNsigmaPi,
                  alice3ftof::FTOFNsigmaKa,
                  alice3ftof::FTOFNsigmaPr);

using FTOF = FTOFs::iterator;

} // namespace o2::aod

#endif // O2_ANALYSIS_ALICE3_FTOF_H_
