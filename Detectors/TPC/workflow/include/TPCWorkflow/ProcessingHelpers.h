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

namespace o2::framework
{
class ProcessingContext;
}

namespace o2::tpc
{
namespace processing_helpers
{

uint64_t getRunNumber(o2::framework::ProcessingContext& pc);

/// \return returns tf from tfCounter
uint32_t getCurrentTF(o2::framework::ProcessingContext& pc);

uint32_t getFirstTForbit(o2::framework::ProcessingContext& pc);

/// \return returns creation time of tf
uint64_t getCreationTime(o2::framework::ProcessingContext& pc);

/// \return returns time stamp in microsecond-precission
/// Note that the input spec has to be defined as: inputSpecs.emplace_back("orbitreset", "CTP", "ORBITRESET", 0, Lifetime::Condition, ccdbParamSpec("CTP/Calib/OrbitReset"));
uint64_t getTimeStamp(o2::framework::ProcessingContext& pc);

/// \return returns time stamp in microsecond-precission
uint64_t getTimeStamp(o2::framework::ProcessingContext& pc, const Long64_t orbitReset);

/// \return returns time stamp in microsecond-precission
uint64_t getTimeStamp(const Long64_t orbitReset, const uint32_t tfOrbitFirst);

/// \return returns the orbit reset time
/// Note that the input spec has to be defined as: inputSpecs.emplace_back("orbitreset", "CTP", "ORBITRESET", 0, Lifetime::Condition, ccdbParamSpec("CTP/Calib/OrbitReset"));
Long64_t getOrbitReset(o2::framework::ProcessingContext& pc);

} // namespace processing_helpers
} // namespace o2::tpc
