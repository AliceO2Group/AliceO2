// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Heartbeatframe.cxx
/// @author Matthias Richter
/// @since  2017-02-02
/// @brief  Some additions to definition of the heartbeat frame layout

#include "Headers/HeartbeatFrame.h"

// define the description with a terminating '0' (meaning 15 characters)
const o2::header::DataDescription o2::header::gDataDescriptionHeartbeatFrame("HEARTBEATFRAME");

const uint32_t o2::header::HeartbeatFrameEnvelope::sVersion = 1;
const o2::header::HeaderType o2::header::HeartbeatFrameEnvelope::sHeaderType = String2<uint64_t>("HBFEnvel");
const o2::header::SerializationMethod o2::header::HeartbeatFrameEnvelope::sSerializationMethod = o2::header::gSerializationMethodNone;
