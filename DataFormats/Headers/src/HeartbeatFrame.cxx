/// @file   Heartbeatframe.cxx
/// @author Matthias Richter
/// @since  2017-02-02
/// @brief  Some additions to definition of the heartbeat frame layout

#include "Headers/HeartbeatFrame.h"

// define the description with a terminating '0' (meaning 15 characters)
const o2::Header::DataDescription o2::Header::gDataDescriptionHeartbeatFrame("HEARTBEATFRAME");

const uint32_t o2::Header::HeartbeatFrameEnvelope::sVersion = 1;
const o2::Header::HeaderType o2::Header::HeartbeatFrameEnvelope::sHeaderType= String2<uint64_t>("HBFEnvel");
const o2::Header::SerializationMethod o2::Header::HeartbeatFrameEnvelope::sSerializationMethod = o2::Header::gSerializationMethodNone;
