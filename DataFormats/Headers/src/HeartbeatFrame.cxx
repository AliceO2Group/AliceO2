/// @file   Heartbeatframe.cxx
/// @author Matthias Richter
/// @since  2017-02-02
/// @brief  Some additions to definition of the heartbeat frame layout

#include "Headers/HeartbeatFrame.h"

// define the description with a terminating '0' (meaning 15 characters)
const AliceO2::Header::DataDescription AliceO2::Header::gDataDescriptionHeartbeatFrame("HEARTBEATFRAME");

const uint32_t AliceO2::Header::HeartbeatFrameEnvelope::sVersion = 1;
const AliceO2::Header::HeaderType AliceO2::Header::HeartbeatFrameEnvelope::sHeaderType= String2<uint64_t>("HBFEnvel");
const AliceO2::Header::SerializationMethod AliceO2::Header::HeartbeatFrameEnvelope::sSerializationMethod = AliceO2::Header::gSerializationMethodNone;
