// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUErrorCodes.h
/// \author David Rohr

// Error Codes for GPU Tracker
GPUCA_ERROR_CODE(0, ERROR_NONE)
GPUCA_ERROR_CODE(1, ERROR_ROWSTARTHIT_OVERFLOW)
GPUCA_ERROR_CODE(2, ERROR_STARTHIT_OVERFLOW)
GPUCA_ERROR_CODE(3, ERROR_TRACKLET_OVERFLOW)
GPUCA_ERROR_CODE(4, ERROR_TRACKLET_HIT_OVERFLOW)
GPUCA_ERROR_CODE(5, ERROR_TRACK_OVERFLOW)
GPUCA_ERROR_CODE(6, ERROR_TRACK_HIT_OVERFLOW)
GPUCA_ERROR_CODE(7, ERROR_GLOBAL_TRACKING_TRACK_OVERFLOW)
GPUCA_ERROR_CODE(8, ERROR_GLOBAL_TRACKING_TRACK_HIT_OVERFLOW)
GPUCA_ERROR_CODE(9, ERROR_LOOPER_OVERFLOW)
