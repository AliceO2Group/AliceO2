//
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file segmentationCInterface.h
 * C Interface to the Muon MCH mapping.
 *
 * This interface is actually the main entrypoint to the mapping.
 * Even the C++ interface is using it (segmentation.h).
 *
 * Based on the idea of hourglass interfaces
 * https://github.com/CppCon/CppCon2014/tree/master/Presentations/Hourglass%20Interfaces%20for%20C%2B%2B%20APIs
 *
 * @author  Laurent Aphecetche
 */

#ifndef O2_MCH_SEGMENTATIONCINTERFACE_H
#define O2_MCH_SEGMENTATIONCINTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MchSegmentation* MchSegmentationHandle;

typedef void (*MchDetectionElementHandler)(void* clientData, int detElemId);

typedef void (*MchDualSampaHandler)(void* clientData, int dualSampaId);

typedef void (*MchPadHandler)(void* clientData, int paduid);

/** @name Creation and destruction of the segmentation handle.
 *
 * Most of the functions of this library require a segmentation handle to work with.
 */
///@{

/// Create a handle to a segmentation for a given plane of a detection element.
MchSegmentationHandle mchSegmentationConstruct(int detElemId, bool isBendingPlane);

/// Delete a segmentation handle.
void mchSegmentationDestruct(MchSegmentationHandle segHandle);
///@}

/** @name Pad Unique Identifier
 * Pads are identified by a unique integer, paduid.
 * @warning This paduid is only valid within the functions of this library
 * (to use the query methods padPosition, padSize, etc...).
 * So do _not_ rely on any given value it might take (as it might change
 * between e.g. library versions)
 */
///@{

/// Return > 0 if paduid is a valid one or <= 1 if not
int mchSegmentationIsPadValid(MchSegmentationHandle segHandle, int paduid);
///@}

/** @name Pad finding.
 * Functions to find a pad.
 * In each case the returned integer
 * represents either a paduid if a pad is found or
 * an integer representing an invalid paduid otherwise.
 * Validity of the returned value can be tested using mchSegmentationIsPadValid()
 */
///@{

/// Find the pad at position (x,y) (in cm).
int mchSegmentationFindPadByPosition(MchSegmentationHandle segHandle, double x, double y);

/// Find the pad connected to the given channel of the given dual sampa.
int mchSegmentationFindPadByFEE(MchSegmentationHandle segHandle, int dualSampaId, int dualSampaChannel);
///@}

/** @name Pad information retrieval.
 * Given a _valid_ paduid those methods return information
 * (position, size, front-end electronics) about that pad.
 *
 * Positions and sizes are in centimetres.
 *
 * If paduid is invalid, you are on your own.
 */
/// @{
double mchSegmentationPadPositionX(MchSegmentationHandle segHandle, int paduid);

double mchSegmentationPadPositionY(MchSegmentationHandle segHandle, int paduid);

double mchSegmentationPadSizeX(MchSegmentationHandle segHandle, int paduid);

double mchSegmentationPadSizeY(MchSegmentationHandle segHandle, int paduid);

int mchSegmentationPadDualSampaId(MchSegmentationHandle segHandle, int paduid);

int mchSegmentationPadDualSampaChannel(MchSegmentationHandle segHandle, int paduid);
///@}

/** @name ForEach methods.
 * Functions to loop over some items : detection elements, dual sampas, and pads.
 */
///@{
void mchSegmentationForEachDetectionElement(MchDetectionElementHandler handler, void* clientData);

void mchSegmentationForOneDetectionElementOfEachSegmentationType(MchDetectionElementHandler handler, void* clientData);

void mchSegmentationForEachDualSampa(MchSegmentationHandle segHandle, MchDualSampaHandler handler, void* clientData);

void mchSegmentationForEachPadInDualSampa(MchSegmentationHandle segHandle, int dualSampaId, MchPadHandler handler,
                                          void* clientData);

void mchSegmentationForEachPadInArea(MchSegmentationHandle segHandle, double xmin, double ymin, double xmax,
                                     double ymax, MchPadHandler handler, void* clientData);

void mchSegmentationForEachNeighbouringPad(MchSegmentationHandle segHandle, int paduid, MchPadHandler handler,
                                           void* userData);
///@}

#ifdef __cplusplus
};
#endif

#endif
