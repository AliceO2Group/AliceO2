// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file CathodeSegmentationCInterface.h
 * C Interface to the Muon MCH mapping.
 *
 * This interface is actually the main entrypoint to the mapping,
 * at least if dealing with cathodes and not full detection elements.
 * (in which case Segmentation.h is probably more convenient)
 *
 * Even the C++ interface is using it (CathodeSegmentation.h).
 *
 * Based on the idea of hourglass interfaces
 * https://github.com/CppCon/CppCon2014/tree/master/Presentations/Hourglass%20Interfaces%20for%20C%2B%2B%20APIs
 *
 * @author  Laurent Aphecetche
 */

#ifndef O2_MCH_CATHODESEGMENTATIONCINTERFACE_H
#define O2_MCH_CATHODESEGMENTATIONCINTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MchCathodeSegmentation* MchCathodeSegmentationHandle;

typedef void (*MchDetectionElementHandler)(void* clientData, int detElemId);

typedef void (*MchDualSampaHandler)(void* clientData, int dualSampaId);

typedef void (*MchPadHandler)(void* clientData, int catPadIndex);

/** @name Creation and destruction of the segmentation handle.
 *
 * Most of the functions of this library require a segmentation handle to work with.
 */
///@{

/// Create a handle to a segmentation for a given plane of a detection element.
MchCathodeSegmentationHandle mchCathodeSegmentationConstruct(int detElemId, bool isBendingPlane);

/// Delete a segmentation handle.
void mchCathodeSegmentationDestruct(MchCathodeSegmentationHandle segHandle);
///@}

/** @name Pad Unique Identifier
 * Pads are identified by a unique integer, catPadIndex.
 * @warning This catPadIndex is only valid within the functions of this library
 * (to use the query methods padPosition, padSize, etc...).
 * So do _not_ rely on any given value it might take (as it might change
 * between e.g. library versions)
 */
///@{

/// Return > 0 if catPadIndex is a valid one or <= 1 if not
int mchCathodeSegmentationIsPadValid(MchCathodeSegmentationHandle segHandle, int catPadIndex);
///@}

/** @name Pad finding.
 * Functions to find a pad.
 * In each case the returned integer
 * represents either a catPadIndex if a pad is found or
 * an integer representing an invalid catPadIndex otherwise.
 * Validity of the returned value can be tested using mchCathodeSegmentationIsPadValid()
 */
///@{

/// Find the pad at position (x,y) (in cm).
int mchCathodeSegmentationFindPadByPosition(MchCathodeSegmentationHandle segHandle, double x, double y);

/// Find the pad connected to the given channel of the given dual sampa.
int mchCathodeSegmentationFindPadByFEE(MchCathodeSegmentationHandle segHandle, int dualSampaId, int dualSampaChannel);
///@}

/** @name Pad information retrieval.
 * Given a _valid_ catPadIndex those methods return information
 * (position, size, front-end electronics) about that pad.
 *
 * Positions and sizes are in centimetres.
 *
 * If catPadIndex is invalid, you are on your own.
 */
/// @{
double mchCathodeSegmentationPadPositionX(MchCathodeSegmentationHandle segHandle, int catPadIndex);

double mchCathodeSegmentationPadPositionY(MchCathodeSegmentationHandle segHandle, int catPadIndex);

double mchCathodeSegmentationPadSizeX(MchCathodeSegmentationHandle segHandle, int catPadIndex);

double mchCathodeSegmentationPadSizeY(MchCathodeSegmentationHandle segHandle, int catPadIndex);

int mchCathodeSegmentationPadDualSampaId(MchCathodeSegmentationHandle segHandle, int catPadIndex);

int mchCathodeSegmentationPadDualSampaChannel(MchCathodeSegmentationHandle segHandle, int catPadIndex);
///@}

/** @name ForEach methods.
 * Functions to loop over some items : detection elements, dual sampas, and pads.
 */
///@{
void mchCathodeSegmentationForEachDetectionElement(MchDetectionElementHandler handler, void* clientData);

void mchCathodeSegmentationForOneDetectionElementOfEachSegmentationType(MchDetectionElementHandler handler, void* clientData);

void mchCathodeSegmentationForEachDualSampa(MchCathodeSegmentationHandle segHandle, MchDualSampaHandler handler, void* clientData);

void mchCathodeSegmentationForEachPadInDualSampa(MchCathodeSegmentationHandle segHandle, int dualSampaId, MchPadHandler handler,
                                                 void* clientData);

void mchCathodeSegmentationForEachPadInArea(MchCathodeSegmentationHandle segHandle, double xmin, double ymin, double xmax,
                                            double ymax, MchPadHandler handler, void* clientData);

void mchCathodeSegmentationForEachPad(MchCathodeSegmentationHandle segHandle, MchPadHandler handler, void* clientData);

void mchCathodeSegmentationForEachNeighbouringPad(MchCathodeSegmentationHandle segHandle, int catPadIndex, MchPadHandler handler,
                                                  void* userData);
///@}

#ifdef __cplusplus
};
#endif

#endif
