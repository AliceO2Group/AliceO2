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

///
/// @author  Laurent Aphecetche

#include "MCHMappingInterface/SegmentationCInterface.h"
#include "SegmentationImpl3.h"
#include "mchmappingimpl3_export.h"
#include <fstream>

extern "C" {

struct MCHMAPPINGIMPL3_EXPORT MchSegmentation
{
    MchSegmentation(o2::mch::mapping::impl3::Segmentation *i) : impl{i}
    {}

    std::unique_ptr<o2::mch::mapping::impl3::Segmentation> impl;
};

MCHMAPPINGIMPL3_EXPORT MchSegmentationHandle
mchSegmentationConstruct(int detElemId, bool isBendingPlane)
{
  auto seg = o2::mch::mapping::impl3::createSegmentation(detElemId, isBendingPlane);
  return seg ? new MchSegmentation(seg) : nullptr;
}

MCHMAPPINGIMPL3_EXPORT
void mchSegmentationDestruct(MchSegmentationHandle sh)
{
  delete sh;
}

MCHMAPPINGIMPL3_EXPORT
int mchSegmentationId(MchSegmentationHandle segHandle)
{
// return segHandle->impl->getId();
  return -1;
}

MCHMAPPINGIMPL3_EXPORT
int mchSegmentationFindPadByPosition(MchSegmentationHandle segHandle, double x, double y)
{
  return segHandle->impl->findPadByPosition(x, y);
}

MCHMAPPINGIMPL3_EXPORT
int mchSegmentationFindPadByFEE(MchSegmentationHandle segHandle, int dualSampaId, int dualSampaChannel)
{
  return segHandle->impl->findPadByFEE(dualSampaId, dualSampaChannel);
}

MCHMAPPINGIMPL3_EXPORT
void mchSegmentationForEachDetectionElement(MchDetectionElementHandler handler, void *clientData)
{
  for (auto detElemId: {
    100, 101, 102, 103, 200, 201, 202, 203, 300, 301, 302, 303, 400, 401, 402, 403, 500, 501, 502, 503, 504, 505, 506,
    507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611,
    612, 613, 614, 615, 616, 617, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716,
    717, 718, 719, 720, 721, 722, 723, 724, 725, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813,
    814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910,
    911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 1000, 1001, 1002, 1003, 1004, 1005, 1006,
    1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025
  }) {
    handler(clientData, detElemId);
  }
}

MCHMAPPINGIMPL3_EXPORT
void mchSegmentationForEachDualSampa(MchSegmentationHandle segHandle, MchDualSampaHandler handler, void *clientData)
{
  for (auto dualSampaId: segHandle->impl->dualSampaIds()) {
    handler(clientData, dualSampaId);
  }
}

MCHMAPPINGIMPL3_EXPORT
void mchSegmentationForOneDetectionElementOfEachSegmentationType(MchDetectionElementHandler handler, void *clientData)
{
  for (auto detElemId : {
    100, 300, 500, 501, 502, 503, 504, 600, 601, 602, 700, 701, 702, 703, 704, 705, 706, 902, 903, 904, 905
  }) {
    handler(clientData, detElemId);
  }
}

MCHMAPPINGIMPL3_EXPORT
int mchSegmentationIsPadValid(MchSegmentationHandle segHandle, int paduid)
{
  return paduid != segHandle->impl->InvalidPadUid;
}

MCHMAPPINGIMPL3_EXPORT
void mchSegmentationForEachPadInDualSampa(MchSegmentationHandle segHandle, int dualSampaId, MchPadHandler handler,
                                          void *clientData)
{
  for (auto p: segHandle->impl->getPadUids(dualSampaId)) {
    handler(clientData, p);
  }
}

MCHMAPPINGIMPL3_EXPORT
double mchSegmentationPadPositionX(MchSegmentationHandle segHandle, int paduid)
{
  return segHandle->impl->padPositionX(paduid);
}

MCHMAPPINGIMPL3_EXPORT
double mchSegmentationPadPositionY(MchSegmentationHandle segHandle, int paduid)
{
  return segHandle->impl->padPositionY(paduid);
}

MCHMAPPINGIMPL3_EXPORT
double mchSegmentationPadSizeX(MchSegmentationHandle segHandle, int paduid)
{
  return segHandle->impl->padSizeX(paduid);
}

MCHMAPPINGIMPL3_EXPORT
double mchSegmentationPadSizeY(MchSegmentationHandle segHandle, int paduid)
{
  return segHandle->impl->padSizeY(paduid);
}

MCHMAPPINGIMPL3_EXPORT
int mchSegmentationPadDualSampaId(MchSegmentationHandle segHandle, int paduid)
{
  return segHandle->impl->padDualSampaId(paduid);
}

MCHMAPPINGIMPL3_EXPORT
int mchSegmentationPadDualSampaChannel(MchSegmentationHandle segHandle, int paduid)
{
  return segHandle->impl->padDualSampaChannel(paduid);

}

MCHMAPPINGIMPL3_EXPORT
void
mchSegmentationForEachPadInArea(MchSegmentationHandle segHandle, double xmin, double ymin, double xmax, double ymax,
                                MchPadHandler handler, void *clientData)
{
  for (auto p: segHandle->impl->getPadUids(xmin,ymin,xmax,ymax)) {
    handler(clientData, p);
  }

}

MCHMAPPINGIMPL3_EXPORT
void
mchSegmentationForEachNeighbouringPad(MchSegmentationHandle segHandle, int paduid, MchPadHandler handler,
                                      void *userData)
{
  for (auto p: segHandle->impl->getNeighbouringPadUids(paduid)) {
    handler(userData, p);
  }
}
} // extern "C"
