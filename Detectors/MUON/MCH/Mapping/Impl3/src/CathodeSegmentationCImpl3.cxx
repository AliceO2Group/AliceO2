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

#include "MCHMappingInterface/CathodeSegmentationCInterface.h"
#include "CathodeSegmentationImpl3.h"
#include "o2mchmappingimpl3_export.h"
#include <fstream>

extern "C" {

struct O2MCHMAPPINGIMPL3_EXPORT MchCathodeSegmentation {
  MchCathodeSegmentation(o2::mch::mapping::impl3::CathodeSegmentation* i) : impl{i} {}

  std::unique_ptr<o2::mch::mapping::impl3::CathodeSegmentation> impl;
};

O2MCHMAPPINGIMPL3_EXPORT MchCathodeSegmentationHandle mchCathodeSegmentationConstruct(int detElemId, bool isBendingPlane)
{
  auto seg = o2::mch::mapping::impl3::createCathodeSegmentation(detElemId, isBendingPlane);
  return seg ? new MchCathodeSegmentation(seg) : nullptr;
}

O2MCHMAPPINGIMPL3_EXPORT
void mchCathodeSegmentationDestruct(MchCathodeSegmentationHandle sh) { delete sh; }

O2MCHMAPPINGIMPL3_EXPORT
int mchCathodeSegmentationId(MchCathodeSegmentationHandle segHandle)
{
  // return segHandle->impl->getId();
  return -1;
}

O2MCHMAPPINGIMPL3_EXPORT
int mchCathodeSegmentationFindPadByPosition(MchCathodeSegmentationHandle segHandle, double x, double y)
{
  return segHandle->impl->findPadByPosition(x, y);
}

O2MCHMAPPINGIMPL3_EXPORT
int mchCathodeSegmentationFindPadByFEE(MchCathodeSegmentationHandle segHandle, int dualSampaId, int dualSampaChannel)
{
  return segHandle->impl->findPadByFEE(dualSampaId, dualSampaChannel);
}

O2MCHMAPPINGIMPL3_EXPORT
void mchCathodeSegmentationForEachDetectionElement(MchDetectionElementHandler handler, void* clientData)
{
  for (auto detElemId :
       {100, 101, 102, 103, 200, 201, 202, 203, 300, 301, 302, 303, 400, 401, 402, 403, 500, 501,
        502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 600, 601,
        602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 700, 701,
        702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719,
        720, 721, 722, 723, 724, 725, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811,
        812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 900, 901, 902, 903,
        904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921,
        922, 923, 924, 925, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
        1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025}) {
    handler(clientData, detElemId);
  }
}

O2MCHMAPPINGIMPL3_EXPORT
void mchCathodeSegmentationForEachDualSampa(MchCathodeSegmentationHandle segHandle, MchDualSampaHandler handler, void* clientData)
{
  for (auto dualSampaId : segHandle->impl->dualSampaIds()) {
    handler(clientData, dualSampaId);
  }
}

O2MCHMAPPINGIMPL3_EXPORT
void mchCathodeSegmentationForOneDetectionElementOfEachSegmentationType(MchDetectionElementHandler handler, void* clientData)
{
  for (auto detElemId :
       {100, 300, 500, 501, 502, 503, 504, 600, 601, 602, 700, 701, 702, 703, 704, 705, 706, 902, 903, 904, 905}) {
    handler(clientData, detElemId);
  }
}

O2MCHMAPPINGIMPL3_EXPORT
int mchCathodeSegmentationIsPadValid(MchCathodeSegmentationHandle segHandle, int catPadIndex)
{
  return catPadIndex != segHandle->impl->InvalidCatPadIndex;
}

O2MCHMAPPINGIMPL3_EXPORT
void mchCathodeSegmentationForEachPadInDualSampa(MchCathodeSegmentationHandle segHandle, int dualSampaId, MchPadHandler handler,
                                                 void* clientData)
{
  for (auto p : segHandle->impl->getCatPadIndexs(dualSampaId)) {
    handler(clientData, p);
  }
}

O2MCHMAPPINGIMPL3_EXPORT
double mchCathodeSegmentationPadPositionX(MchCathodeSegmentationHandle segHandle, int catPadIndex)
{
  return segHandle->impl->padPositionX(catPadIndex);
}

O2MCHMAPPINGIMPL3_EXPORT
double mchCathodeSegmentationPadPositionY(MchCathodeSegmentationHandle segHandle, int catPadIndex)
{
  return segHandle->impl->padPositionY(catPadIndex);
}

O2MCHMAPPINGIMPL3_EXPORT
double mchCathodeSegmentationPadSizeX(MchCathodeSegmentationHandle segHandle, int catPadIndex)
{
  return segHandle->impl->padSizeX(catPadIndex);
}

O2MCHMAPPINGIMPL3_EXPORT
double mchCathodeSegmentationPadSizeY(MchCathodeSegmentationHandle segHandle, int catPadIndex)
{
  return segHandle->impl->padSizeY(catPadIndex);
}

O2MCHMAPPINGIMPL3_EXPORT
int mchCathodeSegmentationPadDualSampaId(MchCathodeSegmentationHandle segHandle, int catPadIndex)
{
  return segHandle->impl->padDualSampaId(catPadIndex);
}

O2MCHMAPPINGIMPL3_EXPORT
int mchCathodeSegmentationPadDualSampaChannel(MchCathodeSegmentationHandle segHandle, int catPadIndex)
{
  return segHandle->impl->padDualSampaChannel(catPadIndex);
}

O2MCHMAPPINGIMPL3_EXPORT
void mchCathodeSegmentationForEachPadInArea(MchCathodeSegmentationHandle segHandle, double xmin, double ymin, double xmax,
                                            double ymax, MchPadHandler handler, void* clientData)
{
  for (auto p : segHandle->impl->getCatPadIndexs(xmin, ymin, xmax, ymax)) {
    handler(clientData, p);
  }
}

O2MCHMAPPINGIMPL3_EXPORT
void mchCathodeSegmentationForEachNeighbouringPad(MchCathodeSegmentationHandle segHandle, int catPadIndex, MchPadHandler handler,
                                                  void* userData)
{
  for (auto p : segHandle->impl->getNeighbouringCatPadIndexs(catPadIndex)) {
    handler(userData, p);
  }
}
} // extern "C"
