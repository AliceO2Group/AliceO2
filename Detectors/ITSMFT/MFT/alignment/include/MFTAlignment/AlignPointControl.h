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

/// \file MilleRecordStore.h
/// \author arakotoz@cern.ch
/// \brief Class dedicated to write AlignPointInfo to output file for MFT

#ifndef ALICEO2_MFT_ALIGN_POINT_CONTROL_H
#define ALICEO2_MFT_ALIGN_POINT_CONTROL_H

#include <Rtypes.h>
#include <TString.h>
#include "MFTAlignment/AlignPointHelper.h"

class TFile;
class TTree;

namespace o2
{
namespace mft
{

class AlignPointControl
{
 public:
  struct AlignPointInfo {
    UShort_t sensor;          // sensor id
    UShort_t layer;           // layer id
    UShort_t disk;            // disk id
    UShort_t half;            // half id
    Double_t measuredGlobalX; // cluster x, global frame (cm)
    Double_t measuredGlobalY; // cluster y, global frame (cm)
    Double_t measuredGlobalZ; // cluster z, global frame (cm)
    Double_t measuredLocalX;  // cluster x, local frame (cm)
    Double_t measuredLocalY;  // cluster y, local frame (cm)
    Double_t measuredLocalZ;  // cluster z, local frame (cm)
    Double_t residualX;       // track global x - cluster global x (cm)
    Double_t residualY;       // track global y - cluster global y (cm)
    Double_t residualZ;       // track global z - cluster global z (cm)
    Double_t residualLocalX;  // track local x - cluster local x (cm)
    Double_t residualLocalY;  // track local y - cluster local y (cm)
    Double_t residualLocalZ;  // track local z - cluster local z (cm)
    Double_t recoGlobalX;     // track x, global frame (cm)
    Double_t recoGlobalY;     // track y, global frame (cm)
    Double_t recoGlobalZ;     // track z, global frame (cm)
    Double_t recoLocalX;      // track x, local frame (cm)
    Double_t recoLocalY;      // track y, local frame (cm)
    Double_t recoLocalZ;      // track z, local frame (cm)
  };
  /// \brief constructor
  AlignPointControl();

  /// \brief destructor
  virtual ~AlignPointControl();

  /// \brief Set the number of entries to be used by TTree::AutoSave()
  void setCyclicAutoSave(const long nEntries);

  /// \brief choose filename
  void setOutFileName(TString fname) { mOutFileName = fname; }

  /// \brief init output file and tree
  void init();

  /// \brief check if init went well
  bool isInitOk() const { return mIsSuccessfulInit; }

  /// \brief fill the tree from an align point
  void fill(o2::mft::AlignPointHelper* aPoint,
            const int iTrack = 0,
            const bool doPrint = false);

  /// \brief write tree and close output file
  void terminate();

 protected:
  TTree* mControlTree;       ///< the ROOT TTree container
  TFile* mControlFile;       ///< the output file
  bool mIsSuccessfulInit;    ///< boolean to monitor the success of the initialization
  long mNEntriesAutoSave;    ///< max entries in the buffer after which TTree::AutoSave() is automatically used
  TString mOutFileName;      ///< name of the output file that will store the TTree
  TString mTreeTitle;        ///< title of the TTree
  AlignPointInfo mPointInfo; ///< information to be written to the output TTree

  bool setControlPoint(o2::mft::AlignPointHelper* aPoint);

  ClassDef(AlignPointControl, 0);
};
} // namespace mft
} // namespace o2

#endif
