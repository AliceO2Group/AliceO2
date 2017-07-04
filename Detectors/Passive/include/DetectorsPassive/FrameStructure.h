// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// -------------------------------------------------------------------------
// -----             Implementation of the AliFrame structure          -----
// -----                Created 05/07/17  by S. Wenzel                 -----
// -------------------------------------------------------------------------


#ifndef ALICEO2_GEOMBASE_FRAMESTRUCTURE_H
#define ALICEO2_GEOMBASE_FRAMESTRUCTURE_H

#include <FairModule.h>

namespace o2 {
namespace Passive {

// class supposed to provide the frame support structure common to TOF and TRD
class FrameStructure : public FairModule {
 public:
    FrameStructure(const char* Name = "FrameStruct", const char * title = "FrameStruct");

    /**      default constructor    */
    FrameStructure() = default;

    /**       destructor     */
    ~FrameStructure() override = default;

    /**      Create the module geometry  */
    void ConstructGeometry() override;

private:
    void MakeHeatScreen(const char* name, Float_t dyP, Int_t rot1, Int_t rot2);
    void WebFrame(const char* name, Float_t dHz, Float_t theta0, Float_t phi0);

    bool mCaveIsAvailable = false; ///! if the mother volume is available (to hook the frame)
    
    ClassDefOverride(FrameStructure, 1);
};

}
}

#endif
