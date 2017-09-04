// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimuClusterShaper.h
/// \brief Cluster shaper for the ALPIDE response simulation

#ifndef ALICEO2_ITSMFT_SIMUCLUSTERSHAPER_H_
#define ALICEO2_ITSMFT_SIMUCLUSTERSHAPER_H_

///////////////////////////////////////////////////////////////////
//                                                               //
// Class to generate the cluster shape in the ITSU simulation    //
// Author: Davide Pagano                                         //
///////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <sstream>

#include "ITSMFTSimulation/ClusterShape.h"

namespace o2 {
  namespace ITSMFT {

    class SimuClusterShaper : public TObject {

    public:
      SimuClusterShaper();
      SimuClusterShaper(const UInt_t &cs);
      ~SimuClusterShaper() override;
      void FillClusterRandomly();
      void FillClusterSorted();
      inline void SetFireCenter(Bool_t v) {
        mFireCenter = v;
      }
      void AddNoisePixel();

      inline void    SetHit(Int_t ix, Int_t iz, Float_t x, Float_t z, const SegmentationPixel* seg) {
        mHitC = ix;
        mHitR = iz;
        mHitX = x;
        mHitZ = z;
        mSeg  = seg;
      }
      inline UInt_t  GetNRows() {return mCShape->GetNRows();}
      inline UInt_t  GetNCols() {return mCShape->GetNCols();}
      inline void    GetShape(std::vector<UInt_t>& v) {mCShape->GetShape(v);}
      inline UInt_t  GetCenterR() {return mCShape->GetCenterR();}
      inline UInt_t  GetCenterC() {return mCShape->GetCenterC();}
      inline size_t  GetCS() {return mCShape->GetNFiredPixels();}

      inline std::string ShapeSting(UInt_t cs, UInt_t *cshape) const {
        std::stringstream out;
        for (Int_t i = 0; i < cs; ++i) {
          out << cshape[i];
          if (i < cs-1) out << " ";
        }
        return out.str();
      }

    private:
      void ReComputeCenters();

      Float_t mHitX;
      Float_t mHitZ;
      Int_t   mHitC;
      Int_t   mHitR;
      Bool_t  mFireCenter;
      UInt_t  mNpixOn;
      const SegmentationPixel* mSeg;
      ClusterShape *mCShape;
    };
  }
}
#endif
