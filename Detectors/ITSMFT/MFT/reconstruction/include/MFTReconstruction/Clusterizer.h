// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//// In applying this license CERN does not waive the privileges and immunities// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// \file Clusterizer.h
/// \brief Implementation of the cluster finder 
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_CLUSTERIZER_H_
#define ALICEO2_MFT_CLUSTERIZER_H_

class TClonesArray;

namespace o2 
{
  namespace ITSMFT 
  {
    class SegmentationPixel;
  }
}

namespace o2 
{
  namespace MFT 
  {
    class Clusterizer
    {
      
    public:
      
      Clusterizer();
      ~Clusterizer();
      
      Clusterizer(const Clusterizer&) = delete;
      Clusterizer& operator=(const Clusterizer&) = delete;
      
      /// Steer conversion of points to digits
      /// @param points Container with ITS points
      /// @return digits container
      void process(const o2::ITSMFT::SegmentationPixel *seg, const TClonesArray* digits, TClonesArray* clusters);
      
    };
  }
}

#endif
