// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterizerTask.h
/// \brief Task driving the cluster finding from digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_CLUSTERIZERTASK_H
#define ALICEO2_MFT_CLUSTERIZERTASK_H

#include "FairTask.h"

#include "MFTBase/GeometryTGeo.h"
#include "MFTReconstruction/PixelReader.h"
#include "MFTReconstruction/Clusterizer.h"

class TClonesArray;

namespace o2 
{
  namespace MFT 
  {
    class EventHeader; 
    class ClusterizerTask : public FairTask
    {
      
    public:
      
      ClusterizerTask();
      ~ClusterizerTask() override;
      
      InitStatus Init() override;
      void Exec(Option_t* opt) override;
      
    private:
      
      GeometryTGeo mGeometry;    ///< MFT geometry
      DigitPixelReader mReader;  ///< Pixel reader
      Clusterizer mClusterizer;  ///< Cluster finder

      TClonesArray* mClustersArray; ///< Array of clusters

      ClassDefOverride(ClusterizerTask,1);
      
    };    
  }
}

#endif
