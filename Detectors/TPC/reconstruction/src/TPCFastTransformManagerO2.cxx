// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCFastTransformManagerO2.cxx
/// \author Sergey Gorbunov

#include "TPCReconstruction/TPCFastTransformManagerO2.h"

#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCFastTransform.h"
#include "Riostream.h"
#include "FairLogger.h"

using namespace ali_tpc_common::tpc_fast_transformation;

namespace o2 {
namespace TPC {

TPCFastTransformManagerO2::TPCFastTransformManagerO2()
  : mLastTimeBin(-1)
{
}

int TPCFastTransformManagerO2::create(TPCFastTransform& fastTransform, Long_t TimeStamp)
{
  /// Initializes TPCFastTransform object

  const static ParameterDetector& detParam = ParameterDetector::defaultInstance();
  const static ParameterGas& gasParam = ParameterGas::defaultInstance();
  const static ParameterElectronics& elParam = ParameterElectronics::defaultInstance();
  
  double vDrift = (elParam.getZBinWidth() * gasParam.getVdrift());

  // find last calibrated time bin
  
  mLastTimeBin = detParam.getTPClength() / vDrift  + 1;

  Mapper& mapper = Mapper::instance();
  const int nRows = mapper.getNumberOfRows();  

  fastTransform.startConstruction( nRows );

  TPCDistortionIRS& distortion = fastTransform.getDistortionNonConst();
  
  distortion.startConstruction( nRows, 1 );
  
  float tpcZlengthSideA = detParam.getTPClength();
  float tpcZlengthSideC = detParam.getTPClength();

  fastTransform.setTPCgeometry(  tpcZlengthSideA, tpcZlengthSideC );
  distortion.setTPCgeometry(  tpcZlengthSideA, tpcZlengthSideC );

  for( int iRow=0; iRow<fastTransform.getNumberOfRows(); iRow++){
    Sector sector = 0;
    int regionNumber = 0;
    while (iRow >= mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber))
      regionNumber++;
 
    const PadRegionInfo& region = mapper.getPadRegionInfo(regionNumber);

    int nPads = mapper.getNumberOfPadsInRowSector(iRow);
    float padWidth = region.getPadWidth();

    const GlobalPadNumber pad = mapper.globalPadNumber(PadPos(iRow, nPads/2));
    const PadCentre& padCentre = mapper.padCentre(pad);
    float xRow = padCentre.X();

    fastTransform.setTPCrow(iRow, xRow, nPads, padWidth);
    distortion.setTPCrow( iRow, xRow, nPads, padWidth, 0 );
  }

  fastTransform.setCalibration( -1, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f );

  IrregularSpline2D3D spline;
  {
    int nKnotsU = 15;
    int nAxisTicksU = mapper.getNumberOfPadsInRowSector(10);
    int nKnotsV = 20;
    int nAxisTicksV = mLastTimeBin+1;
    float knotsU[nKnotsU];
    float knotsV[nKnotsV];
    for( int i=0; i<nKnotsU; i++ ) knotsU[i] = 1./(nKnotsU-1)*i;
    for( int i=0; i<nKnotsV; i++ ) knotsV[i] = 1./(nKnotsV-1)*i;
    
    // TODO: adjust the grid

    double d1 = 0.6;
    double d2 = 0.9 - d1;
    double d3 = 1.-d2 - d1;
    
    for( int i=0; i<5; i++ ){ // 5 bins in first 6% of drift
      knotsV[i] = i / 4. * d1;
    }
    for( int i=0; i<10; i++ ){ // 10 bins for 6% <-> 90%
      knotsV[4+i] = d1 + i/9. * d2;
    }
    for( int i=0; i<5; i++ ){ // 5 bins for last 90% <-> 100%
      knotsV[13+i] = d1 + d2 + i/4. * d3;
    }

    spline.construct( nKnotsU, knotsU, nAxisTicksU,
		      nKnotsV, knotsV, nAxisTicksV );
  }
  distortion.setApproximationScenario( 0, spline );  
  distortion.finishConstruction();
  fastTransform.finishConstruction();

  // check if calculated pad geometry is consistent with the map
  testGeometry(fastTransform);

  return updateCalibration( fastTransform, TimeStamp );
}

  
int TPCFastTransformManagerO2::updateCalibration( ali_tpc_common::tpc_fast_transformation::TPCFastTransform &fastTransform, Long_t TimeStamp )
{
  // Update the calibration with the new time stamp

  Long_t lastTS = fastTransform.getTimeStamp();
  
  // deinitialize

  fastTransform.setTimeStamp( -1 ); 

  if( TimeStamp < 0  ) return 0; 

  // search for the calibration database ...

  const static ParameterDetector& detParam = ParameterDetector::defaultInstance();
  const static ParameterGas& gasParam = ParameterGas::defaultInstance();
  const static ParameterElectronics& elParam = ParameterElectronics::defaultInstance();

  // calibration found, set the initialized status back

  fastTransform.setTimeStamp(lastTS);

  // less than 60 seconds from the previois time stamp, don't do anything

  if( lastTS>=0 && TMath::Abs(lastTS - TimeStamp ) <60 ) return 0; 
  
  // start the initialization
  
  fastTransform.setTimeStamp(TimeStamp);
  
  // find last calibrated time bin
  
  double vDrift = elParam.getZBinWidth() * gasParam.getVdrift();
  mLastTimeBin = detParam.getTPClength() / vDrift  + 1;

 
  // fast transform formula:
  // L = (t-t0)*(mVdrift + mVdriftCorrY*yLab ) + mLdriftCorr 
  // Z = Z(L) +  tpcAlignmentZ
  // spline distortions for xyz
  // Time-of-flight correction: ldrift += dist-to-vtx*tofCorr

  double t0 = 0.;
  double vdCorrY = 0.;
  double ldCorr = 0.;
  double tpcAlignmentZ = 0.;

  double tofCorr = 0.;
  double primVtxZ = 0.;

  fastTransform.setCalibration( TimeStamp, t0, vDrift, vdCorrY, ldCorr, tofCorr, primVtxZ, tpcAlignmentZ);

  // now calculate distortion map: dx,du,dv = ( origTransform() -> x,u,v) - fastTransformNominal:x,u,v

  TPCDistortionIRS& distortion = fastTransform.getDistortionNonConst();
  
  // switch TOF correction off for a while
 
  for( int slice=0; slice<distortion.getNumberOfSlices(); slice++){      
    for( int row=0; row<distortion.getNumberOfRows(); row++ ){
      //const TPCFastTransform::RowInfo &rowInfo = fastTransform.getRowInfo( row );
      const IrregularSpline2D3D& spline = distortion.getSpline( slice, row );
      float *data = distortion.getSplineDataNonConst(slice,row);
      for( int knot=0; knot<spline.getNumberOfKnots(); knot++ ){
	data[3*knot+0] = 0.f;
	data[3*knot+1] = 0.f;
	data[3*knot+2] = 0.f;	
      } // knots      
      spline.correctEdges(data);
    } // row
  } // slice

  // set back the time-of-flight correction

  return 0;
}

void TPCFastTransformManagerO2::testGeometry(const ali_tpc_common::tpc_fast_transformation::TPCFastTransform& fastTransform) const
{
  Mapper& mapper = Mapper::instance();

  if (fastTransform.getNumberOfSlices() != Sector::MAXSECTOR) {
    LOG(FATAL) << "Wrong number of sectors :" << fastTransform.getNumberOfSlices() << " instead of " << Sector::MAXSECTOR << std::endl;
  }

  if (fastTransform.getNumberOfRows() != mapper.getNumberOfRows()) {
    LOG(FATAL) << "Wrong number of rows :" << fastTransform.getNumberOfRows() << " instead of " << mapper.getNumberOfRows() << std::endl;
  }

  double maxDx = 0, maxDy = 0;

  for (int row = 0; row < fastTransform.getNumberOfRows(); row++) {

    int nPads = fastTransform.getRowInfo(row).maxPad + 1;

    if (nPads != mapper.getNumberOfPadsInRowSector(row)) {
      LOG(FATAL) << "Wrong number of pads :" << nPads << " instead of " << mapper.getNumberOfPadsInRowSector(row) << std::endl;
    }

    double x = fastTransform.getRowInfo(row).x;

    // check if calculated pad positions are equal to the real ones

    for (int pad = 0; pad < nPads; pad++) {
      const GlobalPadNumber p = mapper.globalPadNumber(PadPos(row, pad));
      const PadCentre& c = mapper.padCentre(p);
      float u = 0, v = 0;
      int err = fastTransform.convPadTimeToUV(0, row, pad, 10., u, v, 0.);
      if( err!=0 ){
	LOG(FATAL) << "Can not transform a cluster: row " << row << " pad " << pad << " time 10. : error " << err << std::endl;
      }

      double dx = x - c.X();
      double dy = u - (-c.Y()); // diferent sign convention for Y coordinate in the map

      if (fabs(dx) >= 1.e-6 || fabs(dy) >= 1.e-5) {
        LOG(WARNING) << "wrong calculated pad position:"
                     << " row " << row << " pad " << pad << " x calc " << x << " x in map " << c.X() << " dx " << (x - c.X())
                     << " y calc " << u << " y in map " << -c.Y() << " dy " << dy << std::endl;
      }
      if (fabs(maxDx) < fabs(dx))
        maxDx = dx;
      if (fabs(maxDy) < fabs(dy))
        maxDy = dy;
    }
  }

  if (fabs(maxDx) >= 1.e-4 || fabs(maxDy) >= 1.e-4) {
    LOG(FATAL) << "wrong calculated pad position:"
               << " max Dx " << maxDx << " max Dy " << maxDy << std::endl;
  }
}
}
} // namespaces
