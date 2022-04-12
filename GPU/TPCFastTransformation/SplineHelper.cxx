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

/// \file  SplineHelper.cxx
/// \brief Implementation of SplineHelper class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

#include "SplineHelper.h"
#include "Spline2D.h"

#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TDecompBK.h"

#include <vector>
#include "TRandom.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TNtuple.h"
#include "TFile.h"
#include "GPUCommonMath.h"
#include <iostream>

using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT>
SplineHelper<DataT>::SplineHelper() : mError(), mXdimensions(0), mFdimensions(0), mNumberOfDataPoints(0), mHelpers()
{
}

template <typename DataT>
int SplineHelper<DataT>::storeError(int code, const char* msg)
{
  mError = msg;
  return code;
}

////////////////
// pointstoarray
// HILFSFUNKTION,
template <typename DataT>
int SplineHelper<DataT>::pointstoarray(const int indices[], const int numbers[], int dim)
{

  int result = 0;
  int factor = 1;
  for (int i = 0; i < dim; i++) {
    result += indices[i] * factor;
    factor *= numbers[i];
  }
  return result;
}

////////////////
//arraytopoints
// HILFSFUNKTION
template <typename DataT>
int SplineHelper<DataT>::arraytopoints(int point, int result[], const int numbers[], int dim)
{

  if (point == 0) {
    for (int i = 0; i < dim; i++) {
      result[i] = 0;
    }
  } else {
    int divisor = 1;
    int modoperand = 1;
    for (int i = 0; i < dim; i++) {
      modoperand *= numbers[i];
      result[i] = (int)((point % modoperand) / divisor);
      divisor *= numbers[i];
    }
  }
  return 0;
}

template <typename DataT>
void SplineHelper<DataT>::approximateFunction(
  DataT* Fparameters, const double xMin[/* mXdimensions */], const double xMax[/* mXdimensions */],
  std::function<void(const double x[/* mXdimensions */], double f[/* mFdimensions */])> F) const
{
  /// Create best-fit spline parameters for a given input function F
  /// output in Fparameter
  // TODO: implement
  // MY VERSION
  // LOG(info) << "approximateFunction(Fparameters, xMin[],xMax[],F) :" ;
  double scaleX[mXdimensions];
  for (int i = 0; i < mXdimensions; i++) {
    scaleX[i] = (xMax[i] - xMin[i]) / ((double)(mHelpers[i].getSpline().getUmax()));
  }

  // calculate F-Values at all datapoints:
  int nrOfAllPoints = getNumberOfDataPoints();
  std::vector<double> dataPointF(nrOfAllPoints * mFdimensions);

  int nrOfPoints[mXdimensions];
  for (int i = 0; i < mXdimensions; i++) {
    nrOfPoints[i] = mHelpers[i].getNumberOfDataPoints();
  }
  double x[mXdimensions];
  for (int d = 0; d < nrOfAllPoints; d++) { // for all DataPoints

    int indices[mXdimensions];
    int modoperand = 1;
    int divisor = 1;

    // get the DataPoint index
    for (int i = 0; i < mXdimensions; i++) {
      modoperand *= nrOfPoints[i];

      indices[i] = (int)((d % modoperand) / divisor);
      divisor *= nrOfPoints[i];
      // get the respecting u-values:
      x[i] = xMin[i] + mHelpers[i].getDataPoint(indices[i]).u * scaleX[i];
    }

    for (int j = 0; j < mXdimensions; j++) {
      F(x, &dataPointF[d * mFdimensions]);
    }

  } // end for all DataPoints d
  //END MY VERSION

  //std::vector<DataT> dataPointF(getNumberOfDataPoints() * mFdimensions);
  //DUMYY VERSION Commented out
  /* for (int i = 0; i < getNumberOfDataPoints() * mFdimensions; i++) {
    dataPointF[i] = 1.;
  } */
  /*
  double scaleX1 = (x1Max - x1Min) / ((double)mHelperU1.getSpline().getUmax());
  double scaleX2 = (x2Max - x2Min) / ((double)mHelperU2.getSpline().getUmax());

  for (int iv = 0; iv < getNumberOfDataPointsU2(); iv++) {
    DataT x2 = x2Min + mHelperU2.getDataPoint(iv).u * scaleX2;
    for (int iu = 0; iu < getNumberOfDataPointsU1(); iu++) {
      DataT x1 = x1Min + mHelperU1.getDataPoint(iu).u * scaleX1;
      F(x1, x2, &dataPointF[(iv * getNumberOfDataPointsU1() + iu) * mFdimensions]);
    }
  }
  */
  approximateFunction(Fparameters, dataPointF.data());
}

template <typename DataT>
void SplineHelper<DataT>::approximateFunctionBatch(
  DataT* Fparameters, const double xMin[], const double xMax[],
  std::function<void(const std::vector<double> x[], double f[/*mFdimensions*/])> F,
  unsigned int batchsize) const
{
  /// Create best-fit spline parameters for a given input function F.
  /// F calculates values for a batch of points.
  /// output in Fparameters

  double scaleX[mXdimensions];
  for (int i = 0; i < mXdimensions; i++) {
    scaleX[i] = (xMax[i] - xMin[i]) / ((double)(mHelpers[i].getSpline().getUmax()));
  }

  const int nrOfAllPoints = getNumberOfDataPoints();
  std::vector<double> dataPointF(nrOfAllPoints * mFdimensions);

  int nrOfPoints[mXdimensions];
  for (int i = 0; i < mXdimensions; i++) {
    nrOfPoints[i] = mHelpers[i].getNumberOfDataPoints();
  }

  std::vector<double> x[mXdimensions];
  for (int iDim = 0; iDim < mXdimensions; ++iDim) {
    x[iDim].reserve(batchsize);
  }

  unsigned int ibatch = 0;
  int index = 0;
  for (int d = 0; d < nrOfAllPoints; d++) { // for all DataPoints
    int indices[mXdimensions];
    int modoperand = 1;
    int divisor = 1;

    // get the DataPoint index
    for (int i = 0; i < mXdimensions; i++) {
      modoperand *= nrOfPoints[i];
      indices[i] = (int)((d % modoperand) / divisor);
      divisor *= nrOfPoints[i];
      // get the respecting u-values:
      x[i].emplace_back(xMin[i] + mHelpers[i].getDataPoint(indices[i]).u * scaleX[i]);
    }
    ++ibatch;

    if (ibatch == batchsize || d == nrOfAllPoints - 1) {
      ibatch = 0;

      F(x, &dataPointF[index]);
      index = (d + 1) * mFdimensions;

      for (int iDim = 0; iDim < mXdimensions; ++iDim) {
        x[iDim].clear();
      }
    }
  } // end for all DataPoints d

  approximateFunction(Fparameters, dataPointF.data());
}

template <typename DataT>
void SplineHelper<DataT>::approximateFunction(
  DataT* Fparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
{
  /// approximate a function given as an array of values at data points

  int numberOfKnots[mXdimensions]; //getting number of Knots for all dimensions into one array
  for (int i = 0; i < mXdimensions; i++) {
    numberOfKnots[i] = mHelpers[i].getSpline().getNumberOfKnots();
  }

  int numberOfDataPoints[mXdimensions]; // getting number of datapoints (incl knots) in all dimensions into one array
  for (int i = 0; i < mXdimensions; i++) {
    numberOfDataPoints[i] = mHelpers[i].getNumberOfDataPoints();
  }

  int numberOfAllKnots = 1; // getting Number of all knots for the entire spline
  for (int i = 0; i < mXdimensions; i++) {
    numberOfAllKnots *= numberOfKnots[i];
  }
  // TO BE REMOVED (TEST-OUTPUT):
  LOG(info) << "total number of knots: " << numberOfAllKnots << ", ";

  int numberOfAllDataPoints = 1; // getting Number of all Datapoints for the entire spline
  for (int i = 0; i < mXdimensions; i++) {
    numberOfAllDataPoints *= numberOfDataPoints[i];
    // LOG(info) << mHelpers[0].getNumberOfDataPoints();
  }

  // TO BE REMOVED TEST:
  // LOG(info) << "total number of DataPoints (including knots): " <<  numberOfAllDataPoints << ", ";

  int numberOfParameterTypes = (int)(pow(2.0, mXdimensions)); //number of Parameters per Knot

  // TO BE REMOVED TEST:
  // LOG(info) << "number of paramtertypes per knot : " <<  numberOfParameterTypes << ", ";

  std::unique_ptr<double[]> allParameters[numberOfParameterTypes]; //Array for the different parametertypes s, s'u, s'v, s''uv,...
  for (int i = 0; i < numberOfParameterTypes; i++) {
    allParameters[i] = std::unique_ptr<double[]>(new double[numberOfAllDataPoints * mFdimensions]); //To-Do:Fdim!!
  }
  //filling allParameters[0] and FParameters with s:
  for (int i = 0; i < numberOfAllDataPoints; i++) {
    for (int f = 0; f < mFdimensions; f++) {                                     // for all f-dimensions
      allParameters[0][i * mFdimensions + f] = DataPointF[i * mFdimensions + f]; // TO DO - Just get the pointer adress there PLEASE!
    }
    int p0indices[mXdimensions];
    arraytopoints(i, p0indices, numberOfDataPoints, mXdimensions);
    bool isKnot = 1;
    for (int j = 0; j < mXdimensions; j++) { // is the current datapoint a knot?
      if (!mHelpers[j].getDataPoint(p0indices[j]).isKnot) {
        isKnot = 0;
        break;
      }
    }
    if (isKnot) {
      int knotindices[mXdimensions];
      for (int j = 0; j < mXdimensions; j++) { // calculate KNotindices for all dimensions
        //WORKAROUND Getting Knotindices:
        knotindices[j] = p0indices[j] / ((numberOfDataPoints[j] - 1) / (numberOfKnots[j] - 1));
        //knotindices[j] = mHelpers[j].getDataPoint(p0indices[j]).iKnot; //in der Annahme der wert ist ein Knotenindex und falls der datapoint ein knoten ist, gibt er seinen eigenen knotenindex zurück
      }
      // get the knotindexvalue for FParameters:
      int knotind = pointstoarray(knotindices, numberOfKnots, mXdimensions);

      for (int f = 0; f < mFdimensions; f++) {                                                               // for all f-dimensions get function values into Fparameters
        Fparameters[knotind * numberOfParameterTypes * mFdimensions + f] = DataPointF[i * mFdimensions + f]; ///write derivatives in FParameters
      }
    } // end if isKnot
  }   //end i (filling DataPointF Values into allParameters[0] and FParameters)
  //now: allParameters[0] = dataPointF;

  //Array for input DataPointF-values for Spline1D::approximateFunctionGradually(...);
  std::unique_ptr<double[]> dataPointF1D[mXdimensions];
  for (int i = 0; i < mXdimensions; i++) {
    dataPointF1D[i] = std::unique_ptr<double[]>(new double[numberOfDataPoints[i] * mFdimensions]); // To-Do:Fdim!! For s and derivetives at all knots.
  }
  //Array to be filled by Spline1D::approximateFunctionGradually(...);
  std::unique_ptr<DataT[]> par[mXdimensions];
  std::unique_ptr<double[]> parD[mXdimensions];

  for (int i = 0; i < mXdimensions; i++) {
    par[i] = std::unique_ptr<DataT[]>(new DataT[numberOfKnots[i] * mFdimensions * 2]);
    parD[i] = std::unique_ptr<double[]>(new double[numberOfKnots[i] * mFdimensions * 2]);
  }

  // LOG(info) << "NumberOfParameters: " <<  mNumberOfParameters ;

  //STARTING MAIN-LOOP, for all Parametertypes:
  for (int p = 1; p < numberOfParameterTypes; p++) { // p = 1!! Wir kriegen s (p0) durch approximateFunction()oben
    int dimension = 0;                               // find the dimension for approximation
    for (int i = (int)(log2f((float)p)); i >= 0; i--) {
      if (p % (int)(pow(2.0, i)) == 0) {
        dimension = i;
        break;
      }
    }

    int currentDataPointF = p - (int)(pow(2.0, dimension));
    // LOG(info) << "\n" << "p:" << p << ", dim of approximation: " << dimension << ", based on: " << currentDataPointF ;

    int nrOf1DSplines = (numberOfAllDataPoints / numberOfDataPoints[dimension]); // number of Splines for Parametertyp p in direction dim
    // LOG(info) << "nr of splines: " << nrOf1DSplines;

    // getting the numbers of Datapoints for all dimension eccept the dimension of interpolation
    int currentNumbers[mXdimensions - 1];
    for (int i = 0; i < dimension; i++) {
      currentNumbers[i] = numberOfDataPoints[i];
    }
    for (int i = dimension; i < mXdimensions - 1; i++) {
      currentNumbers[i] = numberOfDataPoints[i + 1];
    }
    /// LOG(info) << " current numbers: ";
    for (int i = 0; i < mXdimensions - 1; i++) {
      // LOG(info) << currentNumbers[i] << ",";
    }
    // LOG(info) ;

    //// for all Splines in current dimension:
    for (int s = 0; s < nrOf1DSplines; s++) {
      int indices[mXdimensions - 1];
      arraytopoints(s, indices, currentNumbers, mXdimensions - 1);
      int startpoint[mXdimensions]; //startpoint for the current 1DSpline
      for (int i = 0; i < dimension; i++) {
        startpoint[i] = indices[i];
      }
      startpoint[dimension] = 0;
      for (int i = dimension + 1; i < mXdimensions; i++) {
        startpoint[i] = indices[i - 1];
      }
      // NOW WE HAVE THE DATAPOINTINDICES OF THE CURRENT STARTPOINT IN startpoint-Array.
      int startdatapoint = pointstoarray(startpoint, numberOfDataPoints, mXdimensions);
      int distance = 1; // distance to the next dataPoint in the array for the current dimension
      for (int i = 0; i < dimension; i++) {
        distance *= numberOfDataPoints[i];
      }
      distance *= mFdimensions;

      for (int i = 0; i < numberOfDataPoints[dimension]; i++) { //Fill the dataPointF1D-Array
        for (int f = 0; f < mFdimensions; f++) {
          dataPointF1D[dimension][i * mFdimensions + f] = allParameters[currentDataPointF][startdatapoint * mFdimensions + (i * distance + f)]; // uiuiui index kuddelmuddel???!!
        }
      }
      mHelpers[dimension].approximateFunction(par[dimension].get(), dataPointF1D[dimension].get());
      for (int i = 0; i < numberOfKnots[dimension] * mFdimensions * 2; i++) {
        parD[dimension][i] = par[dimension][i];
      }
      // now we have all s and s' values in par[dimension]

      int redistributionindex[mXdimensions];
      for (int i = 0; i < mXdimensions; i++) {
        redistributionindex[i] = startpoint[i];
      }
      //redistributing the derivatives at dimension-Knots into array p
      for (int i = 0; i < numberOfKnots[dimension]; i++) {                        //for all dimension-Knots
        redistributionindex[dimension] = mHelpers[dimension].getKnotDataPoint(i); //find the indices
        int finalposition = pointstoarray(redistributionindex, numberOfDataPoints, mXdimensions);

        for (int f = 0; f < mFdimensions; f++) {
          allParameters[p][finalposition * mFdimensions + f] = par[dimension][2 * i * mFdimensions + mFdimensions + f];
        }

        bool isKnot = 1;
        for (int j = 0; j < mXdimensions; j++) { //is dataPoint a knot?
          if (!mHelpers[j].getDataPoint(redistributionindex[j]).isKnot) {
            isKnot = 0;
            break;
          } //noch mal checken!! Das muss noch anders!!
        }

        if (isKnot) { // for all knots
          int knotindices[mXdimensions];

          for (int j = 0; j < mXdimensions; j++) { // calculate Knotindices for all dimensions
            knotindices[j] = redistributionindex[j] / ((numberOfDataPoints[j] - 1) / (numberOfKnots[j] - 1));
            //knotindices[j] = mHelpers[j].getDataPoint(redistributionindex[j]).iKnot; //in der Annahme der wert ist ein Knotenindex und falls der datapoint ein knoten ist, gibt er seinen eigenen knotenindex zurück
          }
          // get the knotindexvalue for FParameters:
          int knotind = pointstoarray(knotindices, numberOfKnots, mXdimensions);
          for (int f = 0; f < mFdimensions; f++) {
            Fparameters[knotind * numberOfParameterTypes * mFdimensions + p * mFdimensions + f] = par[dimension][2 * i * mFdimensions + mFdimensions + f]; ///write derivatives in FParameters
          }
        }
      } // end for all fknots (for redistribution)

      // recalculation:
      for (int i = 0; i < numberOfDataPoints[dimension]; i++) { // this is somehow still redundant// TO DO: ONLY PART OF approximateFunction WHERE NDIM is considerd!!
        redistributionindex[dimension] = i;                     // getting current datapointindices
        bool isKnot = 1;                                        // check is current datapoint a knot?
        for (int j = 0; j < mXdimensions; j++) {
          if (!mHelpers[j].getDataPoint(redistributionindex[j]).isKnot) {
            isKnot = 0;
            break;
          }
        }
        double splineF[mFdimensions];
        double u = mHelpers[dimension].getDataPoint(i).u;
        mHelpers[dimension].getSpline().interpolateU(mFdimensions, parD[dimension].get(), u, splineF); //recalculate at all datapoints of dimension
        for (int dim = 0; dim < mFdimensions; dim++) {                                                 //writing it in allParameters
          // LOG(info)<<allParameters [p-(int)(pow(2.0, dimension))] [(int)(startdatapoint*mFdimensions + i*distance + dim)]<<", ";
          allParameters[p - (int)(pow(2.0, dimension))][(int)(startdatapoint * mFdimensions + i * distance + dim)] = splineF[dim]; //write it in the array.
          // LOG(info)<<allParameters [p-(int)(pow(2.0, dimension))] [(int)(startdatapoint*mFdimensions + i*distance + dim)]<<",   ";
        }

        if (isKnot) {
          int knotindices[mXdimensions];

          for (int j = 0; j < mXdimensions; j++) { // calculate KNotindices for all dimensions
            knotindices[j] = redistributionindex[j] / ((numberOfDataPoints[j] - 1) / (numberOfKnots[j] - 1));
            //knotindices[j] = mHelpers[j].getDataPoint(redistributionindex[j]).iKnot; //in der Annahme der wert ist ein Knotenindex und falls der datapoint ein knoten ist, gibt er seinen eigenen knotenindex zurück
          }
          int currentknotarrayindex = pointstoarray(knotindices, numberOfKnots, mXdimensions);
          // getting the recalculated value into FParameters:
          for (int f = 0; f < mFdimensions; f++) {
            Fparameters[currentknotarrayindex * numberOfParameterTypes * mFdimensions + (p - (int)(pow(2.0, dimension))) * mFdimensions + f] = splineF[f];
          }
        } // end if isKnot
      }   // end recalculation
    }     //end of all1DSplines
  }       //end of for parametertypes
} //end of approxymateFunction MYVERSION!

template <typename DataT>
int SplineHelper<DataT>::test(const bool draw, const bool drawDataPoints)
{
  // Test method

  using namespace std;

  constexpr int nDimX = 2;
  constexpr int nDimY = 2;
  constexpr int Fdegree = 4;

  double xMin[nDimX];
  double xMax[nDimX];
  int nKnots[nDimX];
  int* knotsU[nDimX];
  int nAxiliaryDatapoints[nDimX];

  for (int i = 0; i < nDimX; i++) {
    xMin[i] = 0.;
    xMax[i] = 1.;
    nKnots[i] = 4;
    knotsU[i] = new int[nKnots[i]];
    nAxiliaryDatapoints[i] = 4;
  }

  // Function F
  const int nTerms1D = 2 * (Fdegree + 1);
  int nFcoeff = nDimY;
  for (int i = 0; i < nDimX; i++) {
    nFcoeff *= nTerms1D;
  }

  double Fcoeff[nFcoeff];

  auto F = [&](const double x[nDimX], double f[nDimY]) {
    double a[nFcoeff];
    a[0] = 1;
    int na = 1;
    for (int d = 0; d < nDimX; d++) {
      double b[nFcoeff];
      int nb = 0;
      double t = (x[d] - xMin[d]) * TMath::Pi() / (xMax[d] - xMin[d]);
      for (int i = 0; i < nTerms1D; i++) {
        double c = (i % 2) ? cos((i / 2) * t) : cos((i / 2) * t);
        for (int j = 0; j < na; j++) {
          b[nb++] = c * a[j];
          assert(nb <= nFcoeff);
        }
      }
      na = nb;
      for (int i = 0; i < nb; i++) {
        a[i] = b[i];
      }
    }

    double* c = Fcoeff;
    for (int dim = 0; dim < nDimY; dim++) {
      f[dim] = 0;
      for (int i = 0; i < na; i++) {
        f[dim] += a[i] * (*c++);
      }
    }
  };

  auto F2D = [&](double x1, double x2, double f[nDimY]) {
    double x[2] = {x1, x2};
    F(x, f);
  };

  for (int seed = 1; seed < 10; seed++) {

    gRandom->SetSeed(seed);

    // getting the coefficents filled randomly
    for (int i = 0; i < nFcoeff; i++) {
      Fcoeff[i] = gRandom->Uniform(-1, 1);
    }

    for (int i = 0; i < nDimX; i++) {
      knotsU[i][0] = 0;
      for (int j = 1; j < nKnots[i]; j++) {
        knotsU[i][j] = j * 4; //+ int(gRandom->Integer(3)) - 1;
      }
    }

    Spline<float, nDimX, nDimY> spline(nKnots, knotsU);
    Spline2D<float, nDimY> spline2D(nKnots[0], knotsU[0], nKnots[1], knotsU[1]);

    spline.approximateFunction(xMin, xMax, F, nAxiliaryDatapoints);
    spline2D.approximateFunction(xMin[0], xMax[0], xMin[1], xMax[1],
                                 F2D, nAxiliaryDatapoints[0], nAxiliaryDatapoints[0]);

    long double statDf = 0;
    long double statDf2D = 0;

    long double statN = 0;

    double x[nDimX];
    for (int i = 0; i < nDimX; i++) {
      x[i] = xMin[i];
    }
    do {
      float xf[nDimX];
      float s[nDimY];
      float s2D[nDimY];
      double f[nDimY];
      for (int i = 0; i < nDimX; i++) {
        xf[i] = x[i];
      }
      F(x, f);
      spline.interpolate(xf, s);
      spline2D.interpolate(xf[0], xf[1], s2D);

      for (int dim = 0; dim < nDimY; dim++) {
        statDf += (s[dim] - f[dim]) * (s[dim] - f[dim]);
        statDf2D += (s2D[dim] - f[dim]) * (s2D[dim] - f[dim]);
        statN++;
      }
      int dim = 0;
      for (; dim < nDimX; dim++) {
        x[dim] += 0.01;
        if (x[dim] <= xMax[dim]) {
          break;
        }
        x[dim] = xMin[dim];
      }
      if (dim >= nDimX) {
        break;
      }
    } while (1);

    LOG(info) << "\n std dev for SplineND   : " << sqrt(statDf / statN);
    LOG(info) << "\n std dev for Spline2D   : " << sqrt(statDf2D / statN);

  } // seed

  for (int i = 0; i < nDimX; i++) {
    delete[] knotsU[i];
  }

  return 0;
}

template class GPUCA_NAMESPACE::gpu::SplineHelper<float>;
template class GPUCA_NAMESPACE::gpu::SplineHelper<double>;

#endif
