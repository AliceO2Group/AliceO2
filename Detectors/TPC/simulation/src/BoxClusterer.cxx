/// \file AliTPCUpgradeBoxClusterer.cxx
/// \brief Boxclusterer for the TPC

/// Based on AliTPCdataQA
///
/// The BoxClusterer basically tries to find local maximums and builds up 5x5
/// clusters around these. So if c is the local maximum (c for center). It
/// will try to make a cluster like this:
///    --->  pad direction
///    o o o o o    |
///    o i i i o    |
///    o i C i o    V Time direction
///    o i i i o
///    o o o o o
///
/// The outer pad-time cells are only addded if the inner cell has signal. For
/// horizonal verticallly aligned inner cells we test like this:
///        o
///        i
///    o i C i o
///        i
///        o
/// For the diagonal cells we check like this.
///    o o   o o
///    o i   i o
///        C
///    o i   i o
///    o o   o o
///
/// The requirements for a local maxima is:
/// Charge in bin is >= 5 ADC channels.
/// Charge in bin is larger than all the 8 neighboring bins.
/// (in the case when it is similar only one of the clusters will be made, see
/// in the code)
/// At least one of the two pad neighbors has a signal.
/// (this requirent can be loosened up)
/// At least one of the two time neighbors has a signal.
///
/// The RAW data is "expanded" for each sector and stored in a big signal
/// array. Then a simple version of the code in AliTPCclusterer is used to
/// identify the local maxima.
/// With simple we mean here that:
/// 1) there is no attempt to take into account the noise information
/// 2) there is no special handling of small clusters
///
/// Implementation:
///
/// The data are expanded into 3 arrays CRU by CRU
/// ~~~
/// Float_t** mAllBins       2d array [row][bin(pad, time)] ADC signal
/// Int_t**   mAllSigBins    2d array [row][signal#] bin(with signal)
/// Int_t*    mAllNSigBins;  1d array [row] Nsignals
/// ~~~
/// To make sure that one never has to check if one is inside the sector or not
/// the arrays are larger than a sector. 2 pads and time bins are added both
/// as the beginning and the end.
///
/// When data from a new sector is encountered, the method
/// FindLocalMaxima is called on the data from the previous sector, and
/// the clusters are created.
///

/*

  Comments:

  o I have added several R__ASSERTs. I think these are great for development
  to make sure that things are called in the right order. However, for
  operation one might want to add similar checks with errors instead.

  o I do not yet find the timebin information in the digit?

  o For now the values of
  mRowsMax, mPadsMax, mTimeBinsMax
  are set to large numbers in the constructor to be safe. They need som ing of
  parameter lookup eventually.
  (Same for mMinQMax)

 */


#include "TPCSimulation/BoxClusterer.h"
#include "TPCSimulation/DigitMC.h"
#include "TPCSimulation/ClusterContainer.h"
#include "TPCSimulation/BoxCluster.h"

#include "FairLogger.h"
#include "TMath.h"
#include "TError.h"   // for R__ASSERT()
#include "TClonesArray.h"

ClassImp(o2::TPC::BoxClusterer)

using namespace o2::TPC;

//________________________________________________________________________
BoxClusterer::BoxClusterer():
  Clusterer(),
  mAllBins(nullptr),
  mAllSigBins(nullptr),
  mAllNSigBins(nullptr)
{
}

//________________________________________________________________________
BoxClusterer::~BoxClusterer()
{
  delete mClusterContainer;

  for (Int_t iRow = 0; iRow < mRowsMax; iRow++) {
    delete [] mAllBins[iRow];
    delete [] mAllSigBins[iRow];
  }
  delete [] mAllBins;
  delete [] mAllSigBins;
  delete [] mAllNSigBins;
}

//________________________________________________________________________
void BoxClusterer::Init()
{
  // Test that init was not called before
  R__ASSERT(!mClusterContainer);

  mClusterContainer = new ClusterContainer();
  mClusterContainer->InitArray("o2::TPC::BoxCluster");

  mAllBins = new Float_t*[mRowsMax];
  mAllSigBins = new Int_t*[mRowsMax];
  mAllNSigBins = new Int_t[mRowsMax];

  for (Int_t iRow = 0; iRow < mRowsMax; iRow++) {
    //
    Int_t maxBin = (mTimeBinsMax+4)*(mPadsMax+4);
    mAllBins[iRow] = new Float_t[maxBin];
    for(Int_t i = 0; i < maxBin; i++)
      mAllBins[iRow][i] = 0;
    mAllSigBins[iRow] = new Int_t[maxBin];
    mAllNSigBins[iRow] = 0;
  }
}

//________________________________________________________________________
ClusterContainer* BoxClusterer::Process(TClonesArray *digits)
{
  R__ASSERT(mClusterContainer);
  mClusterContainer->Reset();

  Int_t nSignals = 0;
  Int_t lastCRU = -1;
  Int_t iCRU    = -1;

  for (TIter digititer = TIter(digits).Begin(); digititer != TIter::End(); ++digititer)
    {
      DigitMC* digit = dynamic_cast<DigitMC*>(*digititer);

                  iCRU     = digit->getCRU();
      const Int_t iRow     = digit->getRow();
      const Int_t iPad     = digit->getPad();
      const Int_t iTimeBin = digit->getTimeStamp();
      const Float_t charge = digit->getCharge();
//      if (iCRU == 179) {
//        printf("box: digi: %d, %d, %d, %d, %.2f\n", iCRU, iRow, iPad, iTimeBin, charge);
//      }
      if(iCRU != lastCRU) {
        if(nSignals>0) {
          FindLocalMaxima(lastCRU);
          CleanArrays();
        }
        lastCRU = iCRU;
        nSignals = 0;
      } //else { // add signal to array
      Update(iCRU, iRow, iPad, iTimeBin, charge);
      ++nSignals;
      //}
    }

    // processing of last CRU
    if(nSignals>0) {
      FindLocalMaxima(iCRU);
      CleanArrays();
    }

  return mClusterContainer;
}

//_____________________________________________________________________
void BoxClusterer::FindLocalMaxima(const Int_t iCRU)
{
  /// This method is called after the data from each CRU has been
  /// exapanded into an array
  /// Loop over the signals and identify local maxima and fill the
  /// calibration objects with the information

  R__ASSERT(mAllBins);

  Int_t nLocalMaxima = 0;
  // loop over rows
  for (Int_t iRow = 0; iRow < mRowsMax; iRow++) {

    Float_t* allBins = mAllBins[iRow];
    Int_t* sigBins   = mAllSigBins[iRow];
    const Int_t nSigBins   = mAllNSigBins[iRow];

    // loop over all signals
    for (Int_t iSig = 0; iSig < nSigBins; iSig++) {

      Int_t bin  = sigBins[iSig];
      // Array of charged centered at the current signal
      Float_t *qArray = &allBins[bin];
      Float_t qMax = qArray[0];

      // First check that the charge is bigger than the threshold
      if ( qMax < mMinQMax )
	continue;

      // Require at least one neighboring time bin with signal
      if ( qArray[-1] + qArray[1] <= 0 ) continue;
      // Require at least one neighboring pad with signal
      const Int_t maxTimeBin = mTimeBinsMax+4; // Used to step between neighboring
      if ( mRequireNeighbouringPad
	   && (qArray[-maxTimeBin]+qArray[maxTimeBin]<=0) ) continue;
      //
      // Check that this is a local maximum
      // Note that the checking is done so that if 2 charges has the same
      // qMax then only 1 cluster is generated
      // (that is why there is BOTH > and >=)
      //
      if (qArray[-maxTimeBin]   >= qMax) continue;
      if (qArray[+maxTimeBin]   > qMax)  continue;
      if (qArray[-1  ]          >= qMax) continue;
      if (qArray[+1  ]          > qMax)  continue;
      if (qArray[-maxTimeBin-1] >= qMax) continue;
      if (qArray[+maxTimeBin+1] > qMax)  continue;
      if (qArray[+maxTimeBin-1] >= qMax) continue;
      if (qArray[-maxTimeBin+1] > qMax) continue;

//      Short_t tb; Short_t pa;
//      GetPadAndTimeBin(bin,pa,tb);
//      if ((iCRU == 179 && iRow == 2 && pa == 104 && tb == 171) || (iCRU == 256 && iRow == 10 && pa == 27 && tb == 181) ) {
//        std::cout << qArray[+maxTimeBin-1]  << " " << qArray[+maxTimeBin] << " "  << qArray[+maxTimeBin+1] << std::endl;
//        std::cout << qArray[-1]             << " " << qArray[0]           << " "  << qArray[+1] << std::endl;
//        std::cout << qArray[-maxTimeBin-1]  << " " << qArray[-maxTimeBin] << " "  << qArray[-maxTimeBin+1] << std::endl;
//      }
      // We accept the local maximum as a cluster and calculates its
      // parameters
      ++nLocalMaxima;

      //
      // Calculate the total charge as the sum over the region:
      //
      //    o o o o o
      //    o i i i o
      //    o i C i o
      //    o i i i o
      //    o o o o o
      //
      // with qmax at the center C.
      //
      // The inner charge (i) we always add, but we only add the outer
      // charge (o) if the neighboring inner bin (i) has a signal.
      //
      Short_t minP = 0, maxP = 0, minT = 0, maxT = 0;
      Double_t meanP = 0;  // mean pad position
      Double_t meanT = 0;  // mean time position
      Double_t sigmaP = 0; // sigma pad position
      Double_t sigmaT = 0; // sigma time position
      Float_t qTot = qMax; // total charge
	for(Short_t dTime = -1; dTime<=1; dTime++) { // delta time
      for(Short_t dPad = -1; dPad<=1; dPad++) {      // delta pad

	  if( dPad==0 && dTime==0 ) // central pad
	    continue;

	  Float_t charge = GetQ(qArray, dPad, dTime, minT, maxT, minP, maxP);
	  UpdateCluster(charge, dPad, dTime, qTot,
			meanP, sigmaP, meanT, sigmaT);

	  if( !mRequirePositiveCharge || charge>0 ) {
	    // see if the next neighbor is also above threshold

	    if(dPad*dTime==0) { // we are above/below or to the sides
	                        // (dPad, dTime) = (+-1, 0), or (0, +-1)
	                        // so we only have 1 neighbor
	      charge = GetQ(qArray, 2*dPad, 2*dTime, minT, maxT, minP, maxP);
	      UpdateCluster(charge, 2*dPad, 2*dTime, qTot,
			    meanP, sigmaP, meanT, sigmaT);
	    } else { // we are in a diagonal corner so we have 3 neighbors
	             // (dPad, dTime) = (+-1, +-1), or (+-1, -+1)
	      charge = GetQ(qArray,   dPad, 2*dTime, minT, maxT, minP, maxP);
	      UpdateCluster(charge,   dPad, 2*dTime, qTot,
			    meanP, sigmaP, meanT, sigmaT);
	      charge = GetQ(qArray, 2*dPad,   dTime, minT, maxT, minP, maxP);
	      UpdateCluster(charge, 2*dPad,   dTime, qTot,
			    meanP, sigmaP, meanT, sigmaT);
	      charge = GetQ(qArray, 2*dPad, 2*dTime, minT, maxT, minP, maxP);
	      UpdateCluster(charge, 2*dPad, 2*dTime, qTot,
			    meanP, sigmaP, meanT, sigmaT);
	    }
	  }
	}
      }

      // calculate cluster parameters
      if(qTot > 0) {
	meanP  /= qTot;
	meanT  /= qTot;
	sigmaP /= qTot;
	sigmaT /= qTot;
	sigmaP = TMath::Sqrt(sigmaP - meanP*meanP);
	sigmaT = TMath::Sqrt(sigmaT - meanT*meanT);
	Short_t pad, timebin;
	GetPadAndTimeBin(bin, pad, timebin);
	meanP += pad;
	meanT += timebin;
	Short_t nPad = maxP-minP+1;
	Short_t nTimeBins = maxT-minT+1;
	Short_t size = 10*nPad+nTimeBins;
	BoxCluster* cluster = dynamic_cast<BoxCluster*>
	  (mClusterContainer->AddCluster(iCRU, iRow, qTot, qMax, meanP, meanT,
					 sigmaP, sigmaT));
	cluster->setBoxParameters(pad, timebin, size);

//    if ((iCRU == 179)) {// && iRow == 5)){// && (int)meanP == 103 && (int)meanT == 170) || 
////        (iCRU == 256 && iRow == 10 && (int)meanP == 27 && (int)meanT == 181) ) {
//    std::cout << "BoxCluster - ";
//    cluster->Print(std::cout);
//    std::cout << " " << std::endl;
//	for(Short_t dTime = -2; dTime<=2; dTime++) { // delta time
//      for(Short_t dPad = -2; dPad<=2; dPad++) {      // delta pad
//        Float_t charge = GetQ(qArray, dPad, dTime, minT, maxT, minP, maxP);  
//        std::cout << "\t" << charge;
//      }
//      std::cout << std::endl;
//	}
//      std::cout << std::endl;
//    }
////    LOG(INFO) << *cluster << FairLogger::endl;
      }
    } // end loop over signals
  } // end loop over rows
}

//_____________________________________________________________________
void BoxClusterer::CleanArrays()
{
  // here it might be faster to do a memset for very large datasets

  R__ASSERT(mAllBins);

  for (Int_t iRow = 0; iRow < mRowsMax; iRow++) {

    Float_t* allBins = mAllBins[iRow];
    Int_t*   sigBins   = mAllSigBins[iRow];
    const Int_t nSignals = mAllNSigBins[iRow];
    for(Int_t i = 0; i < nSignals; i++)
      allBins[sigBins[i]]=0;

    mAllNSigBins[iRow]=0;
  }
}

//_____________________________________________________________________
void BoxClusterer::GetPadAndTimeBin(Int_t bin, Short_t& iPad, Short_t& iTimeBin)
{
  /// Return pad and timebin for a given bin
  //  (where bin = (iPad+2)*(mTimeBinsMax+4) + (iTimeBin+2)
  iTimeBin  = Short_t(bin%(mTimeBinsMax+4));
  iPad      = Short_t((bin-iTimeBin)/(mTimeBinsMax+4));
  iTimeBin -= 2;
  iPad     -= 2;

  R__ASSERT(iPad>=0     && iPad<(mPadsMax+4));
  R__ASSERT(iTimeBin>=0 && iTimeBin<(mTimeBinsMax+4));
}

//_____________________________________________________________________
Int_t BoxClusterer::Update(const Int_t iCRU,
			   const Int_t iRow,
			   const Int_t iPad,
			   const Int_t iTimeBin,
			   Float_t signal)
{
  /// Signal filling method

  // -ne could have a list of active chambers
  // if (!fActiveChambers[iSector]) return 0;

  // Stop processing if input is out of range
  R__ASSERT(iRow>=0     && iRow<mRowsMax);
  R__ASSERT(iPad>=0     && iPad<(mPadsMax+4));
  R__ASSERT(iTimeBin>=0 && iTimeBin<(mTimeBinsMax+4));

  if (mRequirePositiveCharge && (signal <= 0)){
    return 0; // signal was not accepted
  }

  // Fill signal in array. Add 2 to pad and time to make sure that the 2D
  // array even for (0, 0) has a valid 5x5 matrix.
  Int_t bin = (iPad+2)*(mTimeBinsMax+4) + (iTimeBin+2);

  mAllBins[iRow][bin] = signal;
  mAllSigBins[iRow][mAllNSigBins[iRow]] = bin;
  mAllNSigBins[iRow]++;

  return 1; // signal was accepted
}



//______________________________________________________________________________
Float_t BoxClusterer::GetQ(const Float_t* adcArray,
			   const Short_t pad, const Short_t time, 
			   Short_t& timeMin, Short_t& timeMax,
			   Short_t& padMin,  Short_t& padMax) const
{
  /// This methods return the charge in the bin time+pad*maxTimeBins
  /// If the charge is above 0 it also updates the padMin, padMax, timeMin
  /// and timeMax if necessary

  const Int_t maxTimeBin = mTimeBinsMax+4; // Used to step between neighboring

  Float_t charge = adcArray[time + pad*maxTimeBin];
  if(charge > 0) {
    timeMin = TMath::Min(time, timeMin); timeMax = TMath::Max(time, timeMax);
    padMin = TMath::Min(pad, padMin); padMax = TMath::Max(pad, padMax);
  }
  return charge;
}

//________________________________________________________________________
Bool_t BoxClusterer::UpdateCluster(Float_t charge, Int_t deltaPad, Int_t deltaTime, Float_t& qTotal, Double_t& meanPad, Double_t& sigmaPad, Double_t& meanTime, Double_t& sigmaTime)
{
  if(mRequirePositiveCharge && charge <=0)
    return kFALSE;

  qTotal += charge;
  meanPad  += charge * deltaPad;   sigmaPad += charge * deltaPad*deltaPad;
  meanTime += charge * deltaTime;  sigmaTime += charge* deltaTime*deltaTime;
  return kTRUE;
}
