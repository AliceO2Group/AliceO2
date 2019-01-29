// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/////////////////////////////////////////////////////////
//                                                     //
// Container Class for Dictionary Info                 //
//                                                     //
// Author:                                             //
//   Hermes Leon Vargas (hleon@ikf.uni-frankfurt.de)   //
// Ported to O2                                        //
/////////////////////////////////////////////////////////



#include "TRDBase/TRDArrayDictionary.h"
#include "TRDBase/TRDFeeParam.h"
#include <fairlogger/Logger.h>

using namespace o2::trd;


short *TRDArrayDictionary::mgLutPadNumbering = nullptr;

//________________________________________________________________________________
TRDArrayDictionary::TRDArrayDictionary()
{
  // TRDArrayDictionary default contructor

  CreateLut();

}

//________________________________________________________________________________
TRDArrayDictionary::TRDArrayDictionary(int nrow, int ncol, int ntime)
{
  // TRDArrayDictionary contructor

  CreateLut();
  Allocate(nrow,ncol,ntime);

}

//________________________________________________________________________________
TRDArrayDictionary::TRDArrayDictionary(const TRDArrayDictionary &a)
		      :mNdet(a.mNdet) ,mNrow(a.mNrow) ,mNcol(a.mNcol) ,mNumberOfChannels(a.mNumberOfChannels) ,mNtime(a.mNtime) ,mNDdim(a.mNDdim) ,mFlag(a.mFlag)
{
  //
  // TRDArrayDictionary copy constructor
  //

  mDictionary=a.mDictionary;
}

//________________________________________________________________________________
TRDArrayDictionary::~TRDArrayDictionary(): default;


//________________________________________________________________________________
inline int TRDArrayDictionary::getData(int nrow, int ncol, int ntime) const
{
  //
  // get the data using the pad numbering.
  // To access data using the mcm scheme use instead
  // the method getDataByAdcCol
  //

  int corrcolumn = mgLutPadNumbering[ncol];

  return mDictionary[(nrow*mNumberOfChannels+corrcolumn)*mNtime+ntime];

}
//________________________________________________________________________________
inline void TRDArrayDictionary::setData(int nrow, int ncol, int ntime, int value)
{
  //
  // Set the data using the pad numbering.
  // To write data using the mcm scheme use instead
  // the method setDataByAdcCol
  //

  int colnumb = mgLutPadNumbering[ncol];

  mDictionary[(nrow*mNumberOfChannels+colnumb)*mNtime+ntime]=value;

}



//________________________________________________________________________________
TRDArrayDictionary &TRDArrayDictionary::operator=(const TRDArrayDictionary &a)
{
  //
  // Assignment operator
  //

  if(this==&a)
    {
      return *this;
    }

  mNdet=a.mNdet;
  mNDdim=a.mNDdim;
  mNrow=a.mNrow;
  mNcol=a.mNcol;
  mNumberOfChannels = a.mNumberOfChannels;
  mNtime=a.mNtime;
  mFlag=a.mFlag;

  mDictionary=a.mDictionary;
  return *this;

}

//________________________________________________________________________________
void TRDArrayDictionary::Allocate(int nrow, int ncol, int ntime)
{
  //
  // Allocates memory for the dictionary array with dimensions
  // Row*NumberOfNecessaryMCMs*ADCchannelsInMCM*Time
  // To be consistent with TRDArrayADC
  // Object initialized to -1
  //

  mNrow=nrow;
  mNcol=ncol;
  mNtime=ntime;
  int adcchannelspermcm = TRDFeeParam::GetNadcMcm(); 
  int padspermcm = TRDFeeParam::GetNcolMcm(); 
  int numberofmcms = mNcol/padspermcm;
  mNumberOfChannels = numberofmcms*adcchannelspermcm;
  mNDdim=nrow*mNumberOfChannels*ntime;
  if(mDictionary.size() != mNDdim){
      mDictionary.resize(mNDdim);
  }
  memset(&mDictionary[0],-1,sizeof(mDictionary[0])*mNDdim);

}

//________________________________________________________________________________
void TRDArrayDictionary::Compress()
{
  //
  // Compress the array
  //


  int counter=0;
  int newDim=0;
  int j;                 
  int r=0;
  int k=0;

  std::vector<int> longArr(mNDdim); //do not change to bool 

      memset(&longArr[0],0,sizeof(longArr[0])*mNDdim);

      for(int i=0;i<mNDdim; i++) {
          j=0;
          if(mDictionary[i]==-1) {
	      for(k=i;k<mNDdim;k++) {
	          if(mDictionary[k]==-1) {
		      j=j+1;
		      longArr[r]=j;
		    }
	          else {
		      break;
		    }
	        } 
	      r=r+1;    
	    }
          i=i+j;
        }

      //Calculate the size of the compressed array
      for(int i=0; i<mNDdim;i++) {
          if(longArr[i]!=0)  {
	      counter=counter+longArr[i]-1;
	    }
        }
      newDim=mNDdim-counter;   //Size of the compressed array

      //Fill the buffer of the compressed array
      std::vector<int>  buffer(newDim);
      int counterTwo=0;
      int g=0;
          for(int i=0; i<newDim; i++) {
              if(counterTwo<mNDdim) {
	          if(mDictionary[counterTwo]!=-1) {
	              buffer[i]=mDictionary[counterTwo];
	            }
	          if(mDictionary[counterTwo]==-1) {
	              buffer[i]=-(longArr[g]);
	              counterTwo=counterTwo+longArr[g]-1;
	              g++;
	            }  
	          counterTwo++;
	        }
         }

      mDictionary = buffer;
      mNDdim = newDim;
    
  mFlag=kFALSE; // This way it can be expanded afterwards
}

//________________________________________________________________________________
void TRDArrayDictionary::Expand()
{
  //  
  //  Expand the array
  //  

  if(mNDdim==0)
    {
      LOG(error) << "Called expand with dinesion of zero ";
      return;    
    }


  int dimexp=0;
  
//   if(WasExpandCalled()) 
//     return;

  if(mNDdim==mNrow*mNumberOfChannels*mNtime)
    return;

  if(mDictionary&&mNDdim==1)
    { 
      dimexp = -mDictionary[0];	
      delete [] mDictionary;
      mDictionary=0;
      mDictionary = new int[dimexp];
      mNDdim = dimexp;
      // Re-initialize the array
      memset(mDictionary,-1,sizeof(int)*dimexp);
      mFlag=kTRUE; // Not expand again
      return;
    }

  std::vector<int> longArr(mNDdim);

      //Initialize the array
      memset(&longArr[0],0,sizeof(longArr[0])*mNDdim);

      int r2=0;
      for(int i=0; i<mNDdim;i++) {
          if((mDictionary[i]<0)&&(mDictionary[i]!=-1))  {
	      longArr[r2]=-mDictionary[i]; 
	      r2++;
	    }
        }

      //Calculate new dimensions
      for(int i=0; i<mNDdim;i++) {
          if(longArr[i]!=0){      
	        dimexp=dimexp+longArr[i]-1;
          }
	    if(longArr[i]==0){

	        break;
        }
      }
      dimexp=dimexp+mNDdim;  

      //Write in the buffer the new array
      int contaexp =0;    
      int h=0;
      std::vector<int> bufferE(dimexp);

	  memset(&bufferE[0],-1,sizeof(bufferE)*dimexp);

          for(int i=0; i<dimexp; i++) {
              if(mDictionary[contaexp]>=-1)  {
	          bufferE[i]=mDictionary[contaexp];
	        }
              if(mDictionary[contaexp]<-1)  {
	          i=i+longArr[h]-1;
	          h++;
	        }
              contaexp++;
            }

          //Copy the buffer
          mDictionary=bufferE;
          mNDdim = dimexp;
	}
    }
  if (longArr)
    {
      delete [] longArr; 
    }
  mFlag=kTRUE; // Not expand again

}
//________________________________________________________________________________
void TRDArrayDictionary::Reset()
{
  //
  // Reset the array, the old contents are deleted
  // and the data array elements are set to zero.
  //

  memset(&mDictionary[0],0,sizeof(mDictionary[0])*mNDdim);

}


//________________________________________________________________________________
void TRDArrayDictionary::CreateLut()
{
  //
  // Initializes the Look Up Table to relate
  // pad numbering and mcm channel numbering
  //

  if(!mgLutPadNumberingExists){
  
   if(mgLutPadNumbering.size()!=TRDFeeParam::GetNcol())
       mgLutPadNumbering.resize()
   memset(&mgLutPadNumbering[0],0,sizeof(mgLutPadNumbering[0])*TRDFeeParam::GetNcol());

  for(int mcm=0; mcm<8; mcm++)
    {
      int lowerlimit=0+mcm*18;
      int upperlimit=18+mcm*18;
      int shiftposition = 1+3*mcm;
      for(int index=lowerlimit;index<upperlimit;index++)
	{
	  mgLutPadNumbering[index]= index+shiftposition;
	}
    }
  mgLutPadNumberingExists=kTrue;
  }
}
