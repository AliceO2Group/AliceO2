// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>
#include <TTree.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <string>
#include "TTree.h"

#include "Framework/ControlService.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ITSQCDataReaderWorkflow/TestDataReader.h"
#include "DetectorsBase/GeometryManager.h"
#include <TCanvas.h>
#include <iostream>
#include <dirent.h>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include <thread>

using namespace o2::framework;
using namespace o2::itsmft;
using namespace std;

namespace o2
{
namespace its
{

const std::string TestDataReader::sRunTypeFileName = "Config/RunType.dat";
const std::string TestDataReader::sFakeRateDefConfig = "Config/ConfigFakeRate.dat";
const std::string TestDataReader::sThresholdDefConfig = "Config/ConfigThreshold.dat";

void TestDataReader::init(InitContext& ic)
{
  mIndexPush = 0;

  std::ifstream RunFileType(sRunTypeFileName);
  RunFileType >> mRunType;

  LOG(DEBUG) << "OLD CONFIG: RunFileType = " << mRunType;

  if (mRunType == "FakeHitRate") {
    std::ifstream EventPush(sFakeRateDefConfig);
    EventPush >> mEventPerPush >> mTrackError >> mWorkDir;
  }

  if (mRunType == "ThresholdScan") {
    std::ifstream EventPush(sThresholdDefConfig);
    EventPush >> mEventPerPush >> mTrackError >> mWorkDir;
  }

  LOG(DEBUG) << "OLD CONFIG: EventPerPush = " << mEventPerPush << "   TrackError = " << mTrackError << "  WorkDir = " << mWorkDir;
  LOG(DEBUG) << "DONE Reset Histogram Decision";

  o2::base::GeometryManager::loadGeometry();

  mFolderNames = GetFName(mWorkDir);

  cout << "NFolder = " << mFolderNames.size() << endl;
  for (int i = 0; i < mFolderNames.size(); i++) {

    cout << "FDN = " << mFolderNames[i] << endl;

    mFileNames.push_back(GetFName(mFolderNames[i]));

    cout << "FDN File Size = " << mFileNames[i].size() << endl;

    for (int j = 0; j < mFileNames[i].size(); j++) {

      cout << "FDN File = " << mFileNames[i][j] << endl;
    }
  }
  for (int i = 0; i < sNError; i++) {
    mErrors[i] = 0;
  }

  //		mEventPerPush = 3000;
  mEventRegistered = 0;
  mTotalPixelSize = 0;

  //	GetFileName("infile");

  //

  const Int_t numOfChips = o2::itsmft::ChipMappingITS::getNChips();
  LOG(DEBUG) << "numOfChips = " << numOfChips;
  setNChips(numOfChips);
  //  j = 0;
  mFileDone = 1;
  mFileRemain = 0;
  mNewFileInj = 1;
  mMaxPixelSize = 58700095;
}

void TestDataReader::run(ProcessingContext& pc)
{
  // Keep checking new folders (and files in the folders)
  // If found, process them one by one

  //Defining all local variables
  int j = 0;
  int NEventPre;
  int NEvent;
  double PercentDone = 0;
  int ErrorDetcted;

  cout << "----------------------------------------------------------" << endl
       << endl;

  cout << "New Cycle" << endl;
  cout << "Old Folder Size = " << mFolderNames.size() << endl;

  // Get folders in working directory, put all files in a vector
  mNowFolderNames = GetFName(mWorkDir);
  cout << "Now NFolder = " << mNowFolderNames.size() << endl;
  for (int i = 0; i < mNowFolderNames.size(); i++) {
    mNowFileNames.push_back(GetFName(mNowFolderNames[i]));
  }

  // Check for new folders comparing with the previous cycle
  std::set_difference(mNowFolderNames.begin(), mNowFolderNames.end(), mFolderNames.begin(), mFolderNames.end(), std::inserter(mDiffFolderName, mDiffFolderName.begin()));

  cout << "Difference Size Between New and Initial Runs = " << mDiffFolderName.size() << endl;

  // No new folder
  if (mDiffFolderName.size() == 0) {
    cout << "No New Run -- No Need to Reset" << endl;
    mResetCommand = 0;
    pc.outputs().snapshot(Output{"ITS", "TEST", 0, Lifetime::Timeframe}, mResetCommand);
  }

  // New folders found, send the reset signal and reload configuration
  if (mDiffFolderName.size() > 0) {
    cout << "New Run Started -- Reset All Histograms" << endl;
    mResetCommand = 1;
    pc.outputs().snapshot(Output{"ITS", "TEST", 0, Lifetime::Timeframe}, mResetCommand);
    for (int i = 0; i < sNError; i++) {
      mErrors[i] = 0;
    }
    mErrorsVec.clear();
    mResetCommand = 0;
    mNewFileInj = 1;
    std::ifstream RunFileType(sRunTypeFileName);
    RunFileType >> mRunType;

    LOG(DEBUG) << "NEW CONFIG: RunFileType = " << mRunType;

    if (mRunType == "FakeHitRate") {
      std::ifstream EventPush(sFakeRateDefConfig);
      EventPush >> mEventPerPush >> mTrackError >> mWorkDir;
    }

    if (mRunType == "ThresholdScan") {
      std::ifstream EventPush(sThresholdDefConfig);
      EventPush >> mEventPerPush >> mTrackError >> mWorkDir;
    }

    LOG(DEBUG) << "NEW CONFIG: EventPerPush = " << mEventPerPush << "   TrackError = " << mTrackError << "  mWorkDir = " << mWorkDir;
    LOG(DEBUG) << "DONE Reset Histogram Decision";
  }

  LOG(DEBUG) << "Start Creating New Now Vector";

  LOG(DEBUG) << "Get IN LOOP";
  // Checking for new files, extract them into new vector
  for (int i = 0; i < mFolderNames.size(); i++) {
    std::set_difference(mNowFileNames[i].begin(), mNowFileNames[i].end(), mFileNames[i].begin(), mFileNames[i].end(), std::inserter(mDiffFileNamePush, mDiffFileNamePush.begin()));
    mDiffFileNames.push_back(mDiffFileNamePush);
    cout << "Difference File Size Between New and Initial Runs " << mDiffFileNames[i].size() << endl;

    mDiffFileNamePush.clear();
  }

  LOG(DEBUG) << "DONE GRABING Existing";

  //Getting the new files from new folders that does not exist in the previous cycle
  for (int i = mFolderNames.size(); i < mNowFolderNames.size(); i++) {
    mDiffFileNames.push_back(mNowFileNames[i]);
    cout << "New File Size Between New and Initial Runs " << mDiffFileNames[i].size() << endl;
  }

  LOG(DEBUG) << "Total New Files = " << mDiffFileNames.size();

  LOG(DEBUG) << "DONE Creating Difference";

  LOG(DEBUG) << "mDiffFileNames Size = " << mDiffFileNames.size();

  LOG(DEBUG) << "Start Loop";

  //Start Decoding New Files by loop through the new file vector

  for (int i = 0; i < mNowFolderNames.size(); i++) {

    LOG(DEBUG) << "i = " << i << "    mDiffFileNames[i].size() = " << mDiffFileNames[i].size();

    //Getting the folder name ID

    int pos = mNowFolderNames[i].find_last_of("/");

    if (pos != string::npos)
      mRunID = mNowFolderNames[i].substr(pos + 1);

    LOG(DEBUG) << "FileDone = " << mFileDone << endl;

    //Reading files one by one

    if (mDiffFileNames[i].size() > 0 && mFileDone == 1) {

      mFileRemain = mDiffFileNames[i].size();
      mFileDone = 0;
      cout << "RunID = " << mRunID << endl;
      cout << "File Location = " << mDiffFileNames[i][0] << endl;

      //Getting the RunID
      size_t last_index1 = mRunID.find_last_not_of("0123456789");
      string RunIDS = mRunID.substr(last_index1 + 1);

      //Getting the FileID
      string FileIDS;
      pos = mDiffFileNames[i][0].find_last_of("/");
      if (pos != string::npos)
        FileIDS = mDiffFileNames[i][0].substr(pos + 1);

      cout << "Before FileIDS = " << FileIDS << endl;

      size_t last_index2 = FileIDS.find_last_not_of("0123456789");
      FileIDS = FileIDS.substr(last_index2 + 1);

      //Extracting the RunID and File to integer in order to make it possible to inject to the QC
      mRunNumber = std::stoi(RunIDS);
      mFileID = std::stoi(FileIDS);

      ofstream fout(Form("ErrorData/ErrorLogRun%d_File%d.dat", mRunNumber, mFileID));
      fout << " START OF ERROR REPORT For Run " << mRunNumber << "  File " << mFileID << endl;

      //Reading the first file (Because it is one by one)
      mInputName = mDiffFileNames[i][0];

      mEventRegistered = 0;
      LOG(DEBUG) << "mInputName = " << mInputName;

      //Inject fake thing digits for the QC to update immediately
      //mDigitsTest is the fake digit for updating the QC immediately on the QC GUI

      if (mNewFileInj == 1) {
        cout << "New File Injected, Now Updating the Canvas and Light" << endl;
        mDigitsTest.emplace_back(0, 0, 0, 0, 0);
        mMultiDigitsTest.push_back(mDigitsTest[0]);
        mErrorsVecTest.push_back(mErrors);
        mFileDone = 1;
        mFileInfo = mFileDone + mFileRemain * 10;
        pc.outputs().snapshot(Output{"ITS", "Run", 0, Lifetime::Timeframe}, mRunNumber);
        pc.outputs().snapshot(Output{"ITS", "File", 0, Lifetime::Timeframe}, mFileID);
        pc.outputs().snapshot(Output{"ITS", "Error", 0, Lifetime::Timeframe}, mErrorsVecTest[0]);
        pc.outputs().snapshot(Output{"ITS", "Finish", 0, Lifetime::Timeframe}, mFileInfo);
        pc.outputs().snapshot(Output{"ITS", "DIGITS", 0, Lifetime::Timeframe}, mMultiDigitsTest);
        mNewFileInj = 0;
        mErrorsVecTest.clear();
        mDigitsTest.clear();
        mMultiDigitsTest.clear();
        if (mFolderNames.size() < mNowFolderNames.size())
          mFileNames.push_back(NewNextFold);
        cout << "Done!!! You should see the Canvas Updated " << endl;
        break;
      }

      //DONE Inhection//

      o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS> mRawReader;
      mRawReader.setPadding128(true); // payload GBT words are padded to 16B
      mRawReader.setVerbosity(0);
      mRawReader.setMinTriggersToCache(1025);

      mRawReader.openInput(mInputName);

      int Index = 0;
      int IndexMax = -1;
      int NChip = 0;
      int NChipMax = -1;
      int TimePrint = 0;
      using RawReader = o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS>;
      auto& rawErrorReader = reinterpret_cast<RawReader&>(mRawReader);

      while ((mChipData = mRawReader.getNextChipData(mChips))) {
        if (NChip < NChipMax)
          break;
        //	cout << "Pass Chip" << endl;

        const auto* ruInfo = rawErrorReader.getCurrRUDecodeData()->ruInfo;
        const auto& statRU = rawErrorReader.getRUDecodingStatSW(ruInfo->idSW);

        const auto& pixels = mChipData->getData();
        int pixelSize = mChipData->getData().size();

        NEvent = statRU->nPackets;

        mTotalPixelSize = mTotalPixelSize + pixelSize;

        if (NEvent > (mEventRegistered + 1) * mEventPerPush) {

          if (mTotalPixelSize < mMaxPixelSize || mTotalPixelSize == mMaxPixelSize) {
            cout << "Digit OK for 1 Push" << endl;
            mNDigits.push_back(mTotalPixelSize);
            mEventRegistered = mEventRegistered + 1;
            mErrorsVec.push_back(mErrors);
            cout << "TotalPixelSize = " << mTotalPixelSize << "  Pushed" << endl;
          }

          if (mTotalPixelSize > mMaxPixelSize) {
            cout << "Digit Spilt into 2 Pusbhes" << endl;
            mNDigits.push_back(mTotalPixelSize / 2);
            mNDigits.push_back(mTotalPixelSize / 2);
            mErrorsVec.push_back(mErrors);
            mErrorsVec.push_back(mErrors);
            mEventRegistered = mEventRegistered + 1;
            cout << "TotalPixelSize = " << mTotalPixelSize << "  Pushed" << endl;
          }

          mTotalPixelSize = 0;
        }

        //Printing the Event Number Processed

        if (NEvent % 1000000 == 0 && TimePrint == 0) {
          cout << "Event Number = " << NEvent << endl;
          TimePrint = 1;
        }

        if (NEvent % 100000 != 0)
          TimePrint = 0;

        for (int i = 0; i < o2::itsmft::GBTLinkDecodingStat::NErrorsDefined; i++) {
          if (mErrors[i] < 4294967295)
            mErrors[i] = mErrors[i] + (int)statRU->errorCounts[i];
        }

        if (mTrackError == 1) {
          if (NEventPre != NEvent) {
            ErrorDetcted = 0;

            for (int i = 0; i < o2::itsmft::GBTLinkDecodingStat::NErrorsDefined; i++) {
              if ((int)statRU->errorCounts[i] > 0)
                ErrorDetcted = 1;
            }

            if (ErrorDetcted == 1) {
              fout << "Event Number = " << NEvent << endl;
              fout << " ------------------------------------------------" << endl;
              for (int i = 0; i < o2::itsmft::GBTLinkDecodingStat::NErrorsDefined; i++) {
                if (statRU->errorCounts[i]) {
                  fout << "Error ID " << i << " " << o2::itsmft::GBTLinkDecodingStat::ErrNames[i] << endl;
                }
              }
              fout << " ------------------------------------------------" << endl;
            }
          }
        }

        int ChipID = mChipData->getChipID();

        for (auto& pixel : pixels) {
          if (Index < IndexMax)
            break;
          int col = pixel.getCol();
          int row = pixel.getRow();
          mDigits.emplace_back(ChipID, NEvent, row, col, 0);
          Index = Index + 1;
        }
        NChip = NChip + 1;
        NEventPre = NEvent;
      }

      cout << "Final TotalPixelSize = " << mTotalPixelSize << endl;
      mNDigits.push_back(mTotalPixelSize);
      mErrorsVec.push_back(mErrors);
      LOG(DEBUG) << "Run " << mNowFolderNames[i] << " File " << mInputName << "    Integrated Raw Pixel Pushed " << mDigits.size();
      if (mFolderNames.size() < mNowFolderNames.size())
        mFileNames.push_back(NewNextFold);
      mFileNames[i].push_back(mInputName);
      fout << " END OF ERROR REPORT " << endl;
    }
  }

  LOG(DEBUG) << "DONE Pushing";

  LOG(DEBUG) << "mIndexPush Before = " << mIndexPush << "  mDigits.size() =  " << mDigits.size();

  if (mDigits.size() > 0)
    PercentDone = double(mIndexPush) / double(mDigits.size());
  cout << "Percentage Processed = " << Form("%.2f", 100. * PercentDone) << endl;

  if (mIndexPush < mDigits.size()) {
    for (int i = 0; i < mNDigits[j]; i++) {
      mMultiDigits.push_back(mDigits[mIndexPush + i]);
    }
    LOG(DEBUG) << "j = " << j << "   NDgits = " << mNDigits[j] << "    mMultiDigits Pushed = " << mMultiDigits.size();
    LOG(DEBUG) << "i = " << 10 << "  ErrorShould = " << mErrors[10] << "  ErrorInjected = " << mErrorsVec[j][10];
    ;

    cout << "RunIDS = " << mRunNumber << "   FileIDS = " << mFileID << endl;

    pc.outputs().snapshot(Output{"ITS", "Run", 0, Lifetime::Timeframe}, mRunNumber);
    pc.outputs().snapshot(Output{"ITS", "File", 0, Lifetime::Timeframe}, mFileID);

    pc.outputs().snapshot(Output{"ITS", "Error", 0, Lifetime::Timeframe}, mErrorsVec[j]);
    mIndexPushEx = mIndexPush + mNDigits[j];
    LOG(DEBUG) << "IndexPushEx = " << mIndexPushEx << "  mDigits.size() " << mDigits.size();
    if (mIndexPushEx > mDigits.size() - 5)
      mFileDone = 1;
    LOG(DEBUG) << "FileDone = " << mFileDone;
    LOG(DEBUG) << "FileRemain = " << mFileRemain;

    mFileInfo = mFileDone + mFileRemain * 10;

    pc.outputs().snapshot(Output{"ITS", "Finish", 0, Lifetime::Timeframe}, mFileInfo);

    LOG(DEBUG) << "mIndexPush = " << mIndexPush << "    Chip ID Pushing " << mDigits[mIndexPush].getChipIndex();

    pc.outputs().snapshot(Output{"ITS", "DIGITS", 0, Lifetime::Timeframe}, mMultiDigits);

    mMultiDigits.clear();
    mIndexPush = mIndexPush + mNDigits[j];
    j = j + 1;
  }

  LOG(DEBUG) << "mIndexPush After = " << mIndexPush;

  /*
			   LOG(DEBUG) << "Before:  " << "mIndexPush = " << mIndexPush << "     mDigits.size() = " <<  mDigits.size(); 
			   while(mIndexPush < mDigits.size()){
			//	LOG(DEBUG) << "mDigits.size() = " << mDigits.size();
			pc.outputs().snapshot(Output{ "ITS", "DIGITS", 0, Lifetime::Timeframe }, mDigits[mIndexPush++]);
			if(mIndexPush%100000==0) 	LOG(DEBUG) << "mIndexPush = " << mIndexPush << "    Chip ID Pushing " << mDigits[mIndexPush].getChipIndex();
			}
			//pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
			LOG(DEBUG) << "After:  " << "mIndexPush = " << mIndexPush << "     mDigits.size() = " <<  mDigits.size(); 
			*/

  mFolderNames.clear();
  //	mFileNames.clear();
  NewNextFold.clear();
  mFolderNames = mNowFolderNames;
  //	mFileNames = mNowFileNames;

  mNowFolderNames.clear();
  mNowFileNames.clear();
  mDiffFileNames.clear();
  mDiffFolderName.clear();

  LOG(DEBUG) << "Pushing Reset Histogram Decision";

  cout << "Resetting Pushing Things" << endl;

  //Resetting a New File //
  if (mIndexPush > mDigits.size() - 5) {
    mDigits.clear();
    mIndexPush = 0;
    j = 0;
    mNDigits.clear();
    mFileDone = 1;
    pc.outputs().snapshot(Output{"TST", "Finish", 0, Lifetime::Timeframe}, mFileDone);
    PercentDone = 0;
    mErrorsVec.clear();
  }

  //if(mNewFileInjAction == 1){
  //		mNewFileInjAction = 0;
  //	}

  cout << "Start Sleeping" << endl;
  cout << " " << endl;
  cout << " " << endl;
  cout << " " << endl;
  cout << " " << endl;
  cout << " " << endl;
  cout << " " << endl;
  cout << " " << endl;
  cout << " " << endl;
  cout << " " << endl;
  cout << " " << endl;

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

std::vector<string> TestDataReader::GetFName(std::string folder)
{

  DIR* dirp;

  char cstr[folder.size() + 1];
  strcpy(cstr, folder.c_str());
  dirp = opendir(cstr);
  std::vector<string> names;
  //string search_path = folder + "/*";
  if (dirp) {
    struct dirent* directory;
    while ((directory = readdir(dirp)) != nullptr) {

      //printf("%s\n", directory->d_name);

      if (!(!strcmp(directory->d_name, ".") || !strcmp(directory->d_name, "..")))
        names.push_back(folder + "/" + directory->d_name);
    }

    closedir(dirp);
  }

  cout << "names size = " << names.size() << endl;
  return (names);
}

DataProcessorSpec getTestDataReaderSpec()
{
  return DataProcessorSpec{
    "Raw-Pixel-Reader",
    Inputs{},
    Outputs{
      OutputSpec{"ITS", "DIGITS", 0, Lifetime::Timeframe},
      OutputSpec{"ITS", "TEST", 0, Lifetime::Timeframe},
      OutputSpec{"ITS", "Error", 0, Lifetime::Timeframe},
      OutputSpec{"ITS", "Run", 0, Lifetime::Timeframe},
      OutputSpec{"ITS", "File", 0, Lifetime::Timeframe},
      OutputSpec{"ITS", "Finish", 0, Lifetime::Timeframe},
      /*
						   OutputSpec{ "TST", "Error0", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error1", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error2", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error3", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error4", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error5", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error6", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error7", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error8", 0, Lifetime::Timeframe },
						   OutputSpec{ "TST", "Error9", 0, Lifetime::Timeframe },
						   */
      //		OutputSpec{ "TST", "TEST3", 0, Lifetime::Timeframe },
    },
    AlgorithmSpec{adaptFromTask<TestDataReader>()},
  };
}

} // namespace its
} // namespace o2
