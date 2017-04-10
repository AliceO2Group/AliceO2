//  Path string identifying the object:  			   //
//  (example: "ZDC/Calib/Pedestals") 		         	   //
#include "CCDB/IdPath.h"
#include <FairLogger.h>  // for LOG
#include <TObjArray.h>   // for TObjArray
#include <TObjString.h>  // for TObjString
#include <TRegexp.h>     // for TRegexp

using namespace o2::CDB;

ClassImp(IdPath)

IdPath::IdPath() : TObject(), mPath(""), mLevel0(""), mLevel1(""), mLevel2(""), mValid(kTRUE), mWildcard(kFALSE)
{
  // default constructor
}

IdPath::IdPath(const IdPath &other)
  : TObject(other),
    mPath(other.mPath),
    mLevel0(""),
    mLevel1(""),
    mLevel2(""),
    mValid(other.mValid),
    mWildcard(other.mWildcard)
{
  // constructor
  init();
  InitPath();
}

IdPath::IdPath(const char *level0, const char *level1, const char *level2)
  : TObject(), mPath(""), mLevel0(level0), mLevel1(level1), mLevel2(level2), mValid(kTRUE), mWildcard(kFALSE)
{
  // constructor

  mPath += level0;
  mPath += '/';
  mPath += level1;
  mPath += '/';
  mPath += level2;

  if ((isWord(mLevel0) || mLevel0 == "*") && (isWord(mLevel1) || mLevel1 == "*") &&
      (isWord(mLevel2) || mLevel2 == "*")) {

    mValid = kTRUE;
  } else {
    mValid = kFALSE;
    LOG(ERROR) << R"(Invalid  Path ")" << level0 << "/" << level1 << "/" << level2 << R"("!)" << FairLogger::endl;
  }

  init();
}

IdPath::IdPath(const char *path)
  : TObject(), mPath(path), mLevel0(""), mLevel1(""), mLevel2(""), mValid(kTRUE), mWildcard(kFALSE)
{
  // constructor

  init();
  InitPath();
}

IdPath::IdPath(const TString &path)
  : TObject(), mPath(path), mLevel0(""), mLevel1(""), mLevel2(""), mValid(kTRUE), mWildcard(kFALSE)
{
  init();
  InitPath();
}

void IdPath::InitPath()
{
  // sets mLevel0, mLevel1, mLevel2, validity flagss from mPath

  TSubString strippedString = mPath.Strip(TString::kBoth);
  TString aString(strippedString);
  strippedString = aString.Strip(TString::kBoth, '/');

  TObjArray *anArray = TString(strippedString).Tokenize("/");
  Int_t paramCount = anArray->GetEntriesFast();

  if (paramCount == 1) {
    if (mPath == "*") {
      mLevel0 = "*";
      mLevel1 = "*";
      mLevel2 = "*";

      mValid = kTRUE;
    } else {
      mValid = kFALSE;
    }

  } else if (paramCount == 2) {
    mLevel0 = ((TObjString *) anArray->At(0))->GetString();
    TString bString = ((TObjString *) anArray->At(1))->GetString();

    if (isWord(mLevel0) && bString == "*") {
      mLevel1 = "*";
      mLevel2 = "*";

      mValid = kTRUE;

    } else {
      mValid = kFALSE;
    }

  } else if (paramCount == 3) {
    mLevel0 = ((TObjString *) anArray->At(0))->GetString();
    mLevel1 = ((TObjString *) anArray->At(1))->GetString();
    mLevel2 = ((TObjString *) anArray->At(2))->GetString();

    if ((isWord(mLevel0) || mLevel0 == "*") && (isWord(mLevel1) || mLevel1 == "*") &&
        (isWord(mLevel2) || mLevel2 == "*")) {

      mValid = kTRUE;
    } else {
      mValid = kFALSE;
    }

  } else {
    mValid = kFALSE;
  }

  if (!mValid) {
    LOG(INFO) << R"(Invalid  Path ")" << mPath.Data() << R"("!)" << FairLogger::endl;
  } else {
    mPath = Form("%s/%s/%s", mLevel0.Data(), mLevel1.Data(), mLevel2.Data());
  }

  delete anArray;

  init();
}

IdPath::~IdPath()
{
  // destructor
}

Bool_t IdPath::isWord(const TString &str)
{
  // check if string is a word

  TRegexp pattern("^[a-zA-Z0-9_.-]+$");

  return str.Contains(pattern);
}

void IdPath::init()
{
  // set mWildcard flag

  mWildcard = mPath.MaybeWildcard();
}

Bool_t IdPath::doesLevel0Contain(const TString &str) const
{
  // check if Level0 is wildcard or is equal to str

  if (mLevel0 == "*") {
    return kTRUE;
  }

  return mLevel0 == str;
}

Bool_t IdPath::doesLevel1Contain(const TString &str) const
{
  // check if Level1 is wildcard or is equal to str

  if (mLevel1 == "*") {
    return kTRUE;
  }

  return mLevel1 == str;
}

Bool_t IdPath::doesLevel2Contain(const TString &str) const
{
  // check if Level2 is wildcard or is equal to str

  if (mLevel2 == "*") {
    return kTRUE;
  }

  return mLevel2 == str;
}

Bool_t IdPath::isSupersetOf(const IdPath &other) const
{
  // check if path is wildcard and comprises other

  return doesLevel0Contain(other.mLevel0) && doesLevel1Contain(other.mLevel1) && doesLevel2Contain(other.mLevel2);
}

const char *IdPath::getLevel(Int_t i) const
{
  // return level i of the path

  switch (i) {
    case 0:
      return mLevel0.Data();
      break;
    case 1:
      return mLevel1.Data();
      break;
    case 2:
      return mLevel2.Data();
      break;
    default:
      return nullptr;
  }
}
