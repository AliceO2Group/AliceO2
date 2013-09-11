#ifndef ALIHLTLOGGING_H
#define ALIHLTLOGGING_H

class AliHLTLogging
{
public:
	virtual ~AliHLTLogging();
};

#define HLTError(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTWarning(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTInfo(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTImportant(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTDebug(...) {printf(__VA_ARGS__);printf("\n");}
#define HLTFatal(...) {printf(__VA_ARGS__);printf("\n");exit(1);}

#endif //ALIHLTLOGGING_H
