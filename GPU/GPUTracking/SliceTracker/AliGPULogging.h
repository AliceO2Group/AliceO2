#ifndef ALIGPULOGGING_H
#define ALIGPULOGGING_H

#define GPUError(...) {printf(__VA_ARGS__);printf("\n");}
#define GPUWarning(...) {printf(__VA_ARGS__);printf("\n");}
#define GPUInfo(...) {printf(__VA_ARGS__);printf("\n");}
#define GPUImportant(...) {printf(__VA_ARGS__);printf("\n");}
#define GPUFatal(...) {printf(__VA_ARGS__);printf("\n");exit(1);}

#endif //ALIGPULOGGING_H
