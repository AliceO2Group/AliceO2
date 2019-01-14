#ifndef ALICAGPULOGGING_H
#define ALICAGPULOGGING_H

#define CAGPUError(...) {printf(__VA_ARGS__);printf("\n");}
#define CAGPUWarning(...) {printf(__VA_ARGS__);printf("\n");}
#define CAGPUInfo(...) {printf(__VA_ARGS__);printf("\n");}
#define CAGPUImportant(...) {printf(__VA_ARGS__);printf("\n");}
#define CAGPUDebug(...) {} //{printf(__VA_ARGS__);printf("\n");}
#define CAGPUFatal(...) {printf(__VA_ARGS__);printf("\n");exit(1);}

#endif //ALICAGPULOGGING_H
