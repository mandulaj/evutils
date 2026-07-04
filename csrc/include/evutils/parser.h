#ifndef EVUTILS_PARSER_H
#define EVUTILS_PARSER_H

#include "evutils/types.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    EVUTILS_PARSE_OK = 0,
    EVUTILS_PARSE_INPUT_EMPTY = 1,
    EVUTILS_PARSE_OUTPUT_FULL = 2,
    EVUTILS_PARSE_ERROR = 3,
    EVUTILS_PARSE_INCOMPLETE = 4
} parse_status_t;


typedef struct parser_result_s {
    const void *current;
    parse_status_t status;
} parser_result_t;



#define likely(x)       __builtin_expect(!!(x),1)
#define unlikely(x)     __builtin_expect(!!(x),0)


#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_PARSER_H */