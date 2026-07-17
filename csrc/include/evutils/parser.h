#ifndef EVUTILS_PARSER_H
#define EVUTILS_PARSER_H

#include "evutils/compat.h"
#include "evutils/types.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    EVUTILS_PARSE_OK = 0,
    EVUTILS_PARSE_INPUT_EMPTY = 1,
    EVUTILS_PARSE_OUTPUT_FULL = 2,
    EVUTILS_PARSE_ERROR = 3,
    EVUTILS_PARSE_INCOMPLETE = 4,
    /* A delta_t parser reached the requested time window (an event's timestamp
     * hit end_ts): the window is complete and the parser stopped at that exact
     * boundary. */
    EVUTILS_PARSE_WINDOW_DONE = 5,
    /* Parser encountered an invalid packet, dropped it, and stopped.
     * The caller may log a warning and resume parsing from the returned pointer. */
    EVUTILS_PARSE_WARNING = 6
} parse_status_t;


typedef struct parser_result_s {
    const void *current;
    parse_status_t status;
} parser_result_t;



/* likely()/unlikely() are provided by compat.h (GNU builtins on GCC/Clang,
 * no-ops on MSVC). */


#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_PARSER_H */