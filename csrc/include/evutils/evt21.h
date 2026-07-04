

#ifndef EVUTILS_EVT21_H
#define EVUTILS_EVT21_H

#include "evutils/types.h"
#include "evutils/parser.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct evt21_state_s evt21_state_t;


typedef struct evt21_input_buffer_s {
    const uint64_t *begin;
    const uint64_t *end;
} evt21_input_buffer_t;



parser_result_t EVT21_parse_chunk_soa(
    evt21_state_t *state,
    const evt21_input_buffer_t *input_buffer,
    event_buffer_soa_t *event_buffer,
    trigger_buffer_soa_t *trigger_buffer);

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_EVT21_H */
