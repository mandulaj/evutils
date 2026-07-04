

#ifndef EVUTILS_EVT2_H
#define EVUTILS_EVT2_H

#include "evutils/types.h"
#include "evutils/parser.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct evt2_state_s evt2_state_t;


typedef struct evt2_input_buffer_s {
    const uint32_t *begin;
    const uint32_t *end;
} evt2_input_buffer_t;



parser_result_t EVT2_parse_chunk_soa(
    evt2_state_t *state,
    const evt2_input_buffer_t *input_buffer,
    event_buffer_soa_t *event_buffer,
    trigger_buffer_soa_t *trigger_buffer);


#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_EVT2_H */
