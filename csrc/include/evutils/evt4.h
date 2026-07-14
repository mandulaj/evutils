/* evutils — EVT4 parser public interface.
 *
 * EVT4 is Prophesee's 32-bit-word event format. Each raw word carries a 4-bit
 * type in bits 28..31. The change-detection (CD), TIME_HIGH and external-trigger
 * words share EVT2's bit layout; EVT4 adds vectorized CD (a base word followed
 * by a 32-bit x-offset validity mask). See evt4.c for the decode.
 */
#ifndef EVUTILS_EVT4_H
#define EVUTILS_EVT4_H

#include "evutils/types.h"
#include "evutils/parser.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque parser state. Definition lives in evt4.c. */
typedef struct evt4_state_s evt4_state_t;

typedef struct evt4_input_buffer_s {
    const uint32_t *begin;
    const uint32_t *end;
} evt4_input_buffer_t;

size_t EVT4_state_size(void);

parser_result_t EVT4_parse_chunk_soa(
    evt4_state_t            *state,
    const evt4_input_buffer_t *input_buffer,
    event_buffer_soa_t      *event_buffer,
    trigger_buffer_soa_t    *trigger_buffer);

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_EVT4_H */
