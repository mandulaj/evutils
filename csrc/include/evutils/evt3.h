/* evutils — EVT3 parser public interface.
 *
 * This declares the EVT3 ABI exactly as your parser already exposes it, plus
 * three small lifecycle functions for the opaque parser state. Python never
 * needs to know the layout of evt3_state_t — it only holds the pointer.
 *
 * ACTION REQUIRED in evt3.c: implement the three lifecycle functions below.
 * They are thin wrappers around whatever you already do to zero-initialise
 * your state struct, e.g.:
 *
 *     struct evt3_state_s { uint64_t ts; uint16_t y, vecbase_x; uint8_t vecbase_p; ... };
 *     evt3_state_t *EVT3_state_create(void) { return calloc(1, sizeof(evt3_state_t)); }
 *     void EVT3_state_reset(evt3_state_t *s) { memset(s, 0, sizeof(*s)); }
 *     void EVT3_state_destroy(evt3_state_t *s) { free(s); }
 */
#ifndef EVUTILS_EVT3_H
#define EVUTILS_EVT3_H

#include "evutils/types.h"
#include "evutils/parser.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Returned by the chunk parsers to tell the caller why parsing stopped.*/
typedef enum evt3_parse_status_e {
    EVT3_STATUS_OK               = 0, /* reached end of input cleanly        */
    EVT3_STATUS_INPUT_EMPTY  = 1, /* ran out of input mid-group (carry)  */
    EVT3_STATUS_OUTPUT_FULL      = 2, /* event/trigger buffer hit capacity   */
    EVT3_STATUS_ERROR            = 3,  /* malformed stream                    */
    EVT3_STATUS_INCOMPLETE = 4, /* ran out of input mid-group (carry)  */
} evt3_parse_status_t;

/* Opaque parser state. Definition lives in evt3.c. */
typedef struct evt3_state_s evt3_state_t;





typedef struct evt3_input_buffer_s {
    const uint16_t *begin;
    const uint16_t *end;
} evt3_input_buffer_t;


size_t EVT3_state_size(void);

parser_result_t EVT3_parse_chunk_soa(
    evt3_state_t            *state,
    const evt3_input_buffer_t *input_buffer,
    event_buffer_soa_t      *event_buffer,
    trigger_buffer_soa_t    *trigger_buffer);

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_EVT3_H */
