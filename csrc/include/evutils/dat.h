/* evutils — Prophesee DAT (.dat) CD-event parser.
 *
 * DAT layout: an ASCII "% ..." header, then two bytes (event type = 0x0C for
 * CD, event size = 0x08), then 8-byte little-endian events:
 *   bytes 0-3: timestamp (uint32, microseconds)
 *   bytes 4-7: data word -> x[0:13] (14b), y[14:27] (14b), p[28] (1b)
 * The 32-bit timestamp wraps at 2^32 us; the parser tracks the overflow across
 * chunks.
 */
#ifndef EVUTILS_DAT_H
#define EVUTILS_DAT_H

#include "evutils/types.h"
#include "evutils/parser.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dat_state_s dat_state_t;

/* Input is the binary payload viewed as uint32 words (2 words per event). */
typedef struct dat_input_buffer_s {
    const uint32_t *begin;
    const uint32_t *end;
} dat_input_buffer_t;

size_t DAT_state_size(void);

parser_result_t DAT_parse_chunk_soa(
    dat_state_t              *state,
    const dat_input_buffer_t *input_buffer,
    event_buffer_soa_t       *event_buffer,
    trigger_buffer_soa_t     *trigger_buffer);

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_DAT_H */
