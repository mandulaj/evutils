/* evutils — Prophesee AER CD-event parser.
 *
 * AER is a raw 32-bit-per-event encoding with NO timestamps and NO header:
 *   bits 0-8  : y (9 bits)
 *   bits 9-17 : x (9 bits)
 *   bit  18   : polarity (0: OFF, 1: ON)
 * The 9-bit fields limit coordinates to < 512 (e.g. GenX320). Decoded events
 * carry t = 0 since the format has no time information.
 */
#ifndef EVUTILS_AER_H
#define EVUTILS_AER_H

#include "evutils/types.h"
#include "evutils/parser.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aer_input_buffer_s {
    const uint32_t *begin;
    const uint32_t *end;
} aer_input_buffer_t;

/* AER is stateless (no timestamps, no cross-chunk carry). */
parser_result_t AER_parse_chunk_soa(
    const aer_input_buffer_t *input_buffer,
    event_buffer_soa_t       *event_buffer,
    trigger_buffer_soa_t     *trigger_buffer);

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_AER_H */
