/* evutils — Prophesee AER CD-event parser.
 *
 * AER is a raw 32-bit-per-event encoding with NO timestamps and NO header:
 *   bits 0-8  : y (9 bits)
 *   bits 9-17 : x (9 bits)
 *   bit  18   : polarity (0: OFF, 1: ON)
 * The 9-bit fields limit coordinates to < 512 (e.g. GenX320).
 *
 * Since the format carries no time information, the parser state selects how
 * the decoded timestamps are generated:
 *   AER_TS_ZERO       : t = 0 for every event (default)
 *   AER_TS_SEQUENTIAL : t = t_start + i * t_step, carried across chunks
 * (User-provided custom timestamp arrays are applied on the Python side.)
 */
#ifndef EVUTILS_AER_H
#define EVUTILS_AER_H

#include "evutils/types.h"
#include "evutils/parser.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aer_state_s aer_state_t;

/* Timestamp generation mode. */
enum {
    AER_TS_ZERO = 0,
    AER_TS_SEQUENTIAL = 1,
};

typedef struct aer_input_buffer_s {
    const uint32_t *begin;
    const uint32_t *end;
} aer_input_buffer_t;

size_t AER_state_size(void);

/* (Re)configure the timestamp generator. `t_next` is the timestamp of the
 * next decoded event; `t_step` the increment per event (sequential mode). */
void AER_state_configure(aer_state_t *state, int32_t mode,
                         uint64_t t_next, uint64_t t_step);

parser_result_t AER_parse_chunk_soa(
    aer_state_t              *state,
    const aer_input_buffer_t *input_buffer,
    event_buffer_soa_t       *event_buffer,
    trigger_buffer_soa_t     *trigger_buffer);

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_AER_H */
