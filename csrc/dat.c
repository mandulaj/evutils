#include "evutils/dat.h"

typedef struct dat_state_s {
    uint64_t ts_offset;   /* accumulated 2^32 timestamp wraps */
    uint32_t last_ts;     /* last raw 32-bit timestamp seen    */
    uint8_t  have_last;   /* whether last_ts is valid          */
} dat_state_t;

size_t DAT_state_size(void) {
    return sizeof(dat_state_t);
}

EVUTILS_TARGET_CLONES
parser_result_t DAT_parse_chunk_soa(
    dat_state_t              *state,
    const dat_input_buffer_t *input_buffer,
    event_buffer_soa_t       *event_buffer,
    trigger_buffer_soa_t     *trigger_buffer) {

    (void)trigger_buffer;  /* DAT CD files carry no triggers */

    const uint32_t *restrict current = input_buffer->begin;
    const uint32_t *restrict end = input_buffer->end;

    timestamp_t* restrict out_ts = event_buffer->t;
    uint16_t*    restrict out_x = event_buffer->x;
    uint16_t*    restrict out_y = event_buffer->y;
    uint8_t*     restrict out_p = event_buffer->p;

    size_t n = event_buffer->size;
    const size_t capacity = event_buffer->capacity;

    uint64_t ts_offset = state->ts_offset;
    uint32_t last_ts = state->last_ts;
    uint8_t  have_last = state->have_last;

    parse_status_t status = EVUTILS_PARSE_OK;

    /* Each event is two uint32 words: [timestamp, data]. */
    while (current + 2 <= end && n < capacity) {
        uint32_t ts_raw = current[0];
        uint32_t data = current[1];
        current += 2;

        if (have_last && ts_raw < last_ts) {
            ts_offset += (uint64_t)1 << 32;  /* 32-bit timestamp wrapped */
        }
        last_ts = ts_raw;
        have_last = 1;

        out_ts[n] = ts_offset | (uint64_t)ts_raw;
        out_x[n] = (uint16_t)(data & 0x3FFF);
        out_y[n] = (uint16_t)((data >> 14) & 0x3FFF);
        out_p[n] = (uint8_t)((data >> 28) & 0x1);
        n++;
    }

    event_buffer->size = n;
    state->ts_offset = ts_offset;
    state->last_ts = last_ts;
    state->have_last = have_last;

    return (parser_result_t){
        .current = (const void *)current,
        .status = status
    };
}
