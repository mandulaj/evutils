#include "evutils/evt4.h"

/* EVT4 timestamp reconstruction is identical to EVT2: the 28-bit TIME_HIGH
 * field carries time-base bits 6..33 (so it is shifted left by 6), and the CD /
 * trigger words carry the low 6 bits. The field wraps at 2^34; bump the overflow
 * accumulator each time it does. */
typedef struct evt4_state_s {
    uint64_t last_ts_high;   /* current TIME_HIGH << 6 (bits 6..33)            */
    uint64_t ts_high_high;   /* accumulated overflow of the 28-bit field (34+) */
} evt4_state_t;

#define EVT4_TS_WRAP (1ULL << 34)


size_t EVT4_state_size(void) {
    return sizeof(evt4_state_t);
}


/* 4-bit type field in bits 28..31 of each 32-bit word. */
enum EVT4_PacketType {
    EVT4_OTHERS        = 0x6,
    EVT4_CONTINUED     = 0x7,
    EVT4_EXT_TRIGGER   = 0x9,
    EVT4_CD_OFF        = 0xA,   /* single CD event, polarity 0 */
    EVT4_CD_ON         = 0xB,   /* single CD event, polarity 1 */
    EVT4_CD_VEC_OFF    = 0xC,   /* vector CD base, polarity 0; next word = mask */
    EVT4_CD_VEC_ON     = 0xD,   /* vector CD base, polarity 1; next word = mask */
    EVT4_EVT_TIME_HIGH = 0xE,
    EVT4_PADDING       = 0xF
};


EVUTILS_TARGET_CLONES
parser_result_t EVT4_parse_chunk_soa(
    evt4_state_t *state,
    const evt4_input_buffer_t *input_buffer,
    event_buffer_soa_t *event_buffer,
    trigger_buffer_soa_t *trigger_buffer) {

    const uint32_t *restrict current = input_buffer->begin;
    const uint32_t *restrict end = input_buffer->end;

    // Event output buffers. A single vector word expands to at most 32 events
    // (one per set bit of the 32-bit mask), so stop early enough to always have
    // room for a full expansion without a per-bit capacity check.
    size_t n_events_read = event_buffer->size;
    const size_t events_capacity = event_buffer->capacity;
    // Guard the size_t subtraction: capacity <= 32 would wrap to a huge offset.
    const size_t events_capacity_offset = events_capacity > 32 ? events_capacity - 32 : 0;

    timestamp_t* restrict out_ts = event_buffer->t;
    uint16_t* restrict out_x = event_buffer->x;
    uint16_t* restrict out_y = event_buffer->y;
    uint8_t* restrict out_p = event_buffer->p;

    // Trigger output buffers
    timestamp_t* restrict trigger_ts = trigger_buffer->t;
    uint8_t* restrict trigger_id = trigger_buffer->id;
    uint8_t* restrict trigger_p = trigger_buffer->p;

    size_t n_triggers_read = trigger_buffer->size;
    const size_t triggers_capacity = trigger_buffer->capacity;

    // State variables
    uint64_t last_ts_high = state->last_ts_high;
    uint64_t ts_high_high = state->ts_high_high;

    parse_status_t status = EVUTILS_PARSE_OK;

    while(
        current < end &&
        n_events_read < events_capacity_offset &&
        n_triggers_read < triggers_capacity
    ) {

        uint32_t packet_type = (*current & 0xF0000000) >> 28;
        uint32_t packet_data = *current & 0x0FFFFFFF;

        switch(packet_type){
            case EVT4_CD_OFF:
            case EVT4_CD_ON:
                // Bits 27..22 low timestamp, 21..11 x, 10..0 y (same as EVT2).
                out_ts[n_events_read] = ts_high_high | last_ts_high | ((packet_data >> 22) & 0x3F);
                out_x[n_events_read] = (packet_data >> 11) & 0x7FF;
                out_y[n_events_read] = packet_data & 0x7FF;
                out_p[n_events_read] = packet_type & 1;   // CD_ON (0xB) -> 1
                n_events_read++;
                break;
            case EVT4_CD_VEC_OFF:
            case EVT4_CD_VEC_ON: {
                // Vector CD: this word is the base (x, y, low ts, polarity); the
                // FOLLOWING word is a 32-bit validity mask. Each set bit `off`
                // emits an event at x = base_x + off, same y/ts/polarity.
                // If the mask word is not in this chunk, stop without consuming
                // the base word so the caller resumes the whole group next time.
                if (current + 1 >= end) {
                    status = EVUTILS_PARSE_INPUT_EMPTY;
                    goto done;
                }
                const uint64_t ts = ts_high_high | last_ts_high | ((packet_data >> 22) & 0x3F);
                const uint16_t base_x = (packet_data >> 11) & 0x7FF;
                const uint16_t y = packet_data & 0x7FF;
                const uint8_t p = packet_type & 1;

                uint32_t mask = *(current + 1);
                while (mask) {
                    uint32_t off = (uint32_t)__builtin_ctz(mask);
                    out_ts[n_events_read] = ts;
                    out_x[n_events_read] = (uint16_t)(base_x + off);
                    out_y[n_events_read] = y;
                    out_p[n_events_read] = p;
                    n_events_read++;
                    mask &= mask - 1u;
                }
                current++;   // consume the mask word (base word consumed below)
                break;
            }
            case EVT4_EVT_TIME_HIGH:
                {
                    // Bits 27..0 are event-time bits 33..6. Track 28-bit wraps.
                    uint64_t new_ts_high = (uint64_t)packet_data << 6;
                    if (new_ts_high < last_ts_high) {
                        ts_high_high += EVT4_TS_WRAP;
                    }
                    last_ts_high = new_ts_high;
                }
                break;
            case EVT4_EXT_TRIGGER:
                // value @ bit 0, id @ bits 8..12 (5 bits), low ts @ bits 22..27.
                trigger_ts[n_triggers_read] = ts_high_high | last_ts_high | ((packet_data >> 22) & 0x3F);
                trigger_id[n_triggers_read] = (packet_data >> 8) & 0x1F;
                trigger_p[n_triggers_read] = packet_data & 0x1;
                n_triggers_read++;
                break;
            case EVT4_OTHERS:
            case EVT4_CONTINUED:
            case EVT4_PADDING:
            default:
                break;
        }
        current++;
    }

done:
    event_buffer->size = n_events_read;
    trigger_buffer->size = n_triggers_read;
    state->last_ts_high = last_ts_high;
    state->ts_high_high = ts_high_high;

    /* Report why parsing stopped: output space exhausted vs input drained. */
    if (n_events_read >= events_capacity_offset || n_triggers_read >= triggers_capacity) {
        status = EVUTILS_PARSE_OUTPUT_FULL;
    }

    return (parser_result_t){
        .current = (const void *)current,
        .status = status
    };
}
