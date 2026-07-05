
#include "evutils/evt3.h"
#include "evutils/types.h"
#include "evutils/parser.h"

#include <stdio.h>
#include <stdint.h>

#define USE_SIMD 0

#define EVT3_get_packet_type(packet) (((packet) >> 12) & 0xF)
#define EVT3_get_packet_data(packet) ((packet) & 0x0FFF)

enum EVT3_PacketType {
    EVT3_EVT_ADDR_Y = 0x0,
    EVT3_EVT_ADDR_X = 0x2,
    EVT3_VECT_BASE_X = 0x3,
    EVT3_VECT_12 = 0x4,
    EVT3_VECT_8 = 0x5,
    EVT3_EVT_TIME_LOW = 0x6,
    EVT3_CONTINUED_4 = 0x7,
    EVT3_EVT_TIME_HIGH = 0x8,
    EVT3_EXT_TRIGGER = 0xA,
    EVT3_OTHERS = 0xE,
    EVT3_CONTINUED_12 = 0xF
};


typedef struct evt3_state_s {
    uint32_t ts_high_high;
    uint32_t ts_high;
    uint32_t ts_low;
    uint32_t ts;

    uint16_t y;
    uint16_t vecbase_x;
    uint8_t  vecbase_p;
} evt3_state_t;



size_t EVT3_state_size(void) {
    return sizeof(evt3_state_t); 
}

/* Finds the first high time packet in the input buffer. Returns a pointer to
 * the packet, or `end` if none is found. */

uint16_t * EVT3_find_first_high_time(uint16_t *begin, uint16_t *end){
    while(begin < end){
        if (((*begin & 0xF000) >> 12) == EVT3_EVT_TIME_HIGH){
            return begin;
        }
        begin++;
    }
    return end;
}




#if defined(__x86_64__)
  #include <immintrin.h>
#elif defined(__aarch64__)
  #include <arm_neon.h>
#endif
static uint8_t g_shuf16[256][16];
static uint8_t g_cnt8[256];
void evt3_build_tables(void){
    for(int s=0;s<256;++s){
        int o=0;
        for(int j=0;j<8;++j)
            if(s&(1u<<j)){ g_shuf16[s][2*o]=(uint8_t)(2*j); g_shuf16[s][2*o+1]=(uint8_t)(2*j+1); ++o; }
        g_cnt8[s]=(uint8_t)o;
        for(;o<8;++o){ g_shuf16[s][2*o]=0; g_shuf16[s][2*o+1]=1; }
    }
}

static inline size_t EVT3_expand_simd(
    uint16_t bx, timestamp_t ts, uint16_t y, uint8_t p, uint32_t m, size_t n,
    uint16_t* restrict x, timestamp_t* restrict t, uint16_t* restrict yy, uint8_t* restrict pp)
{
#if defined(__x86_64__)
    const __m128i iota = _mm_setr_epi16(0,1,2,3,4,5,6,7);
    const size_t n0 = n;
    for(int k=0;k<4;++k){
        uint8_t s = (uint8_t)(m >> (8*k));
        __m128i idx = _mm_add_epi16(_mm_set1_epi16((short)(bx+8*k)), iota);
        __m128i sh  = _mm_loadu_si128((const __m128i*)g_shuf16[s]);
        __m128i out = _mm_shuffle_epi8(idx, sh);
        _mm_storeu_si128((__m128i*)&x[n], out);
        n += g_cnt8[s];
    }
    size_t cnt = n - n0;
    _mm_storeu_si128((__m128i*)&t[n0],   _mm_set1_epi32((int)ts));
    _mm_storeu_si128((__m128i*)&t[n0+4], _mm_set1_epi32((int)ts));
    _mm_storeu_si128((__m128i*)&yy[n0],  _mm_set1_epi16((short)y));
    _mm_storeu_si128((__m128i*)&pp[n0],  _mm_set1_epi8((char)p));
    if(unlikely(cnt>8)){
        for(size_t i=8;i<cnt;++i){ t[n0+i]=ts; yy[n0+i]=y; pp[n0+i]=p; }
    }
    return n;
#elif defined(__aarch64__)
    const uint16x8_t iota = (uint16x8_t){0,1,2,3,4,5,6,7};
    const size_t n0 = n;
    for(int k=0;k<4;++k){
        uint8_t s = (uint8_t)(m >> (8*k));
        uint16x8_t idx = vaddq_u16(vdupq_n_u16((uint16_t)(bx+8*k)), iota);
        uint8x16_t sh  = vld1q_u8(g_shuf16[s]);
        uint8x16_t out = vqtbl1q_u8(vreinterpretq_u8_u16(idx), sh);
        vst1q_u16(&x[n], vreinterpretq_u16_u8(out));
        n += g_cnt8[s];
    }
    size_t cnt = n - n0;
    vst1q_u32(&t[n0],   vdupq_n_u32(ts));
    vst1q_u32(&t[n0+4], vdupq_n_u32(ts));
    vst1q_u16(&yy[n0],  vdupq_n_u16(y));
    vst1q_u8 (&pp[n0],  vdupq_n_u8(p));
    if(unlikely(cnt>8)){
        for(size_t i=8;i<cnt;++i){ t[n0+i]=ts; yy[n0+i]=y; pp[n0+i]=p; }
    }
    return n;
#else
    #error "no SIMD path for this target"
#endif
}



// static inline const uint16_t * EVT3_parse_vector_12_12_8_soa(
//     const uint16_t * __restrict__ current,
//     evt3_state_t * __restrict__ state,
//     uint32_t*  __restrict__ out_ts,
//     uint16_t*  __restrict__ out_x,
//     uint16_t*  __restrict__ out_y,
//     uint8_t*   __restrict__ out_p,
//     size_t * n_events) {

//     uint32_t vec_valid = EVT3_get_packet_data(*current);
//     current++;
//     if(likely(EVT3_get_packet_type(*current) == EVT3_VECT_12)) {
//         vec_valid |= (uint32_t)(EVT3_get_packet_data(*current) << 12);
//         current++;
//         if (likely(EVT3_get_packet_type(*current) == EVT3_VECT_8)) {
//             vec_valid |= (uint32_t)(*current & 0x00FF) << 24;
//             current++;
//         }
//     }

//     *n_events = EVT3_expand_simd(
//         state->vecbase_x,   /* bx */
//         state->ts,          /* ts */
//         state->y,           /* y  */
//         state->vecbase_p,   /* p  */
//         vec_valid,          /* m  */
//         *n_events,          /* n  */
//         out_x,              /* x  array  <- out_x, NOT out_ts */
//         out_ts,             /* t  array */
//         out_y,              /* y  array */
//         out_p);             /* p  array */

//     state->vecbase_x += 32u;
//     return current;
// }


// static inline const uint16_t * EVT3_parse_vector_12_12_8_soa(
//     const uint16_t * __restrict__ current,
//     evt3_state_t * __restrict__ state,
//     uint64_t* __restrict__ out_ts, uint16_t* __restrict__ out_x,
//     uint16_t* __restrict__ out_y,  uint8_t* __restrict__ out_p,
//     size_t * n_events)
// {
//     uint32_t w0 = current[0], w1 = current[1], w2 = current[2];
//     uint32_t vec_valid = (w0 & 0x0FFF);
//     int has1 = ((w1 >> 12) == EVT3_VECT_12);
//     int has2 = has1 && ((w2 >> 12) == EVT3_VECT_8);
//     vec_valid |= has1 ? ((w1 & 0x0FFF) << 12) : 0u;
//     vec_valid |= has2 ? ((uint32_t)(w2 & 0x00FF) << 24) : 0u;
//     current += 1 + has1 + has2;

//     #if USE_SIMD

//     *n_events = EVT3_expand_simd(state->vecbase_x, state->ts, state->y, state->vecbase_p,
//                                  vec_valid, *n_events, out_x, out_ts, out_y, out_p);

//     #else



    


//     #endif
//     state->vecbase_x += 32u;
//     return current;
// }




#define EMIT_SOA() do { \
    unsigned has = (vec_valid != 0u); \
    uint32_t lz = (uint32_t)__builtin_ctz(vec_valid | 0x80000000u); \
    out_ts[n] = ts; \
    out_x[n] = (uint16_t)(bx + lz); \
    out_y[n] = y; \
    out_p[n] = p; \
    n += has; \
    vec_valid &= vec_valid - 1u; \
} while (0)


static inline const uint16_t * EVT3_parse_vector_12_12_8_soa(
    const uint16_t * __restrict__ current,
    evt3_state_t * __restrict__ state,
    timestamp_t* __restrict__ out_ts, uint16_t* __restrict__ out_x,
    uint16_t* __restrict__ out_y,  uint8_t* __restrict__ out_p,
    size_t * n_events) {

    uint32_t vec_valid = EVT3_get_packet_data(*current);
    current++;

    if(likely(EVT3_get_packet_type(*current) == EVT3_VECT_12)) {
        vec_valid |= (uint32_t)(EVT3_get_packet_data(*current) << 12);
        current++;

        if (likely(EVT3_get_packet_type(*current) == EVT3_VECT_8)) {
            vec_valid |= (uint32_t)(*current & 0x00FF) << 24;
            current++;
        }
    }
    const uint32_t ts = state->ts;
    const uint16_t y  = state->y;
    const uint8_t  p  = state->vecbase_p;
    const uint16_t bx = state->vecbase_x;
    size_t n = *n_events;

    EMIT_SOA(); EMIT_SOA(); EMIT_SOA(); EMIT_SOA();   /* 4 unconditional — no branch between them */

    while (vec_valid) { EMIT_SOA(); }     /* tail: only pop>4 (~23%) re-enters */

    *n_events = n;
    state->vecbase_x = bx + 32;

    return current;
}



#define EVT3_INPUT_PADDING 4
parser_result_t EVT3_parse_chunk_soa(
    evt3_state_t *state,
    const evt3_input_buffer_t *input_buffer,
    event_buffer_soa_t *event_buffer,
    trigger_buffer_soa_t *trigger_buffer) {

    // Hoist variables for better optimization
    const uint16_t *restrict current = input_buffer->begin;
    const uint16_t * end_offset = input_buffer->end - EVT3_INPUT_PADDING;

    // Event output buffers
    const size_t events_capacity = event_buffer->capacity;
    const size_t events_capacity_offset = events_capacity - 64; // We need to keep some space for vector messages
    size_t n_events_read = event_buffer->size;

    timestamp_t* restrict out_ts = event_buffer->t;
    uint16_t* restrict out_x = event_buffer->x;
    uint16_t* restrict out_y = event_buffer->y;
    uint8_t* restrict out_p = event_buffer->p;
    
    // Trigger output buffers
    const size_t triggers_capacity = trigger_buffer->capacity;
    size_t n_triggers_read = trigger_buffer->size;

    timestamp_t* restrict trigger_ts = trigger_buffer->t;
    uint8_t* restrict trigger_id = trigger_buffer->id;
    uint8_t* restrict trigger_p = trigger_buffer->p;
    
    // State variables
    evt3_state_t local_state = *state;

    parse_status_t status = EVUTILS_PARSE_OK;


    // Main parsing looop
    while(
        current < end_offset &&
        n_events_read < events_capacity_offset &&
        n_triggers_read < triggers_capacity) {

        __builtin_prefetch(current + 64, 0, 0);
        uint32_t packet_type = EVT3_get_packet_type(*current);
        uint32_t packet_data = EVT3_get_packet_data(*current);

        // Most common case checks first
        if(packet_type == EVT3_EVT_ADDR_X) {
            out_ts[n_events_read] = local_state.ts;
            out_y[n_events_read] = local_state.y;
            out_x[n_events_read] = (packet_data & 0x7FF);
            out_p[n_events_read] = (packet_data & 0x800) >> 11;
            n_events_read++;
            current++;
            continue;
        }
        if(packet_type == EVT3_VECT_BASE_X) {
            // Implementation omitted for brevity
            local_state.vecbase_x  = packet_data & 0x7FF;
            local_state.vecbase_p = (packet_data & 0x800) >> 11;

            current++;

            // Parse vector messages
            while (EVT3_get_packet_type(*current) == EVT3_VECT_12 && current < end_offset && n_events_read < events_capacity_offset) {
                current = EVT3_parse_vector_12_12_8_soa(current, &local_state, out_ts, out_x, out_y, out_p, &n_events_read);
            }

            continue;
        }

        switch(packet_type) {
            case EVT3_EVT_ADDR_Y:
                local_state.y = packet_data & 0x7FF;
                break;
            case EVT3_EVT_TIME_HIGH:
                {
                    uint32_t new_ts_high = packet_data << 12;

                    if(unlikely(new_ts_high < local_state.ts_high)) {
                        local_state.ts_high_high += 0x1000000;
                    }
                    local_state.ts_high = new_ts_high;

                    current++;
                    packet_type = EVT3_get_packet_type(*current);

                    if(unlikely(packet_type != EVT3_EVT_TIME_LOW)) {
                        continue;
                    }
                    packet_data = EVT3_get_packet_data(*current);
                }
                __attribute__((fallthrough));
            case EVT3_EVT_TIME_LOW:
                local_state.ts_low = packet_data;
                local_state.ts = local_state.ts_high_high | local_state.ts_high | local_state.ts_low;
                break;
            case EVT3_EXT_TRIGGER:
                trigger_ts[n_triggers_read] = local_state.ts;
                trigger_id[n_triggers_read] = packet_data >> 8;
                trigger_p[n_triggers_read] = packet_data & 0x1;
                n_triggers_read++;
                break;
            case EVT3_VECT_12:
                current = EVT3_parse_vector_12_12_8_soa(current, &local_state, out_ts, out_x, out_y, out_p, &n_events_read);
                continue;
                break;
            case EVT3_VECT_8:
            case EVT3_EVT_ADDR_X:
            case EVT3_VECT_BASE_X:
                // printf("Unexpected packet type %d in vector parsing\n", packet_type);
                __builtin_unreachable();
                break;
            case EVT3_OTHERS:
            case EVT3_CONTINUED_12:
            case EVT3_CONTINUED_4:
            default:
                break;
        }
        current++;
    }

    *state = local_state;

    event_buffer->size = n_events_read;
    trigger_buffer->size = n_triggers_read;


    return (parser_result_t){
        .current = (const void *)current,
        .status = status
    };
}






#define EMIT() do { \
    unsigned has = (vec_valid != 0u); \
    uint32_t lz = (uint32_t)__builtin_ctz(vec_valid | 0x80000000u); \
    events[n].t = ts; \
    events[n].x = (uint16_t)(bx + lz); \
    events[n].y = y; \
    events[n].p = p; \
    n += has; \
    vec_valid &= vec_valid - 1u; \
} while (0)


static inline const uint16_t * EVT3_parse_vector_12_12_8(
    const uint16_t * __restrict__ current,
    evt3_state_t * __restrict__ state,
    event_t*  __restrict__ events,
    size_t * n_events) {

    uint32_t vec_valid = EVT3_get_packet_data(*current);
    current++;

    if(likely(EVT3_get_packet_type(*current) == EVT3_VECT_12)) {
        vec_valid |= (uint32_t)(EVT3_get_packet_data(*current) << 12);
        current++;

        if (likely(EVT3_get_packet_type(*current) == EVT3_VECT_8)) {
            vec_valid |= (uint32_t)(*current & 0x00FF) << 24;
            current++;
        }
    }
    const uint32_t ts = state->ts;
    const uint16_t y  = state->y;
    const uint8_t  p  = state->vecbase_p;
    const uint16_t bx = state->vecbase_x;
    size_t n = *n_events;

    EMIT(); EMIT(); EMIT(); EMIT();   /* 4 unconditional — no branch between them */

    while (vec_valid) { EMIT(); }     /* tail: only pop>4 (~23%) re-enters */

    *n_events = n;
    state->vecbase_x = bx + 32;

    return current;
}



#define EVT3_INPUT_PADDING 4
parser_result_t EVT3_parse_chunk(
    evt3_state_t *state,
    const evt3_input_buffer_t *input_buffer,
    event_buffer_t *event_buffer,
    trigger_buffer_t *trigger_buffer) {

    // Hoist variables for better optimization
    const uint16_t *restrict current = input_buffer->begin;
    const uint16_t *restrict end_offset = input_buffer->end - EVT3_INPUT_PADDING;

    event_t* restrict events = event_buffer->events;
    const size_t events_capacity = event_buffer->capacity;
    const size_t events_capacity_offset = events_capacity - 32; // We need to keep some space for vector messages
    size_t n_events_read = event_buffer->size;

    trigger_t* restrict triggers = trigger_buffer->triggers;
    const size_t triggers_capacity = trigger_buffer->capacity;
    size_t n_triggers_read = trigger_buffer->size;

    evt3_state_t local_state = *state;

    parse_status_t status = EVUTILS_PARSE_OK;


    // Main parsing looop
    while(
        current < end_offset &&
        n_events_read < events_capacity_offset &&
        n_triggers_read < triggers_capacity) {

        uint32_t packet_type = EVT3_get_packet_type(*current);
        uint32_t packet_data = EVT3_get_packet_data(*current);

        // Most common case checks first
        if(packet_type == EVT3_EVT_ADDR_X) {
            events[n_events_read].t = local_state.ts;
            events[n_events_read].y = local_state.y;
            events[n_events_read].x = (packet_data & 0x7FF);
            events[n_events_read].p = (packet_data & 0x800) >> 11;
            n_events_read++;
            current++;
            continue;
        }
        if(packet_type == EVT3_VECT_BASE_X) {
            // Implementation omitted for brevity
            local_state.vecbase_x  = packet_data & 0x7FF;
            local_state.vecbase_p = (packet_data & 0x800) >> 11;

            current++;

            // Parse vector messages
            while (EVT3_get_packet_type(*current) == EVT3_VECT_12 && current < end_offset && n_events_read < events_capacity_offset) {
                current = EVT3_parse_vector_12_12_8(current, &local_state, events, &n_events_read);
            }

            continue;
        }

        switch(packet_type) {
            case EVT3_EVT_ADDR_Y:
                local_state.y = packet_data & 0x7FF;
                break;
            case EVT3_EVT_TIME_HIGH:
                {
                    uint32_t new_ts_high = packet_data << 12;

                    if(unlikely(new_ts_high < local_state.ts_high)) {
                        local_state.ts_high_high += 0x1000000;
                    }
                    local_state.ts_high = new_ts_high;

                    current++;
                    packet_type = EVT3_get_packet_type(*current);

                    if(unlikely(packet_type != EVT3_EVT_TIME_LOW)) {
                        continue;
                    }
                    packet_data = EVT3_get_packet_data(*current);
                }
                __attribute__((fallthrough));
            case EVT3_EVT_TIME_LOW:
                local_state.ts_low = packet_data;
                local_state.ts = local_state.ts_high_high | local_state.ts_high | local_state.ts_low;
                break;
            case EVT3_EXT_TRIGGER:
                triggers[n_triggers_read].t = local_state.ts;
                triggers[n_triggers_read].id = packet_data >> 8;
                triggers[n_triggers_read].p = packet_data & 0x1;
                n_triggers_read++;
                break;
            case EVT3_VECT_12:
                current = EVT3_parse_vector_12_12_8(current, &local_state, events, &n_events_read);
                continue;
                break;
            case EVT3_VECT_8:
            case EVT3_EVT_ADDR_X:
            case EVT3_VECT_BASE_X:
                // printf("Unexpected packet type %d in vector parsing\n", packet_type);
                __builtin_unreachable();
                break;
            case EVT3_OTHERS:
            case EVT3_CONTINUED_12:
            case EVT3_CONTINUED_4:
            default:
                break;
        }
        current++;
    }

    *state = local_state;

    event_buffer->size = n_events_read;
    trigger_buffer->size = n_triggers_read;


    return (parser_result_t){
        .current = (const void *)current,
        .status = status
    };
}