#include "evutils/evt21.h"


enum EVT21_PacketType{
    EVT21_CD_OFF         = 0x0,
    EVT21_CD_ON          = 0x1,
    EVT21_EVT_TIME_HIGH  = 0x8,
    EVT21_EXT_TRIGGER    = 0xA,
    EVT21_OTHERS         = 0xE,
};



typedef struct evt21_state_s {
    uint64_t last_ts_high;
} evt21_state_t;




parser_result_t EVT21_parse_chunk_soa(
    evt21_state_t *state,
    const evt21_input_buffer_t *input_buffer,
    event_buffer_soa_t *event_buffer,
    trigger_buffer_soa_t *trigger_buffer)
{

    const uint64_t *restrict current = input_buffer->begin;
    const uint64_t *restrict end = input_buffer->end;

    // Event output buffers
    size_t n_events_read = event_buffer->size;
    const size_t events_capacity = event_buffer->capacity;

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

    parse_status_t status = EVUTILS_PARSE_OK;

    while(
        current < end &&
        n_events_read < events_capacity &&
        n_triggers_read < triggers_capacity
    ) {
        // printf("Parsing packet %ld, events: %ld\n", i, n_events_read);


        // 31-28 type
        uint64_t type = (*current & 0xF0000000) >> 28;

        // 27-22 ts
        uint64_t ts = last_ts_high | ((*current & 0x0FC00000) >> 22);

        // 11bit 21-11 x
        uint64_t x = ((*current & 0x000FF800) >> 11);

        // 11bit 10-0 y
        uint64_t y = (*current & 0x000007FF);

        // upper remainer data
        uint64_t valid = ((*current & 0xFFFFFFFF00000000) >> 32);

        
        

        if((type == EVT21_CD_OFF || type == EVT21_CD_ON)){
            uint32_t p = !!type;
            uint32_t num_events = __builtin_popcount(valid);
            for(uint32_t j = 0; j < num_events; j++){
                int lz = __builtin_ctz(valid);
                valid &= ~(1 << lz);

                out_ts[n_events_read] = ts;
                out_x[n_events_read] = x + lz;
                out_y[n_events_read] = y;
                out_p[n_events_read] = p;
                n_events_read++;
            }
            continue;
        }


        switch(type){
            case EVT21_EVT_TIME_HIGH:
                last_ts_high = (*current & 0x0FFFFFFF) << 6;
                break;
            case EVT21_EXT_TRIGGER:
                trigger_ts[n_triggers_read] = ts;

                // Trigger id bits 44..40 5bits
                trigger_id[n_triggers_read] = (*current >> 40) & 0x1F;

                trigger_p[n_triggers_read] = (*current >> 32) & 0x1;

                break;
            case EVT21_OTHERS:
                // printf("CD_OTHERS\n");
                break;
            case EVT21_CD_OFF:
            case EVT21_CD_ON:
            default:
                __builtin_unreachable();
                break;

        }
        
    }

    event_buffer->size = n_events_read;
    trigger_buffer->size = n_triggers_read;
    state->last_ts_high = last_ts_high;

    return (parser_result_t){
        .current = (const void *)current,
        .status = status
    };
}
