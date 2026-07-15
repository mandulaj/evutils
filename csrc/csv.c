#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int evutils_read_csv(
    const char *buffer, size_t buffer_len,
    char delimiter,
    void **out_arrays,
    int *array_types,
    int *col_mapping,
    int max_csv_cols,
    size_t max_events,
    size_t *bytes_consumed,
    size_t *events_parsed
) {
    const char *cursor = buffer;
    const char *end = buffer + buffer_len;
    size_t count = 0;

    /* Single pass: parse each field in place and detect the newline inline,
     * so every byte is read once (the old code pre-scanned the whole line to
     * find its end, then re-scanned each field). A line without a terminating
     * newline before `end` is a chunk-boundary fragment: stop without consuming
     * it, so the Python driver re-feeds it with the next block. */
    while (cursor < end && count < max_events) {
        /* Skip blank lines (bare CR/LF). */
        if (*cursor == '\n' || *cursor == '\r') {
            cursor++;
            continue;
        }

        const char *p = cursor;
        int col_idx = 0;
        int saw_newline = 0;

        while (p < end) {
            int target_idx = (col_idx < max_csv_cols) ? col_mapping[col_idx] : -1;

            if (target_idx != -1) {
                /* Parse a (possibly signed) integer starting at p. */
                int64_t val = 0;
                int neg = 0;
                while (p < end && (*p == ' ' || *p == '\t')) p++;
                if (p < end && *p == '-') { neg = 1; p++; }
                else if (p < end && *p == '+') { p++; }
                while (p < end && *p >= '0' && *p <= '9') {
                    val = val * 10 + (*p - '0');
                    p++;
                }
                if (neg) val = -val;

                int type = array_types[target_idx];
                void *col = out_arrays[target_idx];
                if (type == 2) {
                    ((uint16_t*)col)[count] = (uint16_t)val;
                } else if (type == 8) {
                    ((int64_t*)col)[count] = val;
                } else if (type == 1) {
                    ((uint8_t*)col)[count] = (uint8_t)val;
                }
            }

            /* Advance to the next delimiter or line end. */
            while (p < end && *p != delimiter && *p != '\n' && *p != '\r') p++;
            col_idx++;

            if (p < end && *p == delimiter) { p++; continue; }
            if (p < end && (*p == '\n' || *p == '\r')) saw_newline = 1;
            break;
        }

        if (!saw_newline) break;  /* fragment at the chunk boundary: re-feed */

        while (p < end && (*p == '\n' || *p == '\r')) p++;  /* consume EOL (\r\n) */
        cursor = p;
        count++;
    }

    *bytes_consumed = cursor - buffer;
    *events_parsed = count;
    return 0;
}


static inline char* itoa_positive(uint64_t val, char *buf) {
    char *p = buf + 20;
    *p = '\0';
    if (val == 0) {
        *--p = '0';
        return p;
    }
    while (val > 0) {
        *--p = '0' + (val % 10);
        val /= 10;
    }
    return p;
}

static inline char* itoa_signed(int64_t val, char *buf) {
    if (val < 0) {
        char *p = itoa_positive((uint64_t)(-val), buf);
        *--p = '-';
        return p;
    }
    return itoa_positive((uint64_t)val, buf);
}

int evutils_write_csv(
    void **in_arrays,
    int *array_types,
    int num_columns,
    char delimiter,
    size_t num_events,
    char *out_buffer,
    size_t out_buffer_len,
    size_t *bytes_written,
    size_t *events_written
) {
    size_t buf_pos = 0;
    char num_buf[24];
    size_t i = 0;
    
    for (; i < num_events; i++) {
        size_t max_req_len = (size_t)(num_columns * 22);
        if (buf_pos + max_req_len > out_buffer_len) {
            break;
        }
        
        for (int c = 0; c < num_columns; c++) {
            int type = array_types[c];
            char *s;
            if (type == 1) {
                s = itoa_positive(((uint8_t*)in_arrays[c])[i], num_buf);
            } else if (type == 2) {
                s = itoa_positive(((uint16_t*)in_arrays[c])[i], num_buf);
            } else if (type == 8) {
                s = itoa_signed(((int64_t*)in_arrays[c])[i], num_buf);
            } else {
                s = "0";
            }
            
            size_t len = num_buf + 20 - s;
            memcpy(out_buffer + buf_pos, s, len);
            buf_pos += len;
            
            if (c == num_columns - 1) {
                out_buffer[buf_pos++] = '\n';
            } else {
                out_buffer[buf_pos++] = delimiter;
            }
        }
    }
    
    *bytes_written = buf_pos;
    *events_written = i;
    return 0;
}
