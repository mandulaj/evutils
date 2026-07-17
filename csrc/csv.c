#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Parse whitespace-delimited integer rows (CSV/TXT event files) into the
 * caller's per-column output arrays.
 *
 *   out_arrays[i]   destination array for output column i (t/x/y/p)
 *   array_types[i]  element width of output column i: 1, 2 or 8 bytes
 *   col_mapping[j]  output column that CSV column j feeds, or -1 to skip it
 *   max_csv_cols    length of col_mapping
 *   max_events      stop after this many rows
 *   bytes_consumed  (out) bytes fully consumed (whole rows only)
 *   events_parsed   (out) rows parsed
 *
 * Strategy: locate each line end with memchr (SIMD-accelerated in glibc), then
 * parse the *complete* line with no per-byte bounds checks -- the newline is a
 * hard terminator, so the digit loops stop at the delimiter/newline naturally.
 * A line with no '\n' before the buffer end is a chunk-boundary fragment: stop
 * without consuming it so the Python driver re-feeds it in the next block.
 */
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
    const char *buffer_end = buffer + buffer_len;
    size_t n_parsed = 0;

    while (cursor < buffer_end && n_parsed < max_events) {
        /* Skip blank lines (bare CR/LF). */
        if (*cursor == '\n' || *cursor == '\r') {
            cursor++;
            continue;
        }

        const char *line_end =
            (const char *)memchr(cursor, '\n', (size_t)(buffer_end - cursor));
        if (line_end == NULL) {
            break;  /* fragment: no complete line in this block */
        }

        const char *scan = cursor;
        int csv_col = 0;
        while (scan < line_end) {
            int dest_col = (csv_col < max_csv_cols) ? col_mapping[csv_col] : -1;

            if (dest_col != -1) {
                while (*scan == ' ' || *scan == '\t') scan++;  /* bounded by line_end */

                int negative = 0;
                if (*scan == '-') { negative = 1; scan++; }
                else if (*scan == '+') { scan++; }

                /* One comparison per digit: (unsigned char)(c - '0') <= 9 is
                 * false for every non-digit -- including the delimiter and
                 * CR/LF -- so the loop terminates without a separate bounds
                 * test. */
                int64_t value = 0;
                unsigned digit;
                while ((digit = (unsigned char)*scan - '0') <= 9u) {
                    value = value * 10 + (int64_t)digit;
                    scan++;
                }
                if (negative) value = -value;

                void *dest = out_arrays[dest_col];
                switch (array_types[dest_col]) {
                    case 2: ((uint16_t *)dest)[n_parsed] = (uint16_t)value; break;
                    case 8: ((int64_t  *)dest)[n_parsed] = value;           break;
                    case 1: ((uint8_t  *)dest)[n_parsed] = (uint8_t)value;  break;
                    default: break;
                }
            }

            /* Advance to the next delimiter within the line (usually already
             * there for clean integer fields, so this rarely steps). */
            while (scan < line_end && *scan != delimiter) scan++;
            csv_col++;
            if (scan < line_end && *scan == delimiter) { scan++; continue; }
            break;
        }

        // Zero-fill any missing columns for short rows to prevent stale data leaks
        while (csv_col < max_csv_cols) {
            int dest_col = col_mapping[csv_col];
            if (dest_col != -1) {
                void *dest = out_arrays[dest_col];
                switch (array_types[dest_col]) {
                    case 2: ((uint16_t *)dest)[n_parsed] = 0; break;
                    case 8: ((int64_t  *)dest)[n_parsed] = 0; break;
                    case 1: ((uint8_t  *)dest)[n_parsed] = 0; break;
                }
            }
            csv_col++;
        }

        cursor = line_end + 1;  /* past the '\n'; a trailing '\r' sits before it */
        n_parsed++;
    }

    *bytes_consumed = (size_t)(cursor - buffer);
    *events_parsed = n_parsed;
    return 0;
}


/* Write `value` as decimal into the tail of `buf` (>= 21 bytes) and return a
 * pointer to the first digit. Writing right-to-left avoids a reversal step. */
static inline char *format_uint(uint64_t value, char *buf) {
    char *digits = buf + 20;
    *digits = '\0';
    if (value == 0) {
        *--digits = '0';
        return digits;
    }
    while (value > 0) {
        *--digits = (char)('0' + (value % 10));
        value /= 10;
    }
    return digits;
}

static inline char *format_int(int64_t value, char *buf) {
    if (value < 0) {
        char *digits = format_uint((uint64_t)(-value), buf);
        *--digits = '-';
        return digits;
    }
    return format_uint((uint64_t)value, buf);
}

/* Serialise the per-column integer arrays into delimited text rows. Mirrors the
 * reader's column layout (array_types[c] is the element width of column c). */
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
    size_t out_pos = 0;
    char digit_buf[24];
    size_t event_index = 0;

    for (; event_index < num_events; event_index++) {
        /* Bail before the row if it might not fit (each column <= 21 chars + a
         * separator). */
        size_t max_row_len = (size_t)(num_columns * 22);
        if (out_pos + max_row_len > out_buffer_len) {
            break;
        }

        for (int col = 0; col < num_columns; col++) {
            char *digits;
            switch (array_types[col]) {
                case 1: digits = format_uint(((uint8_t  *)in_arrays[col])[event_index], digit_buf); break;
                case 2: digits = format_uint(((uint16_t *)in_arrays[col])[event_index], digit_buf); break;
                case 8: digits = format_int(((int64_t   *)in_arrays[col])[event_index], digit_buf); break;
                default: digits = (char *)"0"; break;
            }

            size_t digit_len = (size_t)(digit_buf + 20 - digits);
            memcpy(out_buffer + out_pos, digits, digit_len);
            out_pos += digit_len;

            out_buffer[out_pos++] = (col == num_columns - 1) ? '\n' : delimiter;
        }
    }

    *bytes_written = out_pos;
    *events_written = event_index;
    return 0;
}
