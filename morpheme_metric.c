/* High-Performance Morpheme Metric C Engine */
#include <string.h>
#include <stdlib.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void backtrack(
    const char *target, int t_len, int t_idx,
    const int *s_indices_counts, const int s_indices[26][32],
    unsigned int used_mask, int m_len, int min_s, int max_s,
    int tails_len, int *tails,
    int *best_mp
) {
    // 1. Insertion bound: number of insertions already made
    int insertions_so_far = t_idx - m_len;
    if (insertions_so_far >= *best_mp) return;

    // 2. Cost bound so far
    if (m_len > 0) {
        int relocations = m_len - tails_len;
        int paid_deletions = (max_s - min_s + 1) - m_len;
        int min_cost = relocations + paid_deletions + insertions_so_far;
        if (min_cost >= *best_mp) return;
    }

    // 3. Base case: end of target string reached
    if (t_idx == t_len) {
        if (m_len > 0) {
            int relocations = m_len - tails_len;
            int paid_deletions = (max_s - min_s + 1) - m_len;
            int actual_cost = relocations + paid_deletions + (t_len - m_len);
            if (actual_cost < *best_mp) {
                *best_mp = actual_cost;
            }
        }
        return;
    }

    int c = target[t_idx] - 'A';
    if (c >= 0 && c < 26) {
        int count = s_indices_counts[c];
        for (int i = 0; i < count; i++) {
            int s_idx = s_indices[c][i];
            if (!(used_mask & (1U << s_idx))) {
                // Update tails (LIS) with binary search (Patience Sorting)
                int l = 0, r = tails_len;
                while (l < r) {
                    int mid = (l + r) / 2;
                    if (tails[mid] >= s_idx) r = mid;
                    else l = mid + 1;
                }
                int old_val = (l < tails_len) ? tails[l] : -1;
                int old_len = tails_len;
                tails[l] = s_idx;
                int new_tails_len = (l == tails_len) ? tails_len + 1 : tails_len;

                int new_min = (m_len == 0) ? s_idx : MIN(min_s, s_idx);
                int new_max = (m_len == 0) ? s_idx : MAX(max_s, s_idx);

                backtrack(target, t_len, t_idx + 1, s_indices_counts, s_indices,
                          used_mask | (1U << s_idx), m_len + 1, new_min, new_max,
                          new_tails_len, tails, best_mp);

                // Restore tails
                if (old_val != -1) tails[l] = old_val;
                tails_len = old_len;
            }
        }
    }

    // Skip target char (insertion)
    backtrack(target, t_len, t_idx + 1, s_indices_counts, s_indices,
              used_mask, m_len, min_s, max_s,
              tails_len, tails, best_mp);
}

int c_calculate_morpheme_metric(const char *source, const char *target, int limit) {
    int s_len = (int)strlen(source);
    int t_len = (int)strlen(target);
    if (s_len == 0 || t_len == 0) return 99;
    if (strstr(source, target) != NULL) return 0;

    // LCS (Linearity)
    int prev[64] = {0};
    int curr[64] = {0};
    for (int i = 0; i < s_len; i++) {
        for (int j = 1; j <= t_len; j++) {
            if (source[i] == target[j-1]) {
                curr[j] = prev[j-1] + 1;
            } else {
                curr[j] = MAX(prev[j], curr[j-1]);
            }
        }
        memcpy(prev, curr, sizeof(int) * (t_len + 1));
    }
    int linearity = prev[t_len];
    if (linearity == 0 || t_len - linearity > limit) return 99;

    int s_indices_counts[26] = {0};
    int s_indices[26][32];
    for (int i = 0; i < s_len; i++) {
        int c = source[i] - 'A';
        if (c >= 0 && c < 26 && s_indices_counts[c] < 32) {
            s_indices[c][s_indices_counts[c]++] = i;
        }
    }

    int best_mp = limit + 1;
    int tails[64] = {0};
    backtrack(target, t_len, 0, s_indices_counts, s_indices, 0, 0, 99, -1, 0, tails, &best_mp);

    return best_mp;
}
