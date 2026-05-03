    def _perform_rescue_sweep(self, board, rows, cols, depth, dictionary, min_word_length, min_words, max_words, excluded, difficulty):
        """
        USER REQUEST: Perform IO operations on random locations until desired word count is reached.
        This is a brute-force rescue for difficult boards (e.g. 5L min on 4x4).
        """
        import time
        start_time = time.time()
        
        # 1. Identify all non-excluded positions
        positions = []
        for f in range(depth):
            for r in range(rows):
                for c in range(cols):
                    pos = (f, r, c) if depth > 1 else (r, c)
                    if pos not in excluded:
                        positions.append(pos)
        
        random.shuffle(positions)
        
        # Solve once to get initial count
        current_solve = self._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=15, store_paths=False)
        current_count = len(current_solve)
        
        if current_count >= min_words:
            return board
            
        print(f"[BoardGen] 🆘 RESCUE SWEEP START (Current: {current_count}, Target: {min_words})")
        
        # Use a subset of positions if the board is huge, to keep it fast
        max_rescue_tiles = 24 if rows * cols >= 48 else len(positions)
        rescue_pool = positions[:max_rescue_tiles]
        
        for pos in rescue_pool:
            if time.time() - start_time > 10.0: # Hard cap on rescue time
                break
                
            f, r, c = (pos[0], pos[1], pos[2]) if depth > 1 else (0, pos[0], pos[1])
            old_char = board[f][r][c] if depth > 1 else board[r][c]
            
            best_char = old_char
            max_score = current_count
            
            # Sample a few random characters + common letters
            test_chars = random.sample("ETAOINSHRDLU", 4) + ["S", "E", "R", "T", "A"]
            test_chars = list(set(test_chars)) # Uniqueness
            
            for char in test_chars:
                if char == old_char: continue
                
                # Forbidden sequence check (Medium/Hard)
                if difficulty in ["Medium", "Hard"] and self._is_creating_forbidden_sequence(board, char, r, c, f, depth=depth):
                    continue
                
                # Apply temporary
                if depth > 1: board[f][r][c] = char
                else: board[r][c] = char
                
                # Solve (Quick depth for speed)
                res = self._solve_board(board, dictionary, (0, 99999), min_word_length, max_depth=12, store_paths=False)
                new_count = len(res)
                
                # We want to increase count but stay under max
                if new_count > max_score and new_count <= max_words:
                    max_score = new_count
                    best_char = char
                
                if max_score >= min_words:
                    break
            
            # Commit best
            if depth > 1: board[f][r][c] = best_char
            else: board[r][c] = best_char
            current_count = max_score
            
            if current_count >= min_words:
                print(f"[BoardGen] ✅ RESCUE SUCCESSFUL: Hit {current_count} words.")
                break
                
        return board
