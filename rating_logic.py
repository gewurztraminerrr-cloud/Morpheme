import math

def is_player_guest(player):
    """Helper to consistently identify guest players across modules"""
    # Check for Guest_ prefix in username (case-insensitive) OR is_guest flag
    username = getattr(player, 'username', '')
    if username.lower().startswith('guest_'):
        return True
    if getattr(player, 'is_guest', False):
        return True
    # If user_id is <= 0, also usually a placeholder or guest
    if getattr(player, 'user_id', 0) <= 0:
        return True
    return False

def calculate_proportional_rating_change(players, is_private=False):
    """
    Calculates rating changes based on relative performance.
    BOTS and GUESTS are excluded from calculations and do not influence human ratings.
    """
    changes = {p.user_id: 0 for p in players}
    
    print(f"[RatingLogic] Calculating change for {len(players)} total players.")
    for p in players:
        print(f"  - Player {getattr(p, 'username', 'N/A')}: id={p.user_id}, AI={getattr(p, 'is_ai', False)}, Guest={is_player_guest(p)}, MidRound={getattr(p, 'joined_mid_round', False)}")

    # Filter to only including competitive human players (non-bots, non-guests, non-mid-round-joiners)
    competitive_humans = [
        p for p in players 
        if not getattr(p, 'is_ai', False) and not is_player_guest(p) and not getattr(p, 'joined_mid_round', False)
    ]
    
    print(f"[RatingLogic] Found {len(competitive_humans)} competitive human players.")
    if not competitive_humans:
        return changes

    # Count players among competitive humans who were actually present AND active
    # "Did not play" = 0 score AND no words found (valid or invalid)
    active_pool = [
        p for p in competitive_humans 
        if getattr(p, 'score', 0) > 0 or getattr(p, 'submitted_words', []) or getattr(p, 'invalid_words', [])
    ]
    
    number_of_players = len(active_pool)
    print(f"[RatingLogic] Found {number_of_players} active participants out of {len(competitive_humans)} humans.")
    for p in active_pool:
        print(f"  - Active: {getattr(p, 'username', 'N/A')} (Score: {getattr(p, 'score', 0)})")
            
    if number_of_players < 2:
        print(f"[RatingLogic] ABORTING: Not enough active human players ({number_of_players}).")
        return changes
    
    # Calculate scoreSum and ratingSum
    score_sum = sum(getattr(p, 'score', 0) for p in active_pool)
    rating_sum = sum(getattr(p, 'rating', 1200) for p in active_pool)
    
    print(f"[RatingLogic] Totals - ScoreSum: {score_sum}, RatingSum: {rating_sum}")
    if rating_sum <= 0 or score_sum <= 0:
        # If no one scored anything, no ratings change (Tie)
        print(f"[RatingLogic] ABORTING: Zero total score or rating.")
        return changes
        
    # K-factor approach: Change = K * (ActualRatio - ExpectedRatio)
    K = 40 

    for p in active_pool:
        the_rating = float(getattr(p, 'rating', 1200))
        the_score = float(getattr(p, 'score', 0))
        
        # Fair share of points based on relative rating
        expected_ratio = the_rating / rating_sum
        actual_ratio = the_score / score_sum
        
        # Calculate raw change: -20 to +20 range typically
        raw_change = K * (actual_ratio - expected_ratio)
        change = int(round(raw_change))
        
        # Clamp to -16/+16 as per existing behavior preference
        change = max(-16, min(16, change))
        
        changes[p.user_id] = change
        print(f"[RatingLogic] Player {getattr(p, 'username', 'Unknown')} ({p.user_id}): Score={the_score}, Rating={the_rating}, Actual={actual_ratio:.3f}, Expected={expected_ratio:.3f}, Change={change}")
            
    return changes
