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
    
    # Filter to only including competitive human players (non-bots, non-guests)
    # Bots are identified by p.is_ai == True
    competitive_humans = [
        p for p in players 
        if not getattr(p, 'is_ai', False) and not is_player_guest(p)
    ]
    
    if not competitive_humans:
        return changes

    # Count players with score >= 1 among competitive humans
    active_humans = [p for p in competitive_humans if getattr(p, 'score', 0) >= 1]
    number_of_players = len(active_humans)
            
    if number_of_players == 0:
        return changes
    
    # Sort active humans by score descending
    active_pool = sorted(active_humans, key=lambda x: getattr(x, 'score', 0), reverse=True)
    
    # Calculate scoreSum and ratingSum ONLY for competitive humans in the active pool
    score_sum = sum(getattr(p, 'score', 0) for p in active_pool)
    rating_sum = sum(getattr(p, 'rating', 1200) for p in active_pool)
    
    if rating_sum <= 0:
        return changes
        
    # K-factor approach: Change = K * (ActualRatio - ExpectedRatio)
    # This yields a more balanced distribution and naturally scales to multiple players.
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
