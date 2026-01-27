
try:
    from board_generator import BoardGenerator
    from spinner_set import SpinnerSet
    
    print("Initializing Generator...")
    gen = BoardGenerator()
    
    dims = '4x4'
    print(f"Generating params for {dims}...")
    params = SpinnerSet.generate_params(dims)
    print(f"Params: {params}")
    
    print("Generating Board...")
    board, words = gen.generate_board(
        dims,
        "TESTWORD", # Bonus word
        params['word_count_range'],
        params['dictionary'],
        params['board_format'],
        params['min_word_length'],
        params['difficulty']
    )
    
    print(f"Board generated: {board is not None}")
    if board:
        for row in board:
            print(row)
    print(f"Word count: {len(words)}")

except Exception as e:
    import traceback
    traceback.print_exc()
