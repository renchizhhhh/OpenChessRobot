"example input:"
"""
{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1",
    "move": "f3",
    "history": "",
    "centipawn": -62,
    "win": 44.317433903349645,
    "pv": "1. f3 e5 2. Nc3 Nc6 3. e3 d5 4. Bb5 Ne7 5. d4 a6 6. Bxc6+ Nxc6 7. dxe5 Be6 8. f4 Qh4+ 9. g3 Qd8 10. Nf3 Qd7 11. Qe2 O-O-O 12. Bd2 f6 13. exf6 gxf6 14. O-O-O Bc5 15. Be1 Rhe8 16. Ne4 Bxe3+ 17. Qxe3 Bg4 18. h3 Rxe4 19. Qxe4 dxe4 20. Rxd7 Bxd7 21. Nd2 Re8 22. Nf1 e3",
    "move_count": "7",
    "question": "can you analyse the game?"
}
"""
"example output:" 
"""
Okay. The Barnes Opening with f3 weakens White's king-side pawn structure and gives Black an opportunity for central dominance with 1... e5. The principal variation showcases Black's enhanced piece activity, while White struggles with king safety and cohesive development. This emphasizes the value of a robust opening strategy.
"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"example input": 
"""
{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1",
"move": "Nc3",
"history": "",
"centipawn": 0,
"win": 50.0,
"pv": "1. Nc3 d5 2. d4 Nf6 3. Bf4 e6 4. Nb5 Na6 5. e3 Be7 6. h4 O-O 7. Nf3 Bd7 8. c3 Ne4 9. Ng5 f5 10. Be5 Bf6 11. Bf4",
"move_count": "3",
"question": "can you analyse the game?"}

"""
"example output:" 
"""
Okay. This fairly uncommon opening Nc3 may have more names than any other: Dunst Opening, Van Geet Opening, Queen's Knight Opening. It develops the knight to a good square where it attacks the central e4 and d5 squares. However, it does not stop Black from occupying the center and blocks White's c-pawn without moving the knight first.
"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"example input": 
"""
{
    "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R_b_KQkq_-_1_2",
    "move": "d6",
    "history": "1. e4 c5 2. Nf3",
    "centipawn": 39,
    "win": 53.58387144707508,
    "pv": "2...d6 3. d4 cxd4 4. Nxd4 Nc6 5. Nc3 Nf6 6. Bg5 e6 7. Qd2 a6 8. O-O-O Bd7 9. f3 Qc7 10. Nxc6 Bxc6 11. Kb1 b5 12. h4 h6 13. Bxf6 gxf6",
    "move_count": "4",
    "question": "can you analyse the game?"
}
"""
"example output:" 
"""
Certainly. The game features the Sicilian Defense, a popular and aggressive response by Black. By playing 2...d6, Black supports the c5 pawn and prepares for e5 or Nf6 in the future. There will be a sharp battle around d4. However, there's a hint of leaving the king vulnerable. Such dynamics underline the complexity of the Sicilian Defense.
"""
---------------------------------------------------------------------------------------------------------------------------------------------------------


