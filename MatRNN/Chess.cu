#include "Chess.cuh"

Position::Position()
{
	i = -1;
	j = -1;
}

Position::Position(int pi, int pj)
{
	i = pi;
	j = pj;
}

Position::Position(const Position& p)
{
	i = p.i;
	j = p.j;
}

void Position::sum(int si, int sj)
{
	i += si;
	j += sj;
}

Move::Move(Position pi, Position pf)
{
	pini = pi;
	pfi = pf;
	mov_t = NORMAL;
}

chess_board::chess_board()
{

}

chess_board::chess_board(const chess_board& cb)
{
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			board[i][j] = cb.board[i][j];
}

chess_board chess_board::move_piece(Move m)
{
	if (board[m.pini.i][m.pini.j] == 0)
		return *this;

	chess_board cb(*this);
	cb.board[m.pfi.i][m.pfi.j] = cb.board[m.pini.i][m.pini.j];
	cb.board[m.pini.i][m.pini.j] = 0;

	if (m.mov_t == EN_PASSANT)
		cb.board[m.pini.i][m.pfi.j] = 0;

	else if (m.mov_t == PROMQ)
		cb.board[m.pfi.i][m.pfi.j] = cb.board[m.pfi.i][m.pfi.j] * 5;

	else if (m.mov_t == PROMN)
		cb.board[m.pfi.i][m.pfi.j] = cb.board[m.pfi.i][m.pfi.j] * 3;

	else if (m.mov_t == PROMR)
		cb.board[m.pfi.i][m.pfi.j] = cb.board[m.pfi.i][m.pfi.j] * 2;

	else if (m.mov_t == PROMB)
		cb.board[m.pfi.i][m.pfi.j] = cb.board[m.pfi.i][m.pfi.j] * 4;

	else if (m.mov_t == CASTLK) {
		cb.board[m.pfi.i][m.pfi.j - 1] = cb.board[m.pfi.i][m.pfi.j + 1];
		cb.board[m.pfi.i][m.pfi.j + 1] = 0;
	}
	else if (m.mov_t == CASTLQ) {
		cb.board[m.pfi.i][m.pfi.j + 1] = cb.board[m.pfi.i][m.pfi.j - 1];
		cb.board[m.pfi.i][m.pfi.j - 1] = 0;
	}

	return cb;
}

void chess_board::print_board()
{
	for (int i = 0; i < 8; ++i)
		std::cout << "__";
	std::cout << std::endl;

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			std::cout << "|";
			if      (board[i][j] ==  0) std::cout << " ";
			else if (board[i][j] ==  1) std::cout << "p";
			else if (board[i][j] ==  2) std::cout << "r";
			else if (board[i][j] ==  3) std::cout << "n";
			else if (board[i][j] ==  4) std::cout << "b";
			else if (board[i][j] ==  5) std::cout << "q";
			else if (board[i][j] ==  6) std::cout << "k";
			else if (board[i][j] == -1) std::cout << "P";
			else if (board[i][j] == -2) std::cout << "R";
			else if (board[i][j] == -3) std::cout << "N";
			else if (board[i][j] == -4) std::cout << "B";
			else if (board[i][j] == -5) std::cout << "Q";
			else if (board[i][j] == -6) std::cout << "K";
		}
		std::cout << "|" << std::endl;
	}
	for (int i = 0; i < 8; ++i)
		std::cout << "__";
	std::cout << std::endl;

}

void chess_game::generate_semi_legal_moves()
{
	possible_moves.clear();
    generate_pawn_moves();
	generate_rook_moves();
	generate_bishop_moves();
	generate_queen_moves();
	generate_knight_moves();
	generate_king_moves();
}

void chess_game::generate_pawn_moves()
{

}

void chess_game::generate_rook_moves()
{

}

void chess_game::generate_bishop_moves()
{

}

void chess_game::generate_queen_moves()
{

}

void chess_game::generate_knight_moves()
{

}

void chess_game::generate_king_moves()
{

}

void chess_game::filter_ilegal_moves()
{

}

void chess_game::exectue_move(int i)
{
	if (i > possible_moves.size())
		return;

	b = b.move_piece(possible_moves[i]);
}

