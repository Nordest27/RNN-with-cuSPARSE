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

	if (m.mov_t == NORMAL) {
		cb.board[m.pfi.i][m.pfi.j] = cb.board[m.pini.i][m.pini.j];
		cb.board[m.pini.i][m.pini.j] = 0;
	}
	else if (m.mov_t == EN_PASSANT)
	{
		cb.board[m.pfi.i][m.pfi.j] = cb.board[m.pini.i][m.pini.j];
		cb.board[m.pini.i][m.pini.j] = 0;
		cb.board[m.pini.i][m.pfi.j] = 0;
	}

	return cb;
}

